"""
Video Analytics System V2 - 高性能版本
基于生产者-消费者模式重构的实时视频分析系统

主要优化:
1. 生产者-消费者解耦：帧读取和检测分离，检测不阻塞帧读取
2. 有界队列：防止内存无限增长
3. 线程池限制：视频生成使用线程池，避免线程爆炸
4. 独立检测器实例：每个流有独立的检测器，避免共享状态问题
5. 环形缓冲区：减少内存分配和拷贝

API接口:
- POST /set_fence: 设置围栏并启动检测流
- POST /delete_stream: 停止流并删除围栏
- GET /status: 获取系统状态（包含性能指标）
"""
import os
import sys
import time
import json
import threading
import subprocess
import queue
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple, Callable
from multiprocessing import Manager, Process
from dataclasses import dataclass, field

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import psutil
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from video_analytics.config.settings_v2 import AppConfigV2 as AppConfig
from video_analytics.config.runtime_files import ensure_runtime_file
from video_analytics.engines.factory import create_infer_engine
from video_analytics.engines.ultralytics_engine import YOLOV8_CLASSES, SAFETY_HELMET_CLASSES, SMOKE_FIRE_CLASSES
from video_analytics.detectors.intrusion_detector import IntrusionDetector, FenceRegion
from video_analytics.detectors.helmet_detector import HelmetDetector
from video_analytics.detectors.overcrowd_detector import OvercrowdDetector
from video_analytics.detectors.smokefire_detector import SmokeFireDetector
from video_analytics.detectors.base_detector import BaseDetector, DetectionResultBundle

# V2 组件
from video_analytics.core.stream_processor_v2 import StreamManagerV2, StreamConfig
from video_analytics.services.video_service_v2 import VideoServiceV2, VideoConfig
from video_analytics.services.storage_service import StorageServiceFactory
from video_analytics.services.alarm_service import AlarmServiceFactory
from video_analytics.utils.viz_utils import draw_text


# =========================
# 配置
# =========================
ENV_PATH = ensure_runtime_file(
    ".env",
    template_name=".env.example",
    env_var_name="VA_ENV_FILE",
)
app = Flask(__name__)
CORS(app)

# 全局状态
class SystemState:
    def __init__(self):
        self.stream_manager: Optional[StreamManagerV2] = None
        self.detectors: Dict[str, BaseDetector] = {}  # 检测器模板
        self.config: Optional[AppConfig] = None
        self.active_fence_streams: Dict[str, Dict] = {}
        self.fence_dict: Dict[str, List] = {}
        self.lock = threading.Lock()
        self.preview_state_manager = None
        self.preview_event_states = None
        self.preview_detection_states = None

        # 性能统计
        self.start_time = datetime.now()
        self.total_requests = 0

system_state = SystemState()


def ensure_preview_state_manager():
    """Lazily create the shared preview state manager in the main process only."""
    if system_state.preview_state_manager is None:
        system_state.preview_state_manager = Manager()
        system_state.preview_event_states = system_state.preview_state_manager.dict()
        system_state.preview_detection_states = system_state.preview_state_manager.dict()


# =========================
# 请求参数规范化
# =========================
def normalize_algorithm_type(value) -> Optional[str]:
    """Accept int/float/string algorithmType values and normalize to internal id."""
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value.isdigit():
            value = int(value)
        else:
            return None
    elif isinstance(value, float):
        if not value.is_integer():
            return None
        value = int(value)

    algo_map = {1: "1", 2: "2", 3: "3", 4: "4"}
    return algo_map.get(value)


def normalize_camera_id(value) -> Optional[str]:
    """Accept numeric or string cam_id and normalize to a trimmed string."""
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
    else:
        value = str(value).strip()

    return value or None


# =========================
# 检测器工厂函数
# =========================
def create_detector_factory(algo_type: str, config: AppConfig) -> Callable[[], BaseDetector]:
    """
    创建检测器工厂函数
    每个流调用工厂创建独立的检测器实例，避免共享状态问题
    """
    def factory() -> Optional[BaseDetector]:
        try:
            if algo_type == "1":
                # 闯入检测器
                engine = create_infer_engine(
                    model_path=config.model.person_model_path,
                    backend=config.model.backend,
                    input_size=config.model.input_size,
                    confidence=config.model.confidence,
                    iou=config.model.iou,
                    classes=YOLOV8_CLASSES,
                    device_id=config.model.device_id,
                    fp16=config.model.fp16
                )
                detector = IntrusionDetector(
                    engine=engine,
                    config={
                        "min_frames": config.detection.intrusion_min_frames,
                        "confidence": config.detection.intrusion_confidence,
                        "cooldown_seconds": config.detection.intrusion_cooldown,
                        "target_classes": [0]
                    }
                )
                # 如果有围栏配置，应用它
                return detector

            elif algo_type == "2":
                # 安全帽检测器
                person_engine = create_infer_engine(
                    model_path=config.model.person_model_path,
                    backend=config.model.backend,
                    input_size=config.model.input_size,
                    confidence=config.model.confidence,
                    classes=YOLOV8_CLASSES,
                    device_id=config.model.device_id
                )

                if config.model.helmet_model_path and os.path.exists(config.model.helmet_model_path):
                    helmet_engine = create_infer_engine(
                        model_path=config.model.helmet_model_path,
                        backend='ultralytics',
                        input_size=config.model.input_size,
                        confidence=0.3,
                        classes=SAFETY_HELMET_CLASSES,
                        device_id=config.model.device_id,
                        verbose=False
                    )
                else:
                    helmet_engine = person_engine

                return HelmetDetector(
                    person_engine=person_engine,
                    helmet_engine=helmet_engine,
                    config={
                        "min_frames": config.detection.helmet_min_frames,
                        "person_confidence": config.detection.helmet_confidence,
                        "helmet_confidence": 0.5,
                        "cooldown_seconds": config.detection.helmet_cooldown,
                        "crop_padding": config.detection.helmet_crop_padding
                    }
                )

            elif algo_type == "3":
                # 超员检测器
                engine = create_infer_engine(
                    model_path=config.model.person_model_path,
                    backend=config.model.backend,
                    input_size=config.model.input_size,
                    confidence=config.model.confidence,
                    classes=YOLOV8_CLASSES,
                    device_id=config.model.device_id
                )
                return OvercrowdDetector(
                    engine=engine,
                    config={
                        "max_people": config.detection.overcrowd_max_people,
                        "duration_threshold": config.detection.overcrowd_duration,
                        "cooldown_seconds": config.detection.overcrowd_cooldown,
                        "confidence": config.detection.overcrowd_confidence
                    }
                )

            elif algo_type == "4":
                # 烟火检测器
                engine = create_infer_engine(
                    model_path=config.model.smokefire_model_path,
                    backend=config.model.backend,
                    input_size=config.model.input_size,
                    confidence=config.model.confidence,
                    classes=SMOKE_FIRE_CLASSES,
                    device_id=config.model.device_id
                )
                return SmokeFireDetector(
                    engine=engine,
                    config={
                        "min_frames": config.detection.fire_min_frames,
                        "confidence": config.detection.fire_confidence,
                        "cooldown_seconds": config.detection.fire_cooldown,
                        "target_classes": [0, 1]  # smoke and fire
                    }
                )

            else:
                logger.error(f"Unknown algorithm type: {algo_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to create detector {algo_type}: {e}")
            return None

    return factory


# =========================
# 画框服务 (保持不变)
# =========================
class FFmpegCapture:
    """FFmpeg拉流捕获"""
    def __init__(self, rtsp_url: str, width: int, height: int, init_wait: float = 3.0):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.process = None
        self.start()
        if init_wait > 0:
            time.sleep(init_wait)

    def start(self):
        import sys
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-max_delay", "500000",
            "-probesize", "32",
            "-analyzeduration", "0",
            "-i", self.rtsp_url,
            "-an",
            "-c:v", "rawvideo",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1"
        ]

        kwargs = {}
        if sys.platform == 'win32':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # 忽略stderr避免阻塞
            bufsize=10**8,
            **kwargs
        )

    def read(self, retry: int = 3) -> Tuple[bool, Optional[np.ndarray]]:
        for attempt in range(retry):
            try:
                if self.process.poll() is not None:
                    return False, None

                raw = self.process.stdout.read(self.frame_size)

                if len(raw) == self.frame_size:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (self.height, self.width, 3)
                    ).copy()
                    return True, frame
                elif len(raw) == 0:
                    time.sleep(0.05)
                    continue
                else:
                    time.sleep(0.05)
                    continue
            except Exception as e:
                logger.error(f"FFmpegCapture read error: {e}")
                return False, None

        return False, None

    def release(self):
        if self.process:
            self.process.kill()
            self.process.wait()
            self.process = None

    def is_healthy(self) -> bool:
        return self.process is not None and self.process.poll() is None


def get_stream_size(rtsp_url: str) -> Tuple[int, int]:
    """获取视频流分辨率"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        rtsp_url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    w = data["streams"][0]["width"]
    h = data["streams"][0]["height"]
    return w, h


def rect_to_polygon(rect: Dict, default_area: Dict, frame_width: int, frame_height: int) -> List[List[int]]:
    """将矩形转换为多边形点坐标"""
    base_width = default_area.get("width", 960)
    base_height = default_area.get("height", 540)

    scale_x = frame_width / base_width
    scale_y = frame_height / base_height

    x = rect["x"] * scale_x
    y = rect["y"] * scale_y
    w = rect["width"] * scale_x
    h = rect["height"] * scale_y

    return [
        [int(x), int(y)],
        [int(x + w), int(y)],
        [int(x + w), int(y + h)],
        [int(x), int(y + h)]
    ]


def fence_worker(
    camera_id: str,
    rtsp_url: str,
    fence_area: List,
    output_host: str = "127.0.0.1",
    algorithm_type: Optional[str] = None,
    preview_event_states=None,
    preview_detection_states=None,
):
    """画框工作进程，使用有界队列解耦读帧和推流。"""
    import signal

    logger.info(f"[FenceWorker] 启动画框进程 camera_id={camera_id}")
    logger.info(f"[FenceWorker] 输入流: {rtsp_url}")

    stop_flag = threading.Event()
    frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
    pts = np.array(fence_area, np.int32)
    cap = None
    push_proc = None
    frame_counter = 0
    last_log_time = time.time()
    overlay_ttl_seconds = 5.0

    def draw_preview_overlay(frame: np.ndarray) -> np.ndarray:
        result = frame

        if preview_detection_states is not None:
            detection_state = preview_detection_states.get(camera_id)
            if detection_state:
                updated_at = float(detection_state.get("updated_at", 0))
                if time.time() - updated_at <= 1.5:
                    for det in detection_state.get("detections", []):
                        bbox = det.get("bbox", [])
                        if len(bbox) != 4:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        label = det.get("label", "")
                        confidence = det.get("confidence")
                        text = label
                        if confidence is not None:
                            text = f"{label} {confidence:.2f}"
                        try:
                            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            result = draw_text(
                                result,
                                text,
                                (x1, max(20, y1 - 8)),
                                color=(0, 255, 255),
                                font_scale=0.6,
                                thickness=2,
                                bg_color=(0, 0, 0),
                            )
                        except Exception:
                            continue

        if preview_event_states is not None:
            event_state = preview_event_states.get(camera_id)
            if event_state:
                updated_at = float(event_state.get("updated_at", 0))
                if time.time() - updated_at <= overlay_ttl_seconds:
                    event_label = event_state.get("event_label", "TRIGGERED")
                    confidence = event_state.get("confidence")
                    event_text = f"Triggered: {event_label}"
                    if confidence is not None:
                        event_text += f" ({confidence:.2f})"
                    result = draw_text(
                        result,
                        event_text,
                        (10, 30),
                        color=(0, 0, 255),
                        font_scale=0.8,
                        thickness=2,
                        bg_color=(0, 0, 0),
                    )

        return result

    def signal_handler(signum, frame):
        logger.info(f"[FenceWorker] 收到终止信号 {camera_id}")
        stop_flag.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    def release_capture():
        nonlocal cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None

    def open_capture():
        capture = cv2.VideoCapture(rtsp_url)
        if not capture.isOpened():
            raise RuntimeError(f"无法打开 RTSP 流: {rtsp_url}")
        return capture

    def start_ffmpeg(width: int, height: int, fps: int):
        rtsp_push = f"rtsp://{output_host}:554/Streaming/Channels/{camera_id}"
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            rtsp_push,
        ]

        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        logger.info(f"[FenceWorker] 启动FFmpeg推流到 {rtsp_push}")
        return subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            **kwargs,
        )

    def stop_ffmpeg():
        nonlocal push_proc
        if push_proc is None:
            return
        try:
            if push_proc.stdin:
                push_proc.stdin.close()
        except Exception:
            pass
        try:
            push_proc.kill()
            push_proc.wait(timeout=2)
        except Exception:
            pass
        finally:
            push_proc = None

    def restart_ffmpeg(width: int, height: int, fps: int):
        stop_ffmpeg()
        return start_ffmpeg(width, height, fps)

    try:
        cap = open_capture()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 25) or 25
        logger.info(f"[FenceWorker] 视频分辨率: {width}x{height}, FPS: {fps}")

        push_proc = start_ffmpeg(width, height, fps)
        logger.info(f"[FenceWorker] FFmpeg启动成功，PID={push_proc.pid}")

        def read_loop():
            nonlocal cap, frame_counter
            reconnect_count = 0
            consecutive_failures = 0

            while not stop_flag.is_set():
                if cap is None or not cap.isOpened():
                    try:
                        release_capture()
                        cap = open_capture()
                        consecutive_failures = 0
                    except Exception as exc:
                        reconnect_count += 1
                        logger.warning(
                            "[FenceWorker] %s 重连拉流失败(%s): %s",
                            camera_id,
                            reconnect_count,
                            exc,
                        )
                        time.sleep(min(2 ** min(reconnect_count, 3), 5))
                        continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 15:
                        logger.warning(f"[FenceWorker] {camera_id} 读取帧失败，准备重连")
                        release_capture()
                        time.sleep(min(2 ** min(reconnect_count, 3), 5))
                        reconnect_count += 1
                        consecutive_failures = 0
                    continue

                reconnect_count = 0
                consecutive_failures = 0
                frame_counter += 1

                try:
                    cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                    frame = draw_preview_overlay(frame)
                except Exception as exc:
                    logger.error(f"[FenceWorker] 画围栏失败: {exc}")

                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

        def push_loop():
            nonlocal push_proc, last_log_time

            while not stop_flag.is_set():
                try:
                    frame = frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if push_proc is None or push_proc.poll() is not None:
                    logger.warning(f"[FenceWorker] FFmpeg 推流进程已退出，重启 {camera_id}")
                    push_proc = restart_ffmpeg(width, height, fps)

                try:
                    push_proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    logger.warning(f"[FenceWorker] FFmpeg 推流中断，重启 {camera_id}")
                    push_proc = restart_ffmpeg(width, height, fps)
                    continue
                except Exception as exc:
                    logger.error(f"[FenceWorker] 推流异常: {exc}")
                    push_proc = restart_ffmpeg(width, height, fps)
                    continue

                now = time.time()
                if now - last_log_time >= 5:
                    logger.info(f"[FenceWorker] {camera_id} 运行正常，已推{frame_counter}帧")
                    last_log_time = now

        reader = threading.Thread(target=read_loop, name=f"FenceRead-{camera_id}", daemon=True)
        writer = threading.Thread(target=push_loop, name=f"FencePush-{camera_id}", daemon=True)
        reader.start()
        writer.start()

        while not stop_flag.is_set():
            time.sleep(1)

        reader.join(timeout=2)
        writer.join(timeout=2)

    except Exception as e:
        logger.error(f"[FenceWorker] 异常 {camera_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_flag.set()
        release_capture()
        stop_ffmpeg()
        logger.info(f"[FenceWorker] 停止 {camera_id}，共推流{frame_counter}帧")


def stop_fence_worker(camera_id: str):
    """停止画框进程"""
    with system_state.lock:
        info = system_state.active_fence_streams.pop(camera_id, None)
        system_state.fence_dict.pop(camera_id, None)

    if info:
        proc = info.get("process")
        if proc and proc.is_alive():
            # 先尝试优雅终止
            proc.terminate()
            proc.join(timeout=1.0)  # 减少等待时间
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=0.5)

        # 强制清理相关FFmpeg进程
        try:
            current_process = psutil.Process()
            current_pid = current_process.pid
            killed = 0
            for p in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if p.info["pid"] == current_pid:
                        continue
                    if p.info["name"] and "ffmpeg" in p.info["name"].lower():
                        cmdline = " ".join(p.info["cmdline"] or [])
                        if str(camera_id) in cmdline:
                            p.kill()
                            killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if killed > 0:
                logger.info(f"[FenceWorker] 清理了{killed}个FFmpeg进程 {camera_id}")
        except Exception as e:
            logger.warning(f"[FenceWorker] 清理FFmpeg进程时出错: {e}")

        logger.info(f"[FenceWorker] 已停止 {camera_id}")


# =========================
# 系统初始化
# =========================
def load_config(path: str = str(ENV_PATH)) -> AppConfig:
    """Load config from .env file."""
    logger.info("[Config] Loading V2 config from %s", path)
    return AppConfig.from_env_file(path)


def get_rtsp_push_host() -> str:
    """Return the configured RTSP push host."""
    if system_state.config and system_state.config.server.rtsp_push_host:
        return system_state.config.server.rtsp_push_host
    return "127.0.0.1"


def get_algorithm_display_name(algo_type: str) -> str:
    """Return a user-friendly algorithm name for preview overlay."""
    return {
        "1": "Intrusion",
        "2": "Helmet",
        "3": "Overcrowd",
        "4": "Smoke/Fire",
    }.get(algo_type, f"Algo-{algo_type}")


def handle_preview_event(camera_id: str, event, result: DetectionResultBundle):
    """Store the latest triggered event so preview overlay can show it."""
    if system_state.preview_event_states is None:
        return
    system_state.preview_event_states[camera_id] = {
        "event_type": event.event_type.value,
        "event_label": event.event_type.value.replace("_", " ").upper(),
        "confidence": round(event.confidence, 2) if event.confidence is not None else None,
        "updated_at": time.time(),
    }


def handle_preview_detections(camera_id: str, result: DetectionResultBundle):
    """Store latest visible detections for preview overlay."""
    if system_state.preview_detection_states is None:
        return

    detections = []
    for det in result.detections or []:
        try:
            detections.append({
                "bbox": [int(det.x1), int(det.y1), int(det.x2), int(det.y2)],
                "label": str(det.class_name or det.class_id),
                "confidence": round(float(det.conf), 2),
            })
        except Exception:
            continue

    system_state.preview_detection_states[camera_id] = {
        "detections": detections,
        "updated_at": time.time(),
    }


def initialize_system() -> Tuple[StreamManagerV2, Dict]:
    """初始化系统组件 V2"""
    ensure_preview_state_manager()
    logger.info("=" * 60)
    logger.info("[System] Initializing Video Analytics System V2")
    logger.info("[System] Optimizations: Producer-Consumer + ThreadPool + CircularBuffer")
    logger.info("=" * 60)

    config = load_config()
    system_state.config = config

    # 创建服务
    logger.info("[System] Creating services...")

    storage_service = StorageServiceFactory.create(
        service_type=config.storage.type,
        endpoint=config.storage.endpoint,
        access_key=config.storage.access_key,
        secret_key=config.storage.secret_key,
        secure=config.storage.secure,
        bucket_name=config.storage.bucket_name,
        public_url=config.storage.public_url,
    )

    alarm_service = AlarmServiceFactory.create(
        service_type=config.alarm.type,
        endpoints=config.alarm.endpoints,
        endpoints_intrusion=config.alarm.endpoints_intrusion,
        endpoints_helmet=config.alarm.endpoints_helmet,
        endpoints_overcrowd=config.alarm.endpoints_overcrowd,
        endpoints_smokefire=config.alarm.endpoints_smokefire,
        timeout=config.alarm.timeout,
        retry_count=config.alarm.retry_count
    )

    # V2 视频服务（使用线程池）
    video_service = VideoServiceV2(
        storage_service=storage_service,
        config=VideoConfig(
            fps=config.stream.fps,
            pre_seconds=config.stream.pre_buffer_seconds,
            post_seconds=config.stream.post_record_seconds,
            max_concurrent_generations=3,  # 限制并发视频生成
            max_pending_tasks=12,
            use_memory_encoding=True
        )
    )

    # V2 流管理器
    logger.info("[System] Creating stream manager V2...")
    stream_manager = StreamManagerV2(
        storage_service=storage_service,
        alarm_service=alarm_service,
        video_service=video_service
    )
    stream_manager.set_event_callback(handle_preview_event)
    stream_manager.set_result_callback(handle_preview_detections)

    # 注册检测器工厂（每个流独立实例）
    logger.info("[System] Registering detector factories...")
    for algo_type in ["1", "2", "3", "4"]:
        factory = create_detector_factory(algo_type, config)
        stream_manager.register_detector_factory(algo_type, factory)

    logger.info("[System] Initialization complete!")
    logger.info("=" * 60)

    return stream_manager, {}


# =========================
# Flask API 路由
# =========================

@app.route("/")
def index():
    """根路由"""
    return jsonify({
        "status": "running",
        "service": "Video Analytics V2 (High Performance)",
        "version": "2.0",
        "optimizations": [
            "Producer-Consumer Architecture",
            "Bounded Frame Queue",
            "Circular Frame Buffer",
            "ThreadPool for Video Generation",
            "Independent Detector Instances"
        ]
    })


@app.route("/set_fence", methods=["POST"])
def set_fence():
    """
    设置围栏并启动检测流

    V2优化:
    - 独立检测器实例，避免线程安全问题
    - 有界队列防止内存无限增长
    - 环形缓冲区减少内存拷贝
    """
    system_state.total_requests += 1
    data = request.get_json() or {}

    algorithm_type = data.get("algorithmType")
    rtsp_url = data.get("url")
    rect = data.get("fence_area")
    default_area = data.get("default_area", {"width": 960, "height": 540})
    camera_id = normalize_camera_id(data.get("cam_id"))
    algo_type_str = normalize_algorithm_type(algorithm_type)

    if not all([algo_type_str, rtsp_url, camera_id]):
        return jsonify({
            "status": "error",
            "message": "缺少必要参数: algorithmType, url, cam_id"
        }), 400

    if not algo_type_str:
        return jsonify({
            "status": "error",
            "message": f"无效的 algorithmType: {algorithm_type}"
        }), 400

    logger.info(f"[API] 请求启动 camera_id={camera_id}, algo={algo_type_str}")

    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return jsonify({
                "status": "error",
                "message": "无法打开RTSP流"
            }), 500

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # 等待确保摄像头连接完全释放，避免fence_worker读取重复帧
        time.sleep(1.0)

        if not width or not height:
            return jsonify({
                "status": "error",
                "message": "无法获取视频分辨率"
            }), 500

        logger.info(f"[API] 视频分辨率: {width}x{height}")

        if rect:
            fence_area = rect_to_polygon(rect, default_area, width, height)
        else:
            fence_area = [[0, 0], [width, 0], [width, height], [0, height]]

        logger.info(f"[API] 围栏坐标: {fence_area}")

        with system_state.lock:
            has_existing_stream = camera_id in system_state.active_fence_streams

        if has_existing_stream:
            logger.info(f"[API] 摄像头 {camera_id} 已存在，停止旧流")
            stop_fence_worker(camera_id)
            system_state.stream_manager.remove_stream(camera_id)
            time.sleep(0.5)  # 增加等待时间确保资源完全释放

        config = system_state.config
        stream_config = StreamConfig(
            camera_id=camera_id,
            rtsp_url=rtsp_url,
            ip_address="",
            algorithm_types={algo_type_str},
            fps=config.stream.fps,
            skip_frames=config.stream.skip_frames,
            max_reconnect=config.stream.max_reconnect,
            pre_buffer_seconds=config.stream.pre_buffer_seconds,
            post_buffer_seconds=config.stream.post_record_seconds,
            enable_display=False,
            frame_queue_size=10,
            detection_queue_size=5,
            event_queue_size=20,
        )

        success = system_state.stream_manager.add_stream(stream_config)
        if not success:
            return jsonify({
                "status": "error",
                "message": "启动检测流失败"
            }), 500

        logger.info(f"[API] 检测流已启动 camera_id={camera_id}")

        try:
            system_state.stream_manager.set_detector_fence(camera_id, fence_area)

            output_host = get_rtsp_push_host()

            p = Process(
                target=fence_worker,
                args=(
                    camera_id,
                    rtsp_url,
                    fence_area,
                    output_host,
                    algo_type_str,
                    system_state.preview_event_states,
                    system_state.preview_detection_states,
                ),
                daemon=True
            )
            p.start()
        except Exception:
            system_state.stream_manager.remove_stream(camera_id)
            raise

        with system_state.lock:
            system_state.active_fence_streams[camera_id] = {
                "process": p,
                "url": rtsp_url,
                "algorithm_type": algo_type_str,
                "fence_area": fence_area
            }
            system_state.fence_dict[camera_id] = fence_area

        output_host = get_rtsp_push_host()

        output_url = f"rtsp://{output_host}:554/Streaming/Channels/{camera_id}"
        logger.info(f"[API] 画框流输出地址: {output_url}")

        return jsonify({
            "status": "success",
            "camera_id": camera_id,
            "algorithm_type": algo_type_str,
            "output_url": output_url,
            "detection_started": True,
            "version": "v2"
        })

    except Exception as e:
        logger.exception(f"处理请求失败: {e}")
        return jsonify({
            "status": "error",
            "message": f"处理请求失败: {str(e)}"
        }), 500


@app.route("/delete_stream", methods=["POST"])
def delete_stream():
    """停止流并删除围栏"""
    system_state.total_requests += 1
    data = request.get_json() or {}
    camera_id = normalize_camera_id(data.get("cam_id"))

    if not camera_id:
        return jsonify({
            "status": "error",
            "message": "cam_id 不能为空"
        }), 400

    try:
        if system_state.preview_event_states is not None:
            system_state.preview_event_states.pop(camera_id, None)
        if system_state.preview_detection_states is not None:
            system_state.preview_detection_states.pop(camera_id, None)
        # stop_fence_worker 内部已处理锁，不要在外层加锁
        stop_fence_worker(camera_id)
        if system_state.stream_manager:
            system_state.stream_manager.remove_stream(camera_id)

        logger.info(f"[API] 已删除摄像头 {camera_id}")

        return jsonify({
            "status": "success",
            "camera_id": camera_id
        })

    except Exception as e:
        logger.exception(f"删除流失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/status", methods=["GET"])
def get_status():
    """获取系统状态（V2增强版，包含性能指标）"""
    with system_state.lock:
        fence_streams = list(system_state.active_fence_streams.keys())

        # V2: 获取详细的流统计
        detection_stats = {}
        if system_state.stream_manager:
            detection_stats = system_state.stream_manager.get_all_stats()

        # V2: 获取视频服务统计
        video_stats = {}
        if system_state.stream_manager and hasattr(system_state.stream_manager.video, 'get_stats'):
            video_stats = system_state.stream_manager.video.get_stats()

        # 系统运行时间
        uptime = (datetime.now() - system_state.start_time).total_seconds()

    return jsonify({
        "status": "running",
        "version": "v2",
        "uptime_seconds": int(uptime),
        "total_requests": system_state.total_requests,
        "active_fence_streams": fence_streams,
        "stream_count": len(fence_streams),
        "detection_stats": detection_stats,
        "video_service_stats": video_stats,
        "optimizations": {
            "architecture": "Producer-Consumer",
            "frame_queue": "Bounded (10 frames)",
            "buffer": "Circular (reduced memory copy)",
            "video_generation": "ThreadPool (max 3 concurrent)",
            "detector_instances": "Independent per stream"
        }
    })


@app.route("/performance", methods=["GET"])
def get_performance():
    """获取详细的性能指标"""
    stats = {}
    video_service_stats = {}
    if system_state.stream_manager:
        if hasattr(system_state.stream_manager.video, 'get_stats'):
            video_service_stats = system_state.stream_manager.video.get_stats()

        for cam_id, cam_stats in system_state.stream_manager.get_all_stats().items():
            stats[cam_id] = {
                "state": cam_stats.get("state", "unknown"),
                "fps": cam_stats.get("fps", 0),
                "detection_fps": cam_stats.get("detection_fps", 0),
                "frame_count": cam_stats.get("frame_count", 0),
                "detection_count": cam_stats.get("detection_count", 0),
                "event_count": cam_stats.get("event_count", 0),
                "dropped_frames": cam_stats.get("dropped_frames", 0),
                "dropped_events": cam_stats.get("dropped_events", 0),
                "dropped_video_tasks": cam_stats.get("dropped_video_tasks", 0),
                "avg_detection_latency_ms": cam_stats.get("avg_detection_latency_ms", 0),
                "error_count": cam_stats.get("error_count", 0),
                "run_time": cam_stats.get("run_time", 0),
                "queues": {
                    "frame_queue": cam_stats.get("queues", {}).get("frame_queue", {}),
                    "event_queue": cam_stats.get("queues", {}).get("event_queue", {}),
                    "video_prepare_queue": cam_stats.get("queues", {}).get("video_prepare_queue", {}),
                },
                "frame_buffer": cam_stats.get("frame_buffer", {}),
            }

    return jsonify({
        "streams": stats,
        "video_service": video_service_stats,
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "uptime": int((datetime.now() - system_state.start_time).total_seconds())
        }
    })


# =========================
# 信号处理
# =========================
def force_exit(signum, frame):
    """强制退出处理"""
    logger.info("[Main] FORCE EXIT...")
    try:
        current_process = psutil.Process()
        for child in current_process.children(recursive=True):
            try:
                child.kill()
            except:
                pass
    except:
        pass
    os._exit(1)


import signal
signal.signal(signal.SIGINT, force_exit)
signal.signal(signal.SIGTERM, force_exit)


# =========================
# 主程序
# =========================
def main():
    """主函数"""
    stream_manager, _ = initialize_system()
    system_state.stream_manager = stream_manager

    host = "0.0.0.0"
    port = 5005
    debug = False
    if system_state.config:
        host = system_state.config.server.host
        port = system_state.config.server.port
        debug = system_state.config.server.debug

    logger.info(f"[Main] Starting API server V2 on port {port}")
    logger.info(f"[Main] API endpoints:")
    logger.info(f"  - POST http://{host}:{port}/set_fence")
    logger.info(f"  - POST http://{host}:{port}/delete_stream")
    logger.info(f"  - GET  http://{host}:{port}/status")
    logger.info(f"  - GET  http://{host}:{port}/performance")
    logger.info("=" * 60)

    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
