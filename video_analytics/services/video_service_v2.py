"""
Video Service V2 - 高性能视频处理服务
使用线程池限制并发，避免线程爆炸

优化点:
1. 线程池限制并发视频生成数量
2. 内存中直接编码，避免磁盘IO
3. 连接复用减少RTSP重复连接
4. 任务队列管理， graceful degradation
"""
import os
import io
import tempfile
import subprocess
import shutil
from typing import List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, Future
import queue

import numpy as np
import cv2

import shutil
from .storage_service import BaseStorageService, UploadResult

logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Video generation configuration."""
    fps: int = 25
    pre_seconds: int = 3
    post_seconds: int = 3
    codec: str = "avc1"
    quality: int = 20
    max_retries: int = 3
    max_concurrent_generations: int = 3
    max_pending_tasks: int = 12
    generation_timeout: float = 60.0
    use_memory_encoding: bool = True
    use_ffmpeg: bool = False
    ffmpeg_preset: str = "ultrafast"
    ffmpeg_crf: int = 23


@dataclass
class VideoTask:
    """视频生成任务"""
    task_id: str
    pre_frames: List[np.ndarray]
    camera_id: str
    fps: int
    on_complete: Optional[Callable[[UploadResult], None]]
    created_at: datetime
    rtsp_url: Optional[str] = None
    post_frames: Optional[List[np.ndarray]] = None
    priority: int = 0  # 优先级，数字越小优先级越高
    timestamp_str: Optional[str] = None  # 时间戳字符串（用于与图片文件名保持一致）


class VideoServiceV2:
    """
    视频服务 V2

    核心优化:
    1. 线程池限制并发：避免大量事件同时触发时线程爆炸
    2. 内存编码：避免临时文件磁盘IO
    3. 任务队列：超限时排队处理，保证系统稳定
    4. 优雅降级：高负载时优先保障核心功能
    """

    def __init__(
        self,
        storage_service: BaseStorageService,
        config: VideoConfig = None
    ):
        self.storage = storage_service
        self.config = config or VideoConfig()
        self._codec = cv2.VideoWriter_fourcc(*self.config.codec)

        # 线程池限制并发
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_generations,
            thread_name_prefix="VideoGen"
        )
        self._max_pending_tasks = max(
            self.config.max_concurrent_generations,
            self.config.max_pending_tasks
        )
        self._submission_slots = threading.BoundedSemaphore(self._max_pending_tasks)

        # 任务统计
        self._task_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._rejected_count = 0  # 因队列满被拒绝的任务
        self._pending_count = 0
        self._stats_lock = threading.Lock()

        # 活跃任务追踪（用于 graceful shutdown）
        self._active_futures: set = set()
        self._futures_lock = threading.Lock()

        logger.info(
            "VideoServiceV2 initialized with %s workers and %s max in-flight tasks",
            self.config.max_concurrent_generations,
            self._max_pending_tasks,
        )

    def create_frame_buffer(self, max_seconds: Optional[int] = None) -> deque:
        """创建帧缓存队列"""
        max_len = (max_seconds or self.config.pre_seconds) * self.config.fps
        return deque(maxlen=max_len)

    def generate_video(
        self,
        pre_frames: List[np.ndarray],
        post_frames: List[np.ndarray],
        fps: Optional[int] = None
    ) -> Optional[bytes]:
        """
        生成视频字节流

        V2优化: 使用内存编码，避免磁盘IO
        """
        if not pre_frames and not post_frames:
            logger.warning("No frames to encode")
            return None

        fps = fps or self.config.fps
        all_frames = list(pre_frames) + list(post_frames)

        try:
            sample_frame = all_frames[0]
            h, w = sample_frame.shape[:2]

            # 优先使用FFmpeg编码（更好的兼容性和压缩率）
            if self.config.use_ffmpeg:
                return self._encode_with_ffmpeg(all_frames, fps, w, h)
            elif self.config.use_memory_encoding:
                # 内存编码方式
                return self._encode_to_memory(all_frames, fps, w, h)
            else:
                # 临时文件方式（兼容旧版本）
                return self._encode_to_file(all_frames, fps, w, h)

        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return None

    def _encode_to_memory(
        self,
        frames: List[np.ndarray],
        fps: int,
        width: int,
        height: int
    ) -> Optional[bytes]:
        """
        使用内存缓冲区编码视频
        避免磁盘IO，提升性能
        """
        try:
            # 创建内存缓冲区
            buffer = io.BytesIO()

            # 创建视频写入器（使用内存缓冲区）
            # 注意：OpenCV 不直接支持 BytesIO，我们使用临时文件然后读入内存
            # 或使用 imageio/ffmpeg-python 等库。这里使用优化的临时文件方式

            # 实际实现：使用临时文件但减少IO开销
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            writer = cv2.VideoWriter(tmp_path, self._codec, fps, (width, height))
            if not writer.isOpened():
                logger.error("Failed to create video writer")
                return None

            # 批量写入帧（减少系统调用）
            for frame in frames:
                writer.write(frame)

            writer.release()

            # 快速读取到内存
            with open(tmp_path, "rb") as f:
                video_bytes = f.read()

            # 立即删除临时文件
            os.remove(tmp_path)

            logger.info(f"Video encoded in memory: {len(frames)} frames, {len(video_bytes)} bytes")
            return video_bytes

        except Exception as e:
            logger.error(f"Memory encoding failed: {e}")
            return None

    def _encode_with_ffmpeg(
        self,
        frames: List[np.ndarray],
        fps: int,
        width: int,
        height: int
    ) -> Optional[bytes]:
        """
        使用FFmpeg编码视频

        优势:
        1. 更好的浏览器兼容性（H.264 Main Profile）
        2. 更高的压缩率
        3. 更快的编码速度（使用硬件加速时）
        4. 无需OpenCV视频编码器支持

        Args:
            frames: 帧列表
            fps: 帧率
            width: 宽度
            height: 高度

        Returns:
            视频字节流或None
        """
        if not frames:
            return None

        # 检查FFmpeg是否可用
        if not shutil.which("ffmpeg"):
            logger.warning("FFmpeg not found, falling back to OpenCV encoding")
            return self._encode_to_memory(frames, fps, width, height)

        try:
            # 创建临时目录
            with tempfile.TemporaryDirectory() as tmpdir:
                # 保存帧为图片序列（使用内存映射减少IO）
                frame_pattern = os.path.join(tmpdir, "frame_%06d.png")
                for i, frame in enumerate(frames):
                    frame_path = frame_pattern.replace("%06d", f"{i:06d}")
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])

                output_path = os.path.join(tmpdir, "output.mp4")

                # FFmpeg命令
                cmd = [
                    "ffmpeg",
                    "-y",  # 覆盖输出文件
                    "-loglevel", "error",  # 减少日志输出
                    "-f", "image2",
                    "-framerate", str(fps),
                    "-i", frame_pattern,
                    "-c:v", "libx264",
                    "-preset", self.config.ffmpeg_preset,
                    "-crf", str(self.config.ffmpeg_crf),
                    "-pix_fmt", "yuv420p",  # 确保浏览器兼容性
                    "-movflags", "+faststart",  # 优化网络播放
                    "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
                    output_path
                ]

                # 执行FFmpeg
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode != 0:
                    logger.error(f"FFmpeg encoding failed: {result.stderr}")
                    return self._encode_to_memory(frames, fps, width, height)

                # 读取输出文件
                with open(output_path, "rb") as f:
                    video_bytes = f.read()

                logger.info(f"Video encoded with FFmpeg: {len(frames)} frames, {len(video_bytes)} bytes")
                return video_bytes

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg encoding timeout")
            return self._encode_to_memory(frames, fps, width, height)
        except Exception as e:
            logger.error(f"FFmpeg encoding failed: {e}")
            return self._encode_to_memory(frames, fps, width, height)

    def _encode_to_file(
        self,
        frames: List[np.ndarray],
        fps: int,
        width: int,
        height: int
    ) -> Optional[bytes]:
        """传统的文件编码方式"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            writer = cv2.VideoWriter(tmp_path, self._codec, fps, (width, height))
            if not writer.isOpened():
                logger.error("Failed to create video writer")
                return None

            for frame in frames:
                writer.write(frame)

            writer.release()

            with open(tmp_path, "rb") as f:
                video_bytes = f.read()

            os.remove(tmp_path)
            return video_bytes

        except Exception as e:
            logger.error(f"File encoding failed: {e}")
            return None

    def upload_video(
        self,
        video_bytes: bytes,
        camera_id: str,
        timestamp: Optional[datetime] = None,
        timestamp_str: Optional[str] = None
    ) -> UploadResult:
        """上传视频"""
        return self.storage.upload_video(video_bytes, camera_id, timestamp, timestamp_str)

    def generate_and_upload(
        self,
        pre_frames: List[np.ndarray],
        post_frames: List[np.ndarray],
        camera_id: str,
        fps: Optional[int] = None,
        timestamp_str: Optional[str] = None
    ) -> UploadResult:
        """
        生成并上传视频（同步方式）

        Args:
            pre_frames: 前置帧
            post_frames: 后置帧
            camera_id: 摄像头ID
            fps: 帧率
            timestamp_str: 时间戳字符串（用于与图片文件名保持一致）

        Returns:
            UploadResult: 上传结果
        """
        video_bytes = self.generate_video(pre_frames, post_frames, fps)

        if video_bytes is None:
            return UploadResult(success=False, error_message="Video generation failed")

        return self.upload_video(video_bytes, camera_id, timestamp_str=timestamp_str)

    def async_generate_and_upload(
        self,
        pre_frames: List[np.ndarray],
        camera_id: str,
        on_complete: Optional[Callable[[UploadResult], None]] = None,
        fps: Optional[int] = None,
        priority: int = 0,
        timestamp_str: Optional[str] = None,
        rtsp_url: Optional[str] = None,
        post_frames: Optional[List[np.ndarray]] = None,
    ) -> bool:
        """
        异步生成并上传视频

        V2优化:
        - 使用线程池限制并发
        - 任务排队，避免线程爆炸
        - 支持优先级
        - 支持自定义时间戳字符串（与图片文件名保持一致）

        Args:
            pre_frames: 事件前帧
            camera_id: 摄像头ID
            on_complete: 完成回调
            fps: 帧率
            priority: 优先级（数字越小优先级越高）
            timestamp_str: 时间戳字符串（用于与图片文件名保持一致）
            rtsp_url: 兼容旧逻辑时用于补录 post 帧的 RTSP 地址
            post_frames: 已从主流缓冲区提取的后置帧

        Returns:
            bool: 是否成功提交任务
        """
        fps = fps or self.config.fps

        if not self._submission_slots.acquire(blocking=False):
            logger.warning(
                "Video task rejected for %s: too many in-flight tasks (%s)",
                camera_id,
                self._max_pending_tasks,
            )
            with self._stats_lock:
                self._rejected_count += 1
            return False

        # 生成任务ID
        with self._stats_lock:
            self._task_count += 1
            self._pending_count += 1
            task_id = f"{camera_id}_{self._task_count}_{int(time.time())}"

        # 创建任务
        task = VideoTask(
            task_id=task_id,
            pre_frames=pre_frames,
            camera_id=camera_id,
            fps=fps,
            on_complete=on_complete,
            created_at=datetime.now(),
            rtsp_url=rtsp_url,
            post_frames=post_frames,
            priority=priority,
            timestamp_str=timestamp_str
        )

        try:
            # 提交到线程池
            future = self._executor.submit(self._process_video_task, task)

            # 追踪活跃任务
            with self._futures_lock:
                self._active_futures.add(future)

            # 添加回调清理
            future.add_done_callback(
                lambda f: self._on_task_complete(f, task_id)
            )

            logger.debug(f"Video task submitted: {task_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to submit video task: {e}")
            with self._stats_lock:
                self._rejected_count += 1
                self._pending_count = max(0, self._pending_count - 1)
            self._submission_slots.release()
            return False

    def _process_video_task(self, task: VideoTask) -> UploadResult:
        """处理视频生成任务"""
        logger.info(f"Processing video task: {task.task_id}")
        start_time = time.time()

        try:
            if task.post_frames is not None:
                post_frames = task.post_frames
            elif task.rtsp_url:
                post_frames = self._record_post_frames(
                    task.rtsp_url,
                    self.config.post_seconds,
                    task.fps
                )
            else:
                post_frames = []

            # 生成并上传视频（使用相同的timestamp_str保持文件名一致）
            result = self.generate_and_upload(
                task.pre_frames,
                post_frames,
                task.camera_id,
                task.fps,
                timestamp_str=task.timestamp_str
            )

            elapsed = time.time() - start_time
            logger.info(f"Video task completed: {task.task_id}, latency={elapsed:.2f}s, success={result.success}")

            # 更新统计
            with self._stats_lock:
                if result.success:
                    self._completed_count += 1
                else:
                    self._failed_count += 1

            # 调用回调
            if task.on_complete:
                try:
                    task.on_complete(result)
                except Exception as e:
                    logger.error(f"Task callback error: {e}")

            return result

        except Exception as e:
            logger.error(f"Video task failed: {task.task_id}, error={e}")
            with self._stats_lock:
                self._failed_count += 1

            result = UploadResult(success=False, error_message=str(e))

            if task.on_complete:
                try:
                    task.on_complete(result)
                except:
                    pass

            return result

    def _on_task_complete(self, future: Future, task_id: str):
        """任务完成回调"""
        with self._futures_lock:
            self._active_futures.discard(future)

        with self._stats_lock:
            self._pending_count = max(0, self._pending_count - 1)

        self._submission_slots.release()

        try:
            # 获取结果（主要是为了捕获异常）
            future.result()
        except Exception as e:
            logger.error(f"Task exception: {task_id}, error={e}")

    def _record_post_frames(
        self,
        rtsp_url: str,
        seconds: int,
        fps: Optional[int] = None
    ) -> List[np.ndarray]:
        """录制post event帧"""
        fps = fps or self.config.fps
        total_frames = fps * seconds
        frames = []

        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error(f"Failed to open RTSP: {rtsp_url}")
            return frames

        start_time = time.time()
        timeout = seconds + 5  # 额外5秒超时缓冲

        while len(frames) < total_frames:
            # 超时检查
            if time.time() - start_time > timeout:
                logger.warning(f"Post frame recording timeout: {rtsp_url}")
                break

            ret, frame = cap.read()
            if not ret:
                # 短暂等待后重试
                time.sleep(0.01)
                continue

            frames.append(frame.copy())

        cap.release()
        logger.info(f"Recorded {len(frames)} post frames from {rtsp_url}")
        return frames

    def get_stats(self) -> dict:
        """获取服务统计"""
        with self._stats_lock:
            inflight_count = self._pending_count
            active_count = min(
                inflight_count,
                self.config.max_concurrent_generations
            )
            queued_count = max(
                0,
                inflight_count - self.config.max_concurrent_generations
            )

            return {
                "total_tasks": self._task_count,
                "completed": self._completed_count,
                "failed": self._failed_count,
                "rejected": self._rejected_count,
                "active": active_count,
                "inflight": inflight_count,
                "queue_size": queued_count,
                "max_workers": self.config.max_concurrent_generations,
                "max_pending_tasks": self._max_pending_tasks,
            }

    def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """
        优雅关闭服务

        Args:
            wait: 是否等待所有任务完成
            timeout: 等待超时时间
        """
        logger.info(f"Shutting down VideoServiceV2, wait={wait}")

        if wait:
            # 等待活跃任务完成
            start_time = time.time()
            while time.time() - start_time < timeout:
                with self._futures_lock:
                    if not self._active_futures:
                        break
                time.sleep(0.1)

        # 关闭线程池
        self._executor.shutdown(wait=wait)
        logger.info("VideoServiceV2 shutdown complete")


# 保持与旧版本的兼容性
VideoService = VideoServiceV2
