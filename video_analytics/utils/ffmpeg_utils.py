"""
FFmpeg工具函数
统一的FFmpeg进程管理和视频流处理
"""
import subprocess
import json
import logging
import numpy as np
from typing import Tuple, Optional, List, Dict, Callable
from dataclasses import dataclass
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class FFmpegConfig:
    """FFmpeg配置"""
    rtsp_transport: str = "tcp"
    timeout_ms: int = 1000000
    max_delay_ms: int = 300000
    probe_size: int = 32
    analyze_duration: int = 0
    buffer_size: int = 10**8
    reconnect: bool = True
    reconnect_delay_sec: int = 1


def get_stream_info(rtsp_url: str, timeout: float = 5.0) -> Optional[Dict]:
    """
    使用ffprobe获取视频流信息

    Args:
        rtsp_url: RTSP流地址
        timeout: 超时时间(秒)

    Returns:
        Dict: 包含 width, height, fps 等信息，失败返回None
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json",
            rtsp_url
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            logger.error(f"ffprobe failed: {result.stderr}")
            return None

        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]

        # 解析帧率
        fps_str = stream.get("r_frame_rate", "0/1")
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den != 0 else 0

        return {
            "width": stream.get("width", 0),
            "height": stream.get("height", 0),
            "fps": fps
        }

    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe timeout for {rtsp_url}")
        return None
    except Exception as e:
        logger.error(f"ffprobe error: {e}")
        return None


def build_ffmpeg_capture_cmd(
    rtsp_url: str,
    width: int,
    height: int,
    config: Optional[FFmpegConfig] = None
) -> List[str]:
    """
    构建FFmpeg捕获命令

    Args:
        rtsp_url: RTSP流地址
        width: 视频宽度
        height: 视频高度
        config: FFmpeg配置

    Returns:
        List[str]: 命令参数列表
    """
    config = config or FFmpegConfig()

    return [
        "ffmpeg",
        "-loglevel", "error",
        "-rtsp_transport", config.rtsp_transport,
        "-fflags", "nobuffer+discardcorrupt",
        "-flags", "low_delay",
        "-max_delay", str(config.max_delay_ms),
        "-probesize", str(config.probe_size),
        "-analyzeduration", str(config.analyze_duration),
        "-stimeout", str(config.timeout_ms),
        "-i", rtsp_url,
        "-an",  # 禁用音频
        "-c:v", "rawvideo",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1"
    ]


def build_ffmpeg_push_cmd(
    output_url: str,
    width: int,
    height: int,
    fps: int = 25,
    codec: str = "libx264",
    preset: str = "ultrafast",
    tune: str = "zerolatency"
) -> List[str]:
    """
    构建FFmpeg推流命令

    Args:
        output_url: 输出RTSP地址
        width: 视频宽度
        height: 视频高度
        fps: 帧率
        codec: 视频编码器
        preset: 编码预设
        tune: 编码调优

    Returns:
        List[str]: 命令参数列表
    """
    return [
        "ffmpeg",
        "-loglevel", "error",
        "-fflags", "+genpts+discardcorrupt",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",  # 从stdin读取
        "-an",  # 禁用音频
        "-c:v", codec,
        "-preset", preset,
        "-tune", tune,
        "-g", str(fps),  # GOP大小
        "-keyint_min", str(fps),
        "-sc_threshold", "0",
        "-pix_fmt", "yuv420p",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        output_url
    ]


class FFmpegCapture:
    """
    FFmpeg视频捕获器
    封装FFmpeg进程管理
    """

    def __init__(
        self,
        rtsp_url: str,
        width: int,
        height: int,
        config: Optional[FFmpegConfig] = None,
        init_wait: float = 3.0
    ):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.config = config or FFmpegConfig()
        self.process: Optional[subprocess.Popen] = None

        self._stop_event = threading.Event()
        self._frame_count = 0
        self._error_count = 0

        self.start()

        if init_wait > 0:
            time.sleep(init_wait)

    def start(self):
        """启动FFmpeg进程"""
        cmd = build_ffmpeg_capture_cmd(
            self.rtsp_url,
            self.width,
            self.height,
            self.config
        )

        logger.info(f"Starting FFmpeg capture: {self.rtsp_url}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.config.buffer_size
            )
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            raise

    def read(self, retry: int = 3) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧

        Args:
            retry: 重试次数

        Returns:
            (success, frame): 是否成功和帧数据
        """
        for attempt in range(retry):
            try:
                # 检查进程状态
                if self.process is None or self.process.poll() is not None:
                    return False, None

                # 读取原始数据
                raw = self.process.stdout.read(self.frame_size)

                if len(raw) == self.frame_size:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (self.height, self.width, 3)
                    ).copy()
                    self._frame_count += 1
                    return True, frame
                elif len(raw) == 0:
                    # 进程可能已结束
                    time.sleep(0.05)
                    continue
                else:
                    # 数据不完整
                    self._error_count += 1
                    time.sleep(0.05)
                    continue

            except Exception as e:
                logger.error(f"FFmpeg read error: {e}")
                return False, None

        return False, None

    def is_healthy(self) -> bool:
        """检查进程是否健康"""
        return self.process is not None and self.process.poll() is None

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "frame_count": self._frame_count,
            "error_count": self._error_count,
            "healthy": self.is_healthy()
        }

    def restart(self):
        """重启FFmpeg进程"""
        logger.warning("Restarting FFmpeg capture")
        self.release()
        time.sleep(1)
        self.start()

    def release(self):
        """释放资源"""
        if self.process:
            try:
                self.process.kill()
                self.process.wait(timeout=2)
            except:
                pass
            finally:
                self.process = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class FFmpegPusher:
    """
    FFmpeg视频推流器
    """

    def __init__(
        self,
        output_url: str,
        width: int,
        height: int,
        fps: int = 25
    ):
        self.output_url = output_url
        self.width = width
        self.height = height
        self.fps = fps
        self.process: Optional[subprocess.Popen] = None

        self._frame_count = 0
        self._error_count = 0

        self.start()

    def start(self):
        """启动推流进程"""
        cmd = build_ffmpeg_push_cmd(
            self.output_url,
            self.width,
            self.height,
            self.fps
        )

        logger.info(f"Starting FFmpeg push: {self.output_url}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0
            )
        except Exception as e:
            logger.error(f"Failed to start FFmpeg pusher: {e}")
            raise

    def write(self, frame: np.ndarray) -> bool:
        """
        写入一帧

        Args:
            frame: 帧数据

        Returns:
            bool: 是否成功
        """
        if not self.is_healthy():
            return False

        try:
            self.process.stdin.write(frame.tobytes())
            self._frame_count += 1
            return True
        except BrokenPipeError:
            logger.error("FFmpeg pusher broken pipe")
            self._error_count += 1
            return False
        except Exception as e:
            logger.error(f"FFmpeg write error: {e}")
            self._error_count += 1
            return False

    def is_healthy(self) -> bool:
        """检查进程是否健康"""
        return self.process is not None and self.process.poll() is None

    def restart(self):
        """重启推流进程"""
        logger.warning("Restarting FFmpeg pusher")
        self.release()
        time.sleep(0.5)
        self.start()

    def release(self):
        """释放资源"""
        if self.process:
            try:
                self.process.stdin.close()
            except:
                pass
            try:
                self.process.kill()
                self.process.wait(timeout=2)
            except:
                pass
            finally:
                self.process = None

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "frame_count": self._frame_count,
            "error_count": self._error_count,
            "healthy": self.is_healthy()
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def kill_ffmpeg_processes(pattern: Optional[str] = None):
    """
    终止FFmpeg进程

    Args:
        pattern: 可选的匹配模式
    """
    import psutil

    killed = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "ffmpeg" in proc.info["name"].lower():
                if pattern is None or pattern in " ".join(proc.info["cmdline"] or []):
                    proc.kill()
                    killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if killed > 0:
        logger.info(f"Killed {killed} FFmpeg processes")
