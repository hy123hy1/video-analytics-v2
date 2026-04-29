"""
自定义异常类
统一的异常体系，便于错误处理和监控
"""


class VideoAnalyticsError(Exception):
    """基础异常类"""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"[{self.error_code}] {self.message} - Details: {self.details}"
        return f"[{self.error_code}] {self.message}"


# ==================== 推理相关异常 ====================

class InferenceError(VideoAnalyticsError):
    """推理错误"""

    def __init__(self, message: str, model_name: str = None, details: dict = None):
        super().__init__(
            message=message,
            error_code="INFERENCE_ERROR",
            details={"model": model_name, **(details or {})}
        )


class ModelLoadError(InferenceError):
    """模型加载错误"""

    def __init__(self, message: str, model_path: str = None):
        super().__init__(
            message=message,
            error_code="MODEL_LOAD_ERROR",
            details={"model_path": model_path}
        )


class ModelWarmupError(InferenceError):
    """模型预热错误"""

    def __init__(self, message: str, model_name: str = None):
        super().__init__(
            message=message,
            error_code="MODEL_WARMUP_ERROR",
            details={"model": model_name}
        )


# ==================== 流处理相关异常 ====================

class StreamError(VideoAnalyticsError):
    """流处理错误"""

    def __init__(self, message: str, camera_id: str = None, details: dict = None):
        super().__init__(
            message=message,
            error_code="STREAM_ERROR",
            details={"camera_id": camera_id, **(details or {})}
        )


class StreamConnectionError(StreamError):
    """流连接错误"""

    def __init__(self, message: str, camera_id: str = None, rtsp_url: str = None):
        super().__init__(
            message=message,
            error_code="STREAM_CONNECTION_ERROR",
            details={"camera_id": camera_id, "rtsp_url": rtsp_url}
        )


class StreamReadError(StreamError):
    """流读取错误"""

    def __init__(self, message: str, camera_id: str = None, retry_count: int = 0):
        super().__init__(
            message=message,
            error_code="STREAM_READ_ERROR",
            details={"camera_id": camera_id, "retry_count": retry_count}
        )


class StreamTimeoutError(StreamError):
    """流超时错误"""

    def __init__(self, message: str, camera_id: str = None, timeout: float = None):
        super().__init__(
            message=message,
            error_code="STREAM_TIMEOUT_ERROR",
            details={"camera_id": camera_id, "timeout": timeout}
        )


# ==================== 检测相关异常 ====================

class DetectionError(VideoAnalyticsError):
    """检测错误"""

    def __init__(self, message: str, camera_id: str = None, algo_type: str = None):
        super().__init__(
            message=message,
            error_code="DETECTION_ERROR",
            details={"camera_id": camera_id, "algorithm": algo_type}
        )


class DetectorNotFoundError(DetectionError):
    """检测器未找到错误"""

    def __init__(self, algo_type: str):
        super().__init__(
            message=f"Detector for algorithm '{algo_type}' not found",
            error_code="DETECTOR_NOT_FOUND",
            details={"algorithm": algo_type}
        )


class FenceConfigError(DetectionError):
    """围栏配置错误"""

    def __init__(self, message: str, camera_id: str = None):
        super().__init__(
            message=message,
            error_code="FENCE_CONFIG_ERROR",
            details={"camera_id": camera_id}
        )


# ==================== 存储相关异常 ====================

class StorageError(VideoAnalyticsError):
    """存储错误"""

    def __init__(self, message: str, service_type: str = None, details: dict = None):
        super().__init__(
            message=message,
            error_code="STORAGE_ERROR",
            details={"service": service_type, **(details or {})}
        )


class UploadError(StorageError):
    """上传错误"""

    def __init__(self, message: str, file_type: str = None, camera_id: str = None):
        super().__init__(
            message=message,
            error_code="UPLOAD_ERROR",
            details={"file_type": file_type, "camera_id": camera_id}
        )


class MinIOConnectionError(StorageError):
    """MinIO连接错误"""

    def __init__(self, message: str, endpoint: str = None):
        super().__init__(
            message=message,
            error_code="MINIO_CONNECTION_ERROR",
            details={"endpoint": endpoint}
        )


# ==================== 报警相关异常 ====================

class AlarmError(VideoAnalyticsError):
    """报警错误"""

    def __init__(self, message: str, endpoint: str = None):
        super().__init__(
            message=message,
            error_code="ALARM_ERROR",
            details={"endpoint": endpoint}
        )


class AlarmSendError(AlarmError):
    """报警发送错误"""

    def __init__(self, message: str, endpoint: str = None, retry_count: int = 0):
        super().__init__(
            message=message,
            error_code="ALARM_SEND_ERROR",
            details={"endpoint": endpoint, "retry_count": retry_count}
        )


# ==================== 配置相关异常 ====================

class ConfigError(VideoAnalyticsError):
    """配置错误"""

    def __init__(self, message: str, config_key: str = None):
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            details={"config_key": config_key}
        )


class ConfigLoadError(ConfigError):
    """配置加载错误"""

    def __init__(self, message: str, config_path: str = None):
        super().__init__(
            message=message,
            error_code="CONFIG_LOAD_ERROR",
            details={"config_path": config_path}
        )


class ConfigValidationError(ConfigError):
    """配置验证错误"""

    def __init__(self, message: str, config_key: str = None, invalid_value: any = None):
        super().__init__(
            message=message,
            error_code="CONFIG_VALIDATION_ERROR",
            details={"config_key": config_key, "invalid_value": str(invalid_value)}
        )


# ==================== 视频服务相关异常 ====================

class VideoServiceError(VideoAnalyticsError):
    """视频服务错误"""

    def __init__(self, message: str, camera_id: str = None):
        super().__init__(
            message=message,
            error_code="VIDEO_SERVICE_ERROR",
            details={"camera_id": camera_id}
        )


class VideoGenerationError(VideoServiceError):
    """视频生成错误"""

    def __init__(self, message: str, camera_id: str = None, frame_count: int = 0):
        super().__init__(
            message=message,
            error_code="VIDEO_GENERATION_ERROR",
            details={"camera_id": camera_id, "frame_count": frame_count}
        )


class VideoEncodingError(VideoServiceError):
    """视频编码错误"""

    def __init__(self, message: str, codec: str = None):
        super().__init__(
            message=message,
            error_code="VIDEO_ENCODING_ERROR",
            details={"codec": codec}
        )


# ==================== 工具函数 ====================

def is_retryable_error(error: Exception) -> bool:
    """
    判断错误是否可重试

    Args:
        error: 异常对象

    Returns:
        bool: 是否可重试
    """
    retryable_errors = (
        StreamConnectionError,
        StreamReadError,
        StreamTimeoutError,
        MinIOConnectionError,
        AlarmSendError,
    )
    return isinstance(error, retryable_errors)


def get_error_level(error: Exception) -> str:
    """
    获取错误级别（用于监控和报警）

    Args:
        error: 异常对象

    Returns:
        str: 错误级别 (CRITICAL, ERROR, WARNING)
    """
    critical_errors = (
        ModelLoadError,
        ConfigLoadError,
    )

    warning_errors = (
        StreamTimeoutError,
        AlarmSendError,
    )

    if isinstance(error, critical_errors):
        return "CRITICAL"
    elif isinstance(error, warning_errors):
        return "WARNING"
    else:
        return "ERROR"
