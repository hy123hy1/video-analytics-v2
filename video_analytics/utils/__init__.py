"""
工具模块

提供日志、异常、YOLO解析、FFmpeg工具、可视化等通用功能
"""

# 日志
from .logger import (
    setup_logging,
    get_logger,
    get_context_logger,
    debug,
    info,
    warning,
    error,
    exception,
    critical,
)

# 异常
from .exceptions import (
    VideoAnalyticsError,
    InferenceError,
    ModelLoadError,
    ModelWarmupError,
    StreamError,
    StreamConnectionError,
    StreamReadError,
    StreamTimeoutError,
    DetectionError,
    DetectorNotFoundError,
    FenceConfigError,
    StorageError,
    UploadError,
    MinIOConnectionError,
    AlarmError,
    AlarmSendError,
    ConfigError,
    ConfigLoadError,
    ConfigValidationError,
    VideoServiceError,
    VideoGenerationError,
    VideoEncodingError,
    is_retryable_error,
    get_error_level,
)

# YOLO工具
from .yolo_utils import (
    YOLOV8_CLASSES,
    SAFETY_HELMET_CLASSES,
    parse_yolov8_output,
    nms,
    compute_iou,
    filter_detections_by_class,
    compute_bbox_area,
    compute_bbox_center,
    is_bbox_in_polygon,
    compute_bbox_overlap_ratio,
)

# FFmpeg工具
from .ffmpeg_utils import (
    FFmpegConfig,
    get_stream_info,
    build_ffmpeg_capture_cmd,
    build_ffmpeg_push_cmd,
    FFmpegCapture,
    FFmpegPusher,
    kill_ffmpeg_processes,
)

# 可视化工具
from .viz_utils import (
    COLORS,
    EVENT_STATE_COLORS,
    ALGO_TYPE_COLORS,
    CLASS_COLORS,
    get_color_for_class,
    get_color_for_state,
    get_color_for_algo,
    draw_bbox,
    draw_detections,
    draw_polygon,
    draw_fence,
    draw_text,
    draw_stats,
    draw_event_info,
    resize_frame,
    add_timestamp,
    create_preview_grid,
)

__all__ = [
    # 日志
    'setup_logging',
    'get_logger',
    'get_context_logger',
    'debug',
    'info',
    'warning',
    'error',
    'exception',
    'critical',

    # 异常
    'VideoAnalyticsError',
    'InferenceError',
    'ModelLoadError',
    'ModelWarmupError',
    'StreamError',
    'StreamConnectionError',
    'StreamReadError',
    'StreamTimeoutError',
    'DetectionError',
    'DetectorNotFoundError',
    'FenceConfigError',
    'StorageError',
    'UploadError',
    'MinIOConnectionError',
    'AlarmError',
    'AlarmSendError',
    'ConfigError',
    'ConfigLoadError',
    'ConfigValidationError',
    'VideoServiceError',
    'VideoGenerationError',
    'VideoEncodingError',
    'is_retryable_error',
    'get_error_level',

    # YOLO工具
    'YOLOV8_CLASSES',
    'SAFETY_HELMET_CLASSES',
    'parse_yolov8_output',
    'nms',
    'compute_iou',
    'filter_detections_by_class',
    'compute_bbox_area',
    'compute_bbox_center',
    'is_bbox_in_polygon',
    'compute_bbox_overlap_ratio',

    # FFmpeg工具
    'FFmpegConfig',
    'get_stream_info',
    'build_ffmpeg_capture_cmd',
    'build_ffmpeg_push_cmd',
    'FFmpegCapture',
    'FFmpegPusher',
    'kill_ffmpeg_processes',

    # 可视化工具
    'COLORS',
    'EVENT_STATE_COLORS',
    'ALGO_TYPE_COLORS',
    'CLASS_COLORS',
    'get_color_for_class',
    'get_color_for_state',
    'get_color_for_algo',
    'draw_bbox',
    'draw_detections',
    'draw_polygon',
    'draw_fence',
    'draw_text',
    'draw_stats',
    'draw_event_info',
    'resize_frame',
    'add_timestamp',
    'create_preview_grid',
]
