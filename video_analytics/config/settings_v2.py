"""
Configuration Settings V2.

This module loads application settings from environment variables and .env
files using the naming convention: VA_<SECTION>_<KEY>.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from video_analytics.utils.exceptions import ConfigValidationError

logger = logging.getLogger(__name__)


def load_env_file(path: str = ".env", override: bool = False) -> Dict[str, str]:
    """
    Load KEY=VALUE pairs from a .env file into os.environ.
    """
    env_path = Path(path)
    if not env_path.exists():
        logger.warning(f".env file not found: {env_path.resolve()}")
        return {}

    loaded: Dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        quoted = (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in ("'", '"')
        )
        if quoted:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()

        if override or key not in os.environ:
            os.environ[key] = value

        loaded[key] = value

    logger.info(f"Loaded {len(loaded)} variables from .env: {env_path.resolve()}")
    return loaded


def _get_env_value(key: str, default: Any, value_type: Type = str) -> Any:
    """
    Read an environment variable and convert it to the requested type.
    """
    value = os.getenv(key)
    if value is None:
        return default

    try:
        if value_type == bool:
            return value.lower() in ("true", "1", "yes", "on")
        if value_type == int:
            return int(value)
        if value_type == float:
            return float(value)
        if value_type == list:
            return [item.strip() for item in value.split(",") if item.strip()]
        if value_type == tuple:
            parts = [int(x.strip()) for x in value.split(",")]
            return tuple(parts)
        return value
    except (ValueError, TypeError) as exc:
        logger.warning(f"Failed to parse env var {key}={value}: {exc}, using default")
        return default


@dataclass
class ModelConfig:
    """Model configuration."""

    person_model_path: str = "models/yolov8n.pt"
    helmet_model_path: str = "models/safehat.pt"
    smokefire_model_path: str = "models/smokefire.pt"
    backend: str = "ultralytics"
    input_size: tuple = (640, 640)
    confidence: float = 0.4
    iou: float = 0.45
    device_id: int = 0
    fp16: bool = False

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Build model configuration from environment variables."""
        return cls(
            person_model_path=_get_env_value(
                "VA_MODEL_PERSON_MODEL_PATH", cls.person_model_path
            ),
            helmet_model_path=_get_env_value(
                "VA_MODEL_HELMET_MODEL_PATH", cls.helmet_model_path
            ),
            smokefire_model_path=_get_env_value(
                "VA_MODEL_SMOKEFIRE_MODEL_PATH", cls.smokefire_model_path
            ),
            backend=_get_env_value("VA_MODEL_BACKEND", cls.backend),
            input_size=_get_env_value("VA_MODEL_INPUT_SIZE", cls.input_size, tuple),
            confidence=_get_env_value("VA_MODEL_CONFIDENCE", cls.confidence, float),
            iou=_get_env_value("VA_MODEL_IOU", cls.iou, float),
            device_id=_get_env_value("VA_MODEL_DEVICE_ID", cls.device_id, int),
            fp16=_get_env_value("VA_MODEL_FP16", cls.fp16, bool),
        )


@dataclass
class DetectionConfig:
    """Detection configuration."""

    intrusion_min_frames: int = 25
    intrusion_cooldown: float = 60.0
    intrusion_confidence: float = 0.5

    helmet_min_frames: int = 25
    helmet_cooldown: float = 60.0
    helmet_confidence: float = 0.4
    helmet_crop_padding: float = 0.2

    overcrowd_max_people: int = 15
    overcrowd_duration: float = 2.0
    overcrowd_cooldown: float = 60.0
    overcrowd_confidence: float = 0.4

    fire_min_frames: int = 25
    fire_cooldown: float = 60.0
    fire_confidence: float = 0.4

    @classmethod
    def from_env(cls) -> "DetectionConfig":
        """Build detection configuration from environment variables."""
        return cls(
            intrusion_min_frames=_get_env_value(
                "VA_DETECTION_INTRUSION_MIN_FRAMES", cls.intrusion_min_frames, int
            ),
            intrusion_cooldown=_get_env_value(
                "VA_DETECTION_INTRUSION_COOLDOWN", cls.intrusion_cooldown, float
            ),
            intrusion_confidence=_get_env_value(
                "VA_DETECTION_INTRUSION_CONFIDENCE", cls.intrusion_confidence, float
            ),
            helmet_min_frames=_get_env_value(
                "VA_DETECTION_HELMET_MIN_FRAMES", cls.helmet_min_frames, int
            ),
            helmet_cooldown=_get_env_value(
                "VA_DETECTION_HELMET_COOLDOWN", cls.helmet_cooldown, float
            ),
            helmet_confidence=_get_env_value(
                "VA_DETECTION_HELMET_CONFIDENCE", cls.helmet_confidence, float
            ),
            helmet_crop_padding=_get_env_value(
                "VA_DETECTION_HELMET_CROP_PADDING", cls.helmet_crop_padding, float
            ),
            overcrowd_max_people=_get_env_value(
                "VA_DETECTION_OVERCROWD_MAX_PEOPLE", cls.overcrowd_max_people, int
            ),
            overcrowd_duration=_get_env_value(
                "VA_DETECTION_OVERCROWD_DURATION", cls.overcrowd_duration, float
            ),
            overcrowd_cooldown=_get_env_value(
                "VA_DETECTION_OVERCROWD_COOLDOWN", cls.overcrowd_cooldown, float
            ),
            overcrowd_confidence=_get_env_value(
                "VA_DETECTION_OVERCROWD_CONFIDENCE", cls.overcrowd_confidence, float
            ),
            fire_min_frames=_get_env_value(
                "VA_DETECTION_FIRE_MIN_FRAMES", cls.fire_min_frames, int
            ),
            fire_cooldown=_get_env_value(
                "VA_DETECTION_FIRE_COOLDOWN", cls.fire_cooldown, float
            ),
            fire_confidence=_get_env_value(
                "VA_DETECTION_FIRE_CONFIDENCE", cls.fire_confidence, float
            ),
        )


@dataclass
class StorageConfig:
    """Storage configuration."""

    type: str = "minio"
    endpoint: str = "localhost:9000"
    access_key: str = ""
    secret_key: str = ""
    secure: bool = False
    bucket_name: str = "yolo"
    public_url: str = "localhost:9000"
    local_path: str = "./output"

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Build storage configuration from environment variables."""
        return cls(
            type=_get_env_value("VA_STORAGE_TYPE", cls.type),
            endpoint=_get_env_value("VA_STORAGE_ENDPOINT", cls.endpoint),
            access_key=_get_env_value("VA_STORAGE_ACCESS_KEY", cls.access_key),
            secret_key=_get_env_value("VA_STORAGE_SECRET_KEY", cls.secret_key),
            secure=_get_env_value("VA_STORAGE_SECURE", cls.secure, bool),
            bucket_name=_get_env_value("VA_STORAGE_BUCKET_NAME", cls.bucket_name),
            public_url=_get_env_value("VA_STORAGE_PUBLIC_URL", cls.public_url),
            local_path=_get_env_value("VA_STORAGE_LOCAL_PATH", cls.local_path),
        )


@dataclass
class AlarmConfig:
    """Alarm configuration."""

    type: str = "http"
    endpoints: List[str] = field(default_factory=list)
    endpoints_intrusion: List[str] = field(default_factory=list)
    endpoints_helmet: List[str] = field(default_factory=list)
    endpoints_overcrowd: List[str] = field(default_factory=list)
    endpoints_smokefire: List[str] = field(default_factory=list)
    timeout: int = 10
    retry_count: int = 3

    @classmethod
    def from_env(cls) -> "AlarmConfig":
        """Build alarm configuration from environment variables."""
        return cls(
            type=_get_env_value("VA_ALARM_TYPE", "http"),
            endpoints=_get_env_value("VA_ALARM_ENDPOINTS", [], list),
            endpoints_intrusion=_get_env_value(
                "VA_ALARM_ENDPOINTS_INTRUSION", [], list
            ),
            endpoints_helmet=_get_env_value("VA_ALARM_ENDPOINTS_HELMET", [], list),
            endpoints_overcrowd=_get_env_value(
                "VA_ALARM_ENDPOINTS_OVERCROWD", [], list
            ),
            endpoints_smokefire=_get_env_value(
                "VA_ALARM_ENDPOINTS_SMOKEFIRE", [], list
            ),
            timeout=_get_env_value("VA_ALARM_TIMEOUT", 10, int),
            retry_count=_get_env_value("VA_ALARM_RETRY_COUNT", 3, int),
        )


@dataclass
class StreamConfig:
    """Stream processing configuration."""

    fps: int = 25
    skip_frames: int = 0
    max_reconnect: int = 5
    reconnect_interval: float = 2.0
    pre_buffer_seconds: int = 3
    post_record_seconds: int = 3
    enable_display: bool = False

    @classmethod
    def from_env(cls) -> "StreamConfig":
        """Build stream configuration from environment variables."""
        return cls(
            fps=_get_env_value("VA_STREAM_FPS", cls.fps, int),
            skip_frames=_get_env_value("VA_STREAM_SKIP_FRAMES", cls.skip_frames, int),
            max_reconnect=_get_env_value(
                "VA_STREAM_MAX_RECONNECT", cls.max_reconnect, int
            ),
            reconnect_interval=_get_env_value(
                "VA_STREAM_RECONNECT_INTERVAL", cls.reconnect_interval, float
            ),
            pre_buffer_seconds=_get_env_value(
                "VA_STREAM_PRE_BUFFER_SECONDS", cls.pre_buffer_seconds, int
            ),
            post_record_seconds=_get_env_value(
                "VA_STREAM_POST_RECORD_SECONDS", cls.post_record_seconds, int
            ),
            enable_display=_get_env_value(
                "VA_STREAM_ENABLE_DISPLAY", cls.enable_display, bool
            ),
        )


@dataclass
class ServerConfig:
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 5005
    debug: bool = False
    rtsp_push_host: str = "127.0.0.1"
    rtsp_push_port: int = 554
    rtsp_public_host: str = ""
    rtsp_public_port: int = 554
    rtsp_push_url_template: str = ""
    rtsp_public_url_template: str = ""
    rtsp_push_username: str = ""
    rtsp_push_password: str = ""
    rtsp_push_sign: str = ""
    rtsp_push_key: str = ""
    rtsp_push_call_id: str = ""
    rtsp_source_public_host: str = ""
    rtsp_source_local_host: str = ""
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Build server configuration from environment variables."""
        return cls(
            host=_get_env_value("VA_SERVER_HOST", cls.host),
            port=_get_env_value("VA_SERVER_PORT", cls.port, int),
            debug=_get_env_value("VA_SERVER_DEBUG", cls.debug, bool),
            rtsp_push_host=_get_env_value(
                "VA_SERVER_RTSP_PUSH_HOST", cls.rtsp_push_host
            ),
            rtsp_push_port=_get_env_value(
                "VA_SERVER_RTSP_PUSH_PORT", cls.rtsp_push_port, int
            ),
            rtsp_public_host=_get_env_value(
                "VA_SERVER_RTSP_PUBLIC_HOST", cls.rtsp_public_host
            ),
            rtsp_public_port=_get_env_value(
                "VA_SERVER_RTSP_PUBLIC_PORT", cls.rtsp_public_port, int
            ),
            rtsp_push_url_template=_get_env_value(
                "VA_SERVER_RTSP_PUSH_URL_TEMPLATE", cls.rtsp_push_url_template
            ),
            rtsp_public_url_template=_get_env_value(
                "VA_SERVER_RTSP_PUBLIC_URL_TEMPLATE", cls.rtsp_public_url_template
            ),
            rtsp_push_username=_get_env_value(
                "VA_SERVER_RTSP_PUSH_USERNAME", cls.rtsp_push_username
            ),
            rtsp_push_password=_get_env_value(
                "VA_SERVER_RTSP_PUSH_PASSWORD", cls.rtsp_push_password
            ),
            rtsp_push_sign=_get_env_value(
                "VA_SERVER_RTSP_PUSH_SIGN", cls.rtsp_push_sign
            ),
            rtsp_push_key=_get_env_value(
                "VA_SERVER_RTSP_PUSH_KEY", cls.rtsp_push_key
            ),
            rtsp_push_call_id=_get_env_value(
                "VA_SERVER_RTSP_PUSH_CALL_ID", cls.rtsp_push_call_id
            ),
            rtsp_source_public_host=_get_env_value(
                "VA_SERVER_RTSP_SOURCE_PUBLIC_HOST", cls.rtsp_source_public_host
            ),
            rtsp_source_local_host=_get_env_value(
                "VA_SERVER_RTSP_SOURCE_LOCAL_HOST", cls.rtsp_source_local_host
            ),
            log_level=_get_env_value("VA_SERVER_LOG_LEVEL", cls.log_level),
            log_file=_get_env_value("VA_SERVER_LOG_FILE", cls.log_file),
        )


@dataclass
class AppConfigV2:
    """Top-level application configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_env_file(
        cls, path: str = ".env", override_env: bool = False
    ) -> "AppConfigV2":
        """
        Load config from a .env file and process environment variables.

        Priority:
            existing process env (unless override_env=True) + .env > defaults
        """
        load_env_file(path, override=override_env)

        config = cls(
            model=ModelConfig.from_env(),
            detection=DetectionConfig.from_env(),
            storage=StorageConfig.from_env(),
            alarm=AlarmConfig.from_env(),
            stream=StreamConfig.from_env(),
            server=ServerConfig.from_env(),
        )

        config.validate()
        logger.info(f"Configuration loaded from env file: {Path(path).resolve()}")
        return config

    @classmethod
    def from_file(cls, path: str) -> "AppConfigV2":
        """
        Backward-compatible entry.

        V2 now prefers .env-based configuration.
        """
        suffix = Path(path).suffix.lower()
        if suffix == ".env":
            return cls.from_env_file(path)

        env_path = os.getenv("VA_ENV_FILE", ".env")
        logger.warning(
            f"JSON config path '{path}' is deprecated for V2. "
            f"Loading from env file '{env_path}' instead."
        )
        return cls.from_env_file(env_path)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AppConfigV2":
        """Build configuration from a plain dictionary."""
        return cls(
            model=ModelConfig(**data.get("model", {})),
            detection=DetectionConfig(**data.get("detection", {})),
            storage=StorageConfig(**data.get("storage", {})),
            alarm=AlarmConfig(**data.get("alarm", {})),
            stream=StreamConfig(**data.get("stream", {})),
            server=ServerConfig(**data.get("server", {})),
        )

    def to_file(self, path: str):
        """Write configuration to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": asdict(self.model),
            "detection": asdict(self.detection),
            "storage": asdict(self.storage),
            "alarm": asdict(self.alarm),
            "stream": asdict(self.stream),
            "server": asdict(self.server),
        }

        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=2, ensure_ascii=False)

        logger.info(f"Configuration saved to {path}")

    def validate(self):
        """Validate the loaded configuration."""
        errors = []

        if not self.model.person_model_path:
            errors.append("Model path cannot be empty")

        if not 0 <= self.model.confidence <= 1:
            errors.append(
                f"Model confidence must be in [0, 1], got {self.model.confidence}"
            )

        if not 0 <= self.model.iou <= 1:
            errors.append(f"Model IoU must be in [0, 1], got {self.model.iou}")

        if not 1 <= self.server.port <= 65535:
            errors.append(f"Server port must be in [1, 65535], got {self.server.port}")

        if errors:
            raise ConfigValidationError(
                "Configuration validation failed",
                details={"errors": errors},
            )

    def __repr__(self) -> str:
        return (
            f"AppConfigV2("
            f"model={self.model.backend}, "
            f"detection=intrusion:{self.detection.intrusion_min_frames}, "
            f"storage={self.storage.type}, "
            f"alarm={self.alarm.type}, "
            f"server={self.server.host}:{self.server.port}"
            f")"
        )


default_config = AppConfigV2()
