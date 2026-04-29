"""
Configuration settings for the legacy V1 entrypoint.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List
import json
import os


@dataclass
class ModelConfig:
    person_model_path: str = "models/yolov8n.pt"
    helmet_model_path: str = "models/safehat.pt"
    smokefire_model_path: str = "models/smokefire.pt"
    backend: str = "ultralytics"
    input_size: tuple = (640, 640)
    confidence: float = 0.4
    iou: float = 0.45
    device_id: int = 0
    fp16: bool = False


@dataclass
class DetectionConfig:
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


@dataclass
class StorageConfig:
    type: str = "minio"
    endpoint: str = "localhost:9000"
    access_key: str = ""
    secret_key: str = ""
    secure: bool = False
    bucket_name: str = "yolo"
    public_url: str = "localhost:9000"
    local_path: str = "./output"


@dataclass
class AlarmConfig:
    type: str = "http"
    endpoints: List[str] = field(default_factory=list)
    endpoints_intrusion: List[str] = field(default_factory=list)
    endpoints_helmet: List[str] = field(default_factory=list)
    endpoints_overcrowd: List[str] = field(default_factory=list)
    endpoints_smokefire: List[str] = field(default_factory=list)
    timeout: int = 10
    retry_count: int = 3


@dataclass
class StreamConfig:
    fps: int = 25
    skip_frames: int = 0
    max_reconnect: int = 5
    reconnect_interval: float = 2.0
    pre_buffer_seconds: int = 3
    post_record_seconds: int = 3
    enable_display: bool = False


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 5005
    debug: bool = False
    rtsp_push_host: str = "127.0.0.1"


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_file(cls, path: str) -> "AppConfig":
        if not os.path.exists(path):
            print(f"[Config] Config file not found: {path}, using defaults")
            return cls()

        with open(path, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        return cls(
            model=ModelConfig(**data.get("model", {})),
            detection=DetectionConfig(**data.get("detection", {})),
            storage=StorageConfig(**data.get("storage", {})),
            alarm=AlarmConfig(**data.get("alarm", {})),
            stream=StreamConfig(**data.get("stream", {})),
            server=ServerConfig(**data.get("server", {})),
        )

    def to_file(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        data = {
            "model": self.model.__dict__,
            "detection": self.detection.__dict__,
            "storage": self.storage.__dict__,
            "alarm": self.alarm.__dict__,
            "stream": self.stream.__dict__,
            "server": self.server.__dict__,
        }

        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=2, ensure_ascii=False)


default_config = AppConfig()
