# Video Analytics System

基于 NVIDIA GPU 的实时视频分析系统，支持 RTSP 视频流接入、事件检测、告警上报、截图留证和事件视频生成。项目已从 Ascend NPU 方案迁移到面向 NVIDIA GPU 的多后端推理架构，并提供 V1 与 V2 两套流处理实现。

当前代码重点已经演进到 V2 高性能架构：

- 生产者-消费者解耦的流处理链路
- 有界队列，避免帧堆积导致内存持续增长
- 环形帧缓冲区，减少频繁分配和复制
- 视频生成线程池，避免事件高峰时线程爆炸
- 每路流独立 detector 实例，避免共享状态带来的线程安全问题

## 功能概览

系统当前支持以下检测算法：

| algorithmType | 算法名称 | 说明 |
| --- | --- | --- |
| `1` | 入侵检测 | 人员进入围栏区域触发事件 |
| `2` | 安全帽检测 | 检测人员是否佩戴安全帽 |
| `3` | 人群聚集检测 | 人数超过阈值时触发事件 |
| `4` | 烟火检测 | 检测烟雾或明火 |

支持的能力包括：

- RTSP 拉流与自动重连
- 多种推理后端切换：`ultralytics`、`tensorrt`、`onnx`、`torch`
- 事件状态机管理：触发、持续、结束、冷却
- 本地或 MinIO 存储截图与视频
- HTTP、控制台等多种告警输出方式
- 围栏绘制、叠框推流与事件证据保留

## 版本说明

### V1

V1 采用串行处理模式：读帧、检测、事件处理在同一条主链路中执行。

- 入口文件：`main_api.py`
- 配置文件：`cfg/config.json`
- 适合功能验证和兼容老部署方式

### V2

V2 是当前推荐版本，采用生产者-消费者架构，检测不再阻塞拉流。

- 入口文件：`main_api_v2.py`
- 配置模块：`video_analytics/config/settings_v2.py`
- 环境变量：`.env` / `.env.example`
- 新增性能接口：`GET /performance`

V2 核心链路：

```text
RTSP Stream
  -> FrameReader (Producer)
  -> Bounded Queue
  -> DetectionWorker (Consumer)
  -> Event Handler
  -> Storage / Alarm / VideoServiceV2
```

## 项目结构

```text
video_analytics/
  core/
    state_machine.py
    stream_processor.py
    stream_processor_v2.py
  detectors/
    base_detector.py
    intrusion_detector.py
    helmet_detector.py
    overcrowd_detector.py
    smokefire_detector.py
  engines/
    base_engine.py
    factory.py
    ultralytics_engine.py
    tensorrt_engine.py
    onnx_engine.py
    torch_engine.py
  services/
    alarm_service.py
    storage_service.py
    video_service.py
    video_service_v2.py
  config/
    settings.py
    settings_v2.py
  utils/
    logger.py
    exceptions.py
    yolo_utils.py
    ffmpeg_utils.py
    viz_utils.py

main.py
main_api.py
main_api_v2.py
cfg/config.json
.env.example
```

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 当前包含：

- `numpy`
- `opencv-python`
- `requests`
- `minio`
- `onnxruntime-gpu`
- `torch`
- `torchvision`
- `ultralytics`

如果要使用 TensorRT，请按 NVIDIA 官方方式安装 TensorRT，并额外安装：

```bash
pip install tensorrt pycuda
```

### 2. 准备模型

默认模型路径：

- 人员检测：`models/yolov8n.pt`
- 安全帽检测：`models/safehat.pt`
- 烟火检测：`video_analytics/models/smokefire.pt`

也可以通过环境变量覆盖这些路径。

## 配置方式

### V1 配置

V1 使用 `cfg/config.json`。

### V2 配置

V2 使用环境变量配置，命名规则为：

```text
VA_<SECTION>_<KEY>
```

建议先复制示例文件：

```bash
copy .env.example .env
```

常用配置项示例：

```env
VA_SERVER_HOST=0.0.0.0
VA_SERVER_PORT=5005
VA_MODEL_BACKEND=ultralytics
VA_MODEL_PERSON_MODEL_PATH=models/yolov8n.pt
VA_MODEL_HELMET_MODEL_PATH=models/safehat.pt
VA_MODEL_SMOKEFIRE_MODEL_PATH=video_analytics/models/smokefire.pt
VA_MODEL_CONFIDENCE=0.4

VA_DETECTION_INTRUSION_MIN_FRAMES=25
VA_DETECTION_HELMET_MIN_FRAMES=25
VA_DETECTION_OVERCROWD_MAX_PEOPLE=15
VA_DETECTION_FIRE_MIN_FRAMES=25

VA_STORAGE_TYPE=minio
VA_ALARM_TYPE=http
VA_STREAM_FPS=25
VA_STREAM_PRE_BUFFER_SECONDS=3
VA_STREAM_POST_RECORD_SECONDS=3
```

## 启动方式

### V1 API 模式

```bash
python main_api.py
```

### V2 API 模式

```bash
python main_api_v2.py
```

### 旧版轮询模式

```bash
python main.py
```

## PyInstaller 打包

推荐使用现有的 spec 文件进行 V2 打包：

```bash
pyinstaller --clean --noconfirm main_api_v2.spec
```

打包完成后，产物默认位于：

```text
dist/main_api_v2/
```

部署时建议只修改外部配置文件，不改代码：

- V2：编辑可执行文件同目录下的 `.env`
- V1：编辑 `cfg/config.runtime.json`

## API 接口

### `POST /set_fence`

启动一路检测流，并为该相机设置围栏。

请求示例：

```json
{
  "cam_id": "cam_001",
  "url": "rtsp://example.com/live",
  "algorithmType": 1,
  "fence_area": {
    "x1": 120,
    "y1": 80,
    "x2": 820,
    "y2": 420
  },
  "default_area": {
    "width": 960,
    "height": 540
  }
}
```

说明：

- `algorithmType` 必填，支持 `1`、`2`、`3`、`4`
- `cam_id` 为流的唯一标识
- `url` 为 RTSP 地址
- 未提供 `fence_area` 时，系统会默认使用整帧区域

调用示例：

```bash
curl -X POST "http://127.0.0.1:5005/set_fence" \
  -H "Content-Type: application/json" \
  -d "{\"cam_id\":\"cam_001\",\"url\":\"rtsp://example.com/live\",\"algorithmType\":1}"
```

### `POST /delete_stream`

停止并移除指定流。

```json
{
  "cam_id": "cam_001"
}
```

### `GET /status`

返回系统运行状态，包含：

- 当前活跃流数量
- 每路流的检测统计
- 视频服务统计
- V2 架构优化信息

### `GET /performance`

V2 专用性能接口，返回：

- 每路流的 `fps`
- `detection_fps`
- 平均检测延迟
- 丢帧数、丢事件数、丢视频任务数
- 队列与环形缓冲区状态
- 系统 CPU / 内存占用

## 核心设计

### 推理引擎层

所有推理后端都实现统一接口，工厂方法会根据模型文件或显式配置创建对应引擎：

- `UltralyticsEngine`
- `TensorRTInferEngine`
- `ONNXInferEngine`
- `TorchInferEngine`

### Detector 层

Detector 对外统一表现为：

```text
DetectionContext -> DetectionResultBundle
```

每个 detector 内部通过状态机管理事件生命周期，避免每帧都重复触发告警。

### 事件状态机

状态机位于 `video_analytics/core/state_machine.py`，负责管理：

- `IDLE`
- `COUNTING`
- `TRIGGERED`
- `ONGOING`
- `COOLDOWN`

### 工具模块

`video_analytics/utils/` 已经沉淀出一组通用工具：

- `logger.py`：统一日志能力
- `exceptions.py`：标准化异常体系
- `yolo_utils.py`：YOLO 输出解析与几何工具
- `ffmpeg_utils.py`：FFmpeg 拉流、推流与进程清理
- `viz_utils.py`：框、围栏、时间戳等可视化工具

## 运行与验证

### 基础验证

```bash
python test_ultralytics.py
python test_detection.py
```

### API 验证

启动后可用以下方式验证：

```bash
curl -X POST "http://127.0.0.1:5005/set_fence" \
  -H "Content-Type: application/json" \
  -d "{\"cam_id\":\"test\",\"url\":\"rtsp://your-stream\",\"algorithmType\":1}"
```

重点验证项：

- `POST /set_fence` 可正常启动检测
- 对同一个 `cam_id` 再次调用时，旧流能被干净重启
- `Ctrl+C` 后所有线程和 FFmpeg 进程能及时退出
- V2 下 `GET /performance` 能返回性能指标

## 当前已落地的优化点

- V2 流处理已引入生产者-消费者解耦
- 帧队列为有界队列，避免无限堆积
- 事件视频生成改为 `VideoServiceV2` 线程池模型
- 每路流独立 detector 实例，提升线程安全性
- 配置系统已支持 `.env` 与环境变量覆盖
- 统一工具模块已替换大量重复逻辑
- 已支持烟火检测 `algorithmType=4`

## 已知注意点

- 围栏坐标缩放依赖前端传入的 `default_area`
- 围栏推流与检测共用同一路 RTSP 源，但链路已经解耦
- FFmpeg 清理仍然依赖进程级回收逻辑，部署时建议重点验证关闭流程
- V2 为了线程安全采用独立 detector 实例，会增加一定内存占用

## 相关文档

- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md)
- [QUICK_START.md](QUICK_START.md)
- [CODE_QUALITY_IMPROVEMENTS.md](CODE_QUALITY_IMPROVEMENTS.md)

## License

MIT
