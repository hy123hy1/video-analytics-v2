# 代码质量和可维护性改进

## 已完成的改进

### 1. 日志系统

**文件**: `video_analytics/utils/logger.py`

**特性**:
- 统一日志格式，包含时间、级别、文件名、行号
- 支持上下文日志（自动添加 camera_id 等信息）
- 支持同时输出到控制台和文件

**使用示例**:
```python
from video_analytics.utils import setup_logging, get_logger, get_context_logger

# 初始化日志系统
setup_logging(
    level="INFO",
    log_file="logs/video_analytics.log",
    enable_console=True
)

# 获取日志记录器
logger = get_logger(__name__)
logger.info("Stream started")
logger.error(f"Connection failed: {e}")

# 使用上下文日志
ctx_logger = get_context_logger(__name__, camera_id="cam_001", algo="intrusion")
ctx_logger.info("Event detected")  # 输出: [camera_id=cam_001] [algo=intrusion] Event detected
```

### 2. 配置环境变量支持

**文件**: `video_analytics/config/settings_v2.py`, `.env.example`

**特性**:
- 支持从环境变量覆盖配置
- 配置验证
- 类型安全转换

**环境变量命名规则**: `VA_<SECTION>_<KEY>`

**使用示例**:
```bash
# .env 文件
VA_MODEL_BACKEND=tensorrt
VA_SERVER_PORT=5006
VA_STORAGE_SECRET_KEY=your-secret-key
```

```python
from video_analytics.config.settings_v2 import AppConfigV2

# 加载配置（自动应用环境变量覆盖）
config = AppConfigV2.from_file("./cfg/config.json")

# 访问配置
print(config.model.backend)  # 从环境变量读取
print(config.server.port)    # 从环境变量读取
```

### 3. 重复代码提取

#### 3.1 YOLO工具 (`video_analytics/utils/yolo_utils.py`)

**提取的函数**:
- `parse_yolov8_output()` - 统一解析YOLO输出
- `nms()` - 非极大值抑制
- `compute_iou()` - 计算IoU
- `is_bbox_in_polygon()` - 检测框与多边形相交检查

**使用示例**:
```python
from video_analytics.utils import parse_yolov8_output, nms, is_bbox_in_polygon

# 解析YOLO输出
detections = parse_yolov8_output(
    output=model_output,
    input_shape=(640, 640),
    orig_shape=(1080, 1920),
    confidence_threshold=0.4,
    target_classes=[0]  # 只检测person
)

# 检查结果是否在围栏内
for det in detections:
    if is_bbox_in_polygon(det['bbox'], fence_polygon):
        print("Intrusion detected!")
```

#### 3.2 FFmpeg工具 (`video_analytics/utils/ffmpeg_utils.py`)

**提取的类**:
- `FFmpegCapture` - FFmpeg视频捕获
- `FFmpegPusher` - FFmpeg视频推流

**使用示例**:
```python
from video_analytics.utils import FFmpegCapture, FFmpegPusher, get_stream_info

# 获取流信息
info = get_stream_info("rtsp://...")
print(f"Resolution: {info['width']}x{info['height']}")

# 捕获视频
with FFmpegCapture("rtsp://...", width=1920, height=1080) as cap:
    while True:
        success, frame = cap.read()
        if success:
            process(frame)

# 推流视频
with FFmpegPusher("rtsp://output", width=1920, height=1080) as pusher:
    for frame in frames:
        pusher.write(frame)
```

#### 3.3 可视化工具 (`video_analytics/utils/viz_utils.py`)

**提取的函数**:
- `draw_bbox()` - 绘制边界框
- `draw_detections()` - 绘制检测结果
- `draw_fence()` - 绘制电子围栏
- `draw_text()` - 绘制文本
- `add_timestamp()` - 添加时间戳

**使用示例**:
```python
from video_analytics.utils import (
    draw_detections, draw_fence, draw_text,
    get_color_for_state, add_timestamp
)

# 绘制检测结果
result = draw_detections(
    image=frame,
    detections=detections,
    show_confidence=True
)

# 绘制围栏
result = draw_fence(
    image=result,
    fence_points=fence_coords,
    is_triggered=True
)

# 添加时间戳
result = add_timestamp(result)
```

### 4. 规范化错误处理

**文件**: `video_analytics/utils/exceptions.py`

**特性**:
- 统一的异常体系
- 错误分类（可重试/不可重试）
- 错误级别判断

**使用示例**:
```python
from video_analytics.utils import (
    StreamConnectionError,
    is_retryable_error,
    get_error_level
)

try:
    connect_stream(rtsp_url)
except StreamConnectionError as e:
    if is_retryable_error(e):
        logger.warning(f"Connection failed, will retry: {e}")
        schedule_retry()
    else:
        logger.error(f"Permanent error: {e}")

# 根据错误级别处理
level = get_error_level(error)
if level == "CRITICAL":
    send_alert_to_ops(error)
elif level == "WARNING":
    log_and_continue(error)
```

---

## 新增文件清单

```
video_analytics/utils/
├── __init__.py              # 统一导出
├── logger.py                # 日志系统
├── exceptions.py            # 异常体系
├── yolo_utils.py            # YOLO工具函数
├── ffmpeg_utils.py          # FFmpeg工具
└── viz_utils.py             # 可视化工具

video_analytics/config/
└── settings_v2.py           # 支持环境变量的配置

.env.example                 # 环境变量配置示例
```

---

## 下一步建议

1. **迁移现有代码** - 使用新工具模块替换重复代码
2. **添加类型检查** - 使用 mypy 进行静态类型检查
3. **编写单元测试** - 为工具模块编写测试用例
4. **代码格式化** - 使用 black 统一代码风格
