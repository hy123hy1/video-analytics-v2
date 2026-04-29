"""
可视化工具函数
统一的图像绘制和可视化
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# 预定义颜色 (BGR格式)
COLORS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'orange': (0, 165, 255),
    'purple': (128, 0, 128),
}

# 检测状态颜色映射
EVENT_STATE_COLORS = {
    'idle': COLORS['green'],
    'counting': COLORS['yellow'],
    'triggered': COLORS['red'],
    'ongoing': COLORS['red'],
    'cooldown': COLORS['cyan'],
}

# 算法类型颜色映射
ALGO_TYPE_COLORS = {
    '1': COLORS['red'],      # 闯入检测
    '2': COLORS['yellow'],   # 安全帽检测
    '3': COLORS['blue'],     # 超员检测
    '4': COLORS['green'],    # 新算法
}

# 类别颜色映射 (用于不同类别的检测框)
CLASS_COLORS = [
    (0, 0, 255),    # 红色 - person
    (0, 255, 0),    # 绿色
    (255, 0, 0),    # 蓝色
    (0, 255, 255),  # 黄色
    (255, 255, 0),  # 青色
    (255, 0, 255),  # 紫色
    (128, 128, 128),# 灰色
    (0, 165, 255),  # 橙色
]


def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """获取类别对应的颜色"""
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def get_color_for_state(state: str) -> Tuple[int, int, int]:
    """获取状态对应的颜色"""
    return EVENT_STATE_COLORS.get(state.lower(), COLORS['white'])


def get_color_for_algo(algo_type: str) -> Tuple[int, int, int]:
    """获取算法类型对应的颜色"""
    return ALGO_TYPE_COLORS.get(algo_type, COLORS['white'])


def draw_bbox(
    image: np.ndarray,
    bbox: List[float],
    color: Tuple[int, int, int] = COLORS['green'],
    thickness: int = 2,
    label: Optional[str] = None,
    confidence: Optional[float] = None,
    font_scale: float = 0.6
) -> np.ndarray:
    """
    绘制边界框

    Args:
        image: 输入图像
        bbox: 边界框 [x1, y1, x2, y2]
        color: 框颜色
        thickness: 线宽
        label: 标签文本
        confidence: 置信度
        font_scale: 字体大小

    Returns:
        np.ndarray: 绘制后的图像
    """
    x1, y1, x2, y2 = map(int, bbox)

    # 绘制边界框
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # 绘制标签
    if label or confidence is not None:
        text = label if label else ""
        if confidence is not None:
            text = f"{text} {confidence:.2f}" if text else f"{confidence:.2f}"

        # 计算文本大小
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        # 绘制标签背景
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 4),
            (x1 + text_width, y1),
            color,
            -1
        )

        # 绘制文本
        cv2.putText(
            image,
            text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            COLORS['white'],
            1,
            cv2.LINE_AA
        )

    return image


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    thickness: int = 2,
    show_confidence: bool = True,
    show_class: bool = True
) -> np.ndarray:
    """
    绘制检测结果

    Args:
        image: 输入图像
        detections: 检测结果列表，每个元素包含 'bbox', 'class_id', 'class_name', 'confidence'
        color_map: 类别到颜色的映射
        thickness: 线宽
        show_confidence: 是否显示置信度
        show_class: 是否显示类别名

    Returns:
        np.ndarray: 绘制后的图像
    """
    result = image.copy()

    for det in detections:
        bbox = det.get('bbox', [0, 0, 0, 0])
        class_id = det.get('class_id', 0)
        class_name = det.get('class_name', f'class_{class_id}')
        confidence = det.get('confidence', 0.0)

        # 确定颜色
        if color_map and class_id in color_map:
            color = color_map[class_id]
        else:
            color = get_color_for_class(class_id)

        # 构建标签
        label = None
        if show_class:
            label = class_name
        if show_confidence:
            label = f"{label} {confidence:.2f}" if label else f"{confidence:.2f}"

        result = draw_bbox(result, bbox, color, thickness, label)

    return result


def draw_polygon(
    image: np.ndarray,
    points: List[List[int]],
    color: Tuple[int, int, int] = COLORS['red'],
    thickness: int = 2,
    fill: bool = False,
    alpha: float = 0.3
) -> np.ndarray:
    """
    绘制多边形

    Args:
        image: 输入图像
        points: 多边形顶点 [[x1,y1], [x2,y2], ...]
        color: 线条颜色
        thickness: 线宽
        fill: 是否填充
        alpha: 填充透明度

    Returns:
        np.ndarray: 绘制后的图像
    """
    result = image.copy()
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))

    if fill:
        # 创建填充层
        overlay = result.copy()
        cv2.fillPoly(overlay, [pts], color)
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

    # 绘制边框
    cv2.polylines(result, [pts], True, color, thickness)

    return result


def draw_fence(
    image: np.ndarray,
    fence_points: List[List[int]],
    is_triggered: bool = False,
    thickness: int = 2,
    fill_alpha: float = 0.2
) -> np.ndarray:
    """
    绘制电子围栏

    Args:
        image: 输入图像
        fence_points: 围栏顶点
        is_triggered: 是否触发报警
        thickness: 线宽
        fill_alpha: 填充透明度

    Returns:
        np.ndarray: 绘制后的图像
    """
    color = COLORS['red'] if is_triggered else COLORS['green']
    return draw_polygon(image, fence_points, color, thickness, fill=True, alpha=fill_alpha)


def draw_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = COLORS['white'],
    font_scale: float = 0.6,
    thickness: int = 1,
    bg_color: Optional[Tuple[int, int, int]] = None,
    padding: int = 4
) -> np.ndarray:
    """
    绘制文本

    Args:
        image: 输入图像
        text: 文本内容
        position: 位置 (x, y)
        color: 文本颜色
        font_scale: 字体大小
        thickness: 线宽
        bg_color: 背景颜色
        padding: 内边距

    Returns:
        np.ndarray: 绘制后的图像
    """
    result = image.copy()
    x, y = position

    # 计算文本大小
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    # 绘制背景
    if bg_color:
        cv2.rectangle(
            result,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + padding),
            bg_color,
            -1
        )

    # 绘制文本
    cv2.putText(
        result,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )

    return result


def draw_stats(
    image: np.ndarray,
    stats: Dict[str, any],
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = COLORS['white'],
    font_scale: float = 0.5,
    line_spacing: int = 20
) -> np.ndarray:
    """
    绘制统计信息

    Args:
        image: 输入图像
        stats: 统计信息字典
        position: 起始位置
        color: 文本颜色
        font_scale: 字体大小
        line_spacing: 行间距

    Returns:
        np.ndarray: 绘制后的图像
    """
    result = image.copy()
    x, y = position

    for key, value in stats.items():
        text = f"{key}: {value}"
        result = draw_text(result, text, (x, y), color, font_scale)
        y += line_spacing

    return result


def draw_event_info(
    image: np.ndarray,
    event_type: str,
    camera_id: str,
    confidence: float = None,
    position: str = 'top-left'
) -> np.ndarray:
    """
    绘制事件信息

    Args:
        image: 输入图像
        event_type: 事件类型
        camera_id: 摄像头ID
        confidence: 置信度
        position: 位置 ('top-left', 'top-right', 'bottom-left', 'bottom-right')

    Returns:
        np.ndarray: 绘制后的图像
    """
    result = image.copy()
    h, w = result.shape[:2]

    # 构建文本
    texts = [f"EVENT: {event_type}", f"Camera: {camera_id}"]
    if confidence is not None:
        texts.append(f"Confidence: {confidence:.2f}")

    # 确定位置
    if position == 'top-left':
        x, y = 10, 30
    elif position == 'top-right':
        x = w - 200
        y = 30
    elif position == 'bottom-left':
        x = 10
        y = h - 40
    else:  # bottom-right
        x = w - 200
        y = h - 40

    # 绘制背景
    max_width = max(len(t) for t in texts) * 10
    cv2.rectangle(result, (x - 5, y - 25), (x + max_width, y + len(texts) * 20), (0, 0, 0), -1)

    # 绘制文本
    for i, text in enumerate(texts):
        color = COLORS['red'] if i == 0 else COLORS['white']
        result = draw_text(result, text, (x, y + i * 20), color, font_scale=0.6)

    return result


def resize_frame(frame: np.ndarray, max_size: int = 1920) -> np.ndarray:
    """
    缩放帧到最大尺寸（保持宽高比）

    Args:
        frame: 输入帧
        max_size: 最大边长

    Returns:
        np.ndarray: 缩放后的帧
    """
    h, w = frame.shape[:2]
    max_dim = max(h, w)

    if max_dim <= max_size:
        return frame

    scale = max_size / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def add_timestamp(
    image: np.ndarray,
    timestamp: str = None,
    position: Tuple[int, int] = None,
    color: Tuple[int, int, int] = COLORS['white']
) -> np.ndarray:
    """
    添加时间戳

    Args:
        image: 输入图像
        timestamp: 时间戳字符串，None则使用当前时间
        position: 位置，None则使用右下角
        color: 文本颜色

    Returns:
        np.ndarray: 绘制后的图像
    """
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if position is None:
        h, w = image.shape[:2]
        position = (w - 200, h - 10)

    return draw_text(image, timestamp, position, color, font_scale=0.5, bg_color=(0, 0, 0))


def create_preview_grid(
    frames: List[np.ndarray],
    cols: int = 2,
    target_size: Tuple[int, int] = (640, 480)
) -> np.ndarray:
    """
    创建预览网格

    Args:
        frames: 帧列表
        cols: 列数
        target_size: 每个格子的目标大小 (width, height)

    Returns:
        np.ndarray: 网格图像
    """
    if not frames:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # 统一尺寸
    resized = []
    for frame in frames:
        if frame.shape[:2] != (target_size[1], target_size[0]):
            resized.append(cv2.resize(frame, target_size))
        else:
            resized.append(frame)

    # 计算行数
    rows = (len(resized) + cols - 1) // cols

    # 创建网格
    grid_rows = []
    for i in range(rows):
        row_frames = resized[i * cols:(i + 1) * cols]

        # 补齐空位
        while len(row_frames) < cols:
            row_frames.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))

        grid_rows.append(np.hstack(row_frames))

    return np.vstack(grid_rows)
