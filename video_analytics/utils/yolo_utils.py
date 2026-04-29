"""
YOLO工具函数
统一的YOLO输出解析和后处理
"""
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# 标准YOLOv8类别定义
YOLOV8_CLASSES: Dict[int, str] = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
    4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
    8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
    18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
    26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
    48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
    76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# 安全帽检测专用类别
SAFETY_HELMET_CLASSES: Dict[int, str] = {
    0: 'helmet',
    1: 'head'
}


def parse_yolov8_output(
    output: np.ndarray,
    input_shape: Tuple[int, int],
    orig_shape: Tuple[int, int],
    confidence_threshold: float = 0.4,
    iou_threshold: float = 0.45,
    target_classes: List[int] = None
) -> List[Dict]:
    """
    解析YOLOv8输出

    Args:
        output: 模型输出，shape为(N, 8400)或(N, 84, 8400)
        input_shape: 模型输入尺寸 (H, W)
        orig_shape: 原始图像尺寸 (H, W)
        confidence_threshold: 置信度阈值
        iou_threshold: NMS IoU阈值
        target_classes: 目标类别列表，None表示所有类别

    Returns:
        List[Dict]: 检测结果列表，每个元素包含:
            - bbox: [x1, y1, x2, y2]
            - confidence: 置信度
            - class_id: 类别ID
            - class_name: 类别名称
    """
    try:
        # 处理不同格式的输出
        if len(output.shape) == 3:
            # (batch, 84, 8400) -> (batch, 8400, 84)
            predictions = np.transpose(output[0], (1, 0))
        else:
            predictions = output

        # 分离边界框和类别分数
        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]

        # 获取每个检测的最高类别分数和类别ID
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # 过滤低置信度
        valid_mask = max_scores >= confidence_threshold
        if target_classes is not None:
            class_mask = np.isin(class_ids, target_classes)
            valid_mask = valid_mask & class_mask

        boxes = boxes[valid_mask]
        scores = max_scores[valid_mask]
        class_ids = class_ids[valid_mask]

        if len(boxes) == 0:
            return []

        # 转换框格式 (xywh -> xyxy)
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # 缩放框到原始图像尺寸
        input_h, input_w = input_shape
        orig_h, orig_w = orig_shape

        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y

        # 裁剪到图像边界
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        # NMS
        indices = nms(boxes_xyxy, scores, iou_threshold)

        # 构建结果
        results = []
        for idx in indices:
            class_id = int(class_ids[idx])
            results.append({
                'bbox': boxes_xyxy[idx].tolist(),
                'confidence': float(scores[idx]),
                'class_id': class_id,
                'class_name': YOLOV8_CLASSES.get(class_id, f'class_{class_id}')
            })

        return results

    except Exception as e:
        logger.error(f"Failed to parse YOLO output: {e}")
        return []


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """
    非极大值抑制 (NMS)

    Args:
        boxes: 边界框数组，shape (N, 4)，格式 [x1, y1, x2, y2]
        scores: 置信度数组，shape (N,)
        iou_threshold: IoU阈值

    Returns:
        List[int]: 保留的索引列表
    """
    if len(boxes) == 0:
        return []

    # 按分数降序排序
    indices = np.argsort(scores)[::-1]

    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # 计算当前框与其他框的IoU
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]

        ious = compute_iou(current_box, other_boxes)

        # 保留IoU小于阈值的框
        mask = ious < iou_threshold
        indices = indices[1:][mask]

    return keep


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    计算IoU

    Args:
        box: 单个边界框 [x1, y1, x2, y2]
        boxes: 边界框数组 (N, 4)

    Returns:
        np.ndarray: IoU数组 (N,)
    """
    # 计算交集
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # 计算并集
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - intersection

    # 避免除零
    iou = np.where(union > 0, intersection / union, 0)

    return iou


def filter_detections_by_class(
    detections: List[Dict],
    target_classes: List[int],
    min_confidence: float = None
) -> List[Dict]:
    """
    按类别过滤检测结果

    Args:
        detections: 检测结果列表
        target_classes: 目标类别ID列表
        min_confidence: 最小置信度阈值

    Returns:
        List[Dict]: 过滤后的检测结果
    """
    filtered = []
    for det in detections:
        if det['class_id'] in target_classes:
            if min_confidence is None or det['confidence'] >= min_confidence:
                filtered.append(det)
    return filtered


def compute_bbox_area(bbox: List[float]) -> float:
    """计算边界框面积"""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def compute_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """计算边界框中心点"""
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return center_x, center_y


def is_bbox_in_polygon(bbox: List[float], polygon: np.ndarray) -> bool:
    """
    检查边界框是否与多边形相交

    Args:
        bbox: 边界框 [x1, y1, x2, y2]
        polygon: 多边形点数组，shape (N, 2)

    Returns:
        bool: 是否相交
    """
    import cv2

    # 计算边界框中心点
    center = compute_bbox_center(bbox)

    # 使用 OpenCV 的 pointPolygonTest 检查中心点是否在多边形内
    # 如果中心点在多边形内或边界上，返回非负值
    dist = cv2.pointPolygonTest(polygon.astype(np.float32), center, False)

    return dist >= 0


def compute_bbox_overlap_ratio(bbox: List[float], polygon: np.ndarray) -> float:
    """
    计算边界框与多边形的重叠比例

    Args:
        bbox: 边界框 [x1, y1, x2, y2]
        polygon: 多边形点数组

    Returns:
        float: 重叠比例 (0-1)
    """
    import cv2

    # 创建边界框掩码
    x1, y1, x2, y2 = map(int, bbox)
    bbox_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    bbox_mask[:] = 255

    # 创建多边形掩码
    poly_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    pts = polygon - [x1, y1]  # 转换到bbox坐标系
    pts = pts.astype(np.int32)
    cv2.fillPoly(poly_mask, [pts], 255)

    # 计算重叠面积
    overlap = cv2.bitwise_and(bbox_mask, poly_mask)
    overlap_area = np.sum(overlap > 0)
    bbox_area = (x2 - x1) * (y2 - y1)

    return overlap_area / bbox_area if bbox_area > 0 else 0
