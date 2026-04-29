"""
SmokeFire Detector - 烟火检测器
电子围栏场景：检测指定区域内的烟雾和火焰

检测类别:
- 0: smoke (烟雾)
- 1: fire (火焰)
"""
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from datetime import datetime
from dataclasses import dataclass

from video_analytics.detectors.base_detector import (
    BaseDetector, DetectionContext, DetectionResultBundle,
    DetectionEvent, DetectionEventType
)
from video_analytics.engines.base_engine import BaseInferEngine, DetectionResult
from video_analytics.core.state_machine import EventStateMachine, EventState


@dataclass
class FenceRegion:
    """电子围栏区域"""
    points: List[tuple]  # 多边形顶点 [(x1,y1), (x2,y2), ...]
    name: str = "fence"

    def contains_point(self, x: float, y: float) -> bool:
        """判断点是否在区域内"""
        return cv2.pointPolygonTest(
            np.array(self.points, dtype=np.int32),
            (float(x), float(y)),
            False
        ) >= 0

    def draw(self, frame: np.ndarray, color: tuple = (0, 0, 255), thickness: int = 2):
        """在图像上绘制围栏"""
        pts = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, color, thickness)


class SmokeFireDetector(BaseDetector):
    """
    烟火检测器

    检测逻辑:
    1. 使用YOLO检测烟雾和火焰
    2. 计算检测框中心点
    3. 判断中心点是否在电子围栏区域内
    4. 应用状态机防抖 (连续N帧检测到才触发)
    5. 事件冷却 (避免重复报警)

    配置参数:
        min_frames: 连续检测帧数阈值 (默认25帧)
        confidence: 检测置信度 (默认0.4)
        cooldown_seconds: 事件冷却时间 (默认60秒)
        target_classes: 目标类别ID列表 (默认[0, 1] - smoke和fire)
    """

    # 类别名称映射
    CLASS_NAMES = {
        0: "smoke",
        1: "fire"
    }

    # 类别颜色 (BGR格式)
    CLASS_COLORS = {
        0: (128, 128, 128),  # 烟雾 - 灰色
        1: (0, 0, 255),      # 火焰 - 红色
    }

    def __init__(
        self,
        engine: BaseInferEngine,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            engine=engine,
            config=config,
            event_type=DetectionEventType.FIRE  # 使用火灾事件类型
        )

        # 配置
        self.min_frames = self.config.get("min_frames", 25)
        self.confidence = self.config.get("confidence", 0.4)
        self.cooldown_seconds = self.config.get("cooldown_seconds", 60)
        self.target_classes = self.config.get("target_classes", [0, 1])  # smoke和fire

        # 状态机管理 {camera_id: EventStateMachine}
        self._state_machines: Dict[str, EventStateMachine] = {}

        # 围栏配置 {camera_id: FenceRegion}
        self._fences: Dict[str, FenceRegion] = {}

    def set_fence(self, camera_id: str, fence: FenceRegion):
        """设置电子围栏"""
        self._fences[camera_id] = fence

    def set_fence_from_points(self, camera_id: str, points: List[tuple]):
        """从点列表设置围栏"""
        self._fences[camera_id] = FenceRegion(points=points)

    def set_fence_full_frame(self, camera_id: str, margin: float = 0.0):
        """
        设置全图围栏 (用于全画面检测)

        Args:
            camera_id: 摄像头ID
            margin: 边距比例 (0.0 = 全图, 0.1 = 留10%边距)
        """
        self._fences[camera_id] = FenceRegion(
            points=[(0, 0), (100, 0), (100, 100), (0, 100)],
            name="full_frame_auto"
        )
        # 标记为需要自动调整
        self._fences[camera_id]._auto_resize = True
        self._fences[camera_id]._margin = margin

    def process(self, context: DetectionContext) -> DetectionResultBundle:
        """
        处理单帧并检测烟火事件

        Args:
            context: 检测上下文

        Returns:
            DetectionResultBundle: 检测结果
        """
        import time
        start_time = time.perf_counter()

        camera_id = context.camera_id
        frame = context.frame

        # 检查围栏配置
        if camera_id not in self._fences:
            # 如果没有设置围栏，默认检测整个画面
            h, w = frame.shape[:2]
            self._fences[camera_id] = FenceRegion(
                points=[(0, 0), (w, 0), (w, h), (0, h)],
                name="full_frame"
            )

        fence = self._fences[camera_id]

        # 1. 执行推理
        detections, infer_context = self._infer(frame)

        # 2. 过滤目标类别和置信度
        filtered_dets = self._filter_by_class(
            detections,
            self.target_classes,
            self.confidence
        )

        # 3. 检测围栏内的烟火
        in_fence_dets = self._detect_in_fence(filtered_dets, fence)

        # 4. 获取/创建状态机
        if camera_id not in self._state_machines:
            self._state_machines[camera_id] = EventStateMachine(
                min_trigger_frames=self.min_frames,
                min_end_frames=25,
                cooldown_seconds=self.cooldown_seconds
            )

        state_machine = self._state_machines[camera_id]

        # 5. 更新状态机
        has_smokefire = len(in_fence_dets) > 0
        state = state_machine.update(has_smokefire)

        # 6. 可视化
        visualized = self._visualize(frame, filtered_dets, in_fence_dets, fence, state)

        # 7. 构建结果
        result = DetectionResultBundle(
            triggered=state in [EventState.TRIGGERED, EventState.ONGOING],
            detections=filtered_dets,
            visualized_frame=visualized,
            debug_info={
                "state": state.value,
                "in_fence_count": len(in_fence_dets),
                "total_detections": len(filtered_dets),
                "infer_time": infer_context.inference_time
            }
        )

        # 8. 事件触发时创建事件对象
        if state == EventState.TRIGGERED:
            result.event = self._create_event(context, in_fence_dets)
            self._trigger_count += 1

        self._process_count += 1
        self._total_process_time += time.perf_counter() - start_time

        return result

    def _detect_in_fence(
        self,
        detections: List[DetectionResult],
        fence: FenceRegion
    ) -> List[DetectionResult]:
        """
        检测围栏内的烟火

        Args:
            detections: 检测结果
            fence: 围栏区域

        Returns:
            围栏内的烟火列表
        """
        in_fence = []

        for det in detections:
            # 计算中心点
            cx = (det.x1 + det.x2) / 2
            cy = (det.y1 + det.y2) / 2

            # 检查是否在围栏内
            if fence.contains_point(cx, cy):
                in_fence.append(det)

        return in_fence

    def _visualize(
        self,
        frame: np.ndarray,
        all_dets: List[DetectionResult],
        in_fence_dets: List[DetectionResult],
        fence: FenceRegion,
        state: EventState
    ) -> np.ndarray:
        """
        可视化检测结果

        Args:
            frame: 原始帧
            all_dets: 所有检测到的烟火
            in_fence_dets: 围栏内的烟火
            fence: 围栏区域
            state: 当前状态

        Returns:
            可视化后的帧
        """
        img = frame.copy()

        # 绘制围栏 (根据状态改变颜色)
        fence_color = {
            EventState.IDLE: (0, 255, 0),       # 绿色 - 正常
            EventState.COOLDOWN: (255, 255, 0),  # 青色 - 冷却
            EventState.TRIGGERED: (0, 0, 255),   # 红色 - 触发
            EventState.ONGOING: (0, 0, 255),     # 红色 - 持续
            EventState.ENDING: (0, 165, 255),    # 橙色 - 结束中
        }.get(state, (128, 128, 128))

        fence.draw(img, color=fence_color, thickness=2)

        # 绘制所有检测 (灰色表示围栏外)
        for det in all_dets:
            if det not in in_fence_dets:
                color = self.CLASS_COLORS.get(det.class_id, (128, 128, 128))
                cv2.rectangle(
                    img,
                    (int(det.x1), int(det.y1)),
                    (int(det.x2), int(det.y2)),
                    (128, 128, 128), 1  # 灰色表示围栏外
                )

        # 绘制围栏内的检测 (彩色高亮)
        for det in in_fence_dets:
            color = self.CLASS_COLORS.get(det.class_id, (0, 0, 255))
            cv2.rectangle(
                img,
                (int(det.x1), int(det.y1)),
                (int(det.x2), int(det.y2)),
                color, 3
            )

            # 中心点
            cx = int((det.x1 + det.x2) / 2)
            cy = int((det.y1 + det.y2) / 2)
            cv2.circle(img, (cx, cy), 5, color, -1)

            # 标签
            label = det.class_name or self.CLASS_NAMES.get(det.class_id, f"class_{det.class_id}")
            label_text = f"{label}: {det.conf:.2f}"
            cv2.putText(
                img, label_text,
                (int(det.x1), int(det.y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        # 状态文字
        smoke_count = sum(1 for d in in_fence_dets if d.class_id == 0)
        fire_count = sum(1 for d in in_fence_dets if d.class_id == 1)
        status_text = f"State: {state.value} | Smoke: {smoke_count} Fire: {fire_count}"
        cv2.putText(
            img, status_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, fence_color, 2
        )

        return img

    def _create_event(
        self,
        context: DetectionContext,
        in_fence_dets: List[DetectionResult]
    ) -> DetectionEvent:
        """创建烟火事件"""
        objects = []
        for det in in_fence_dets:
            label = det.class_name or self.CLASS_NAMES.get(det.class_id, f"class_{det.class_id}")
            objects.append({
                "label": label,
                "confidence": round(det.conf, 2),
                "bbox": [int(det.x1), int(det.y1), int(det.x2), int(det.y2)],
                "status": "in_fence"
            })

        avg_conf = sum(d.conf for d in in_fence_dets) / len(in_fence_dets) if in_fence_dets else 0

        # 统计烟雾和火焰数量
        smoke_count = sum(1 for d in in_fence_dets if d.class_id == 0)
        fire_count = sum(1 for d in in_fence_dets if d.class_id == 1)

        return DetectionEvent(
            event_type=self.event_type,
            camera_id=context.camera_id,
            timestamp=context.timestamp,
            objects=objects,
            confidence=avg_conf,
            metadata={
                "smoke_count": smoke_count,
                "fire_count": fire_count,
                "total_in_fence": len(in_fence_dets),
                "fence_name": self._fences.get(context.camera_id, FenceRegion([])).name
            }
        )

    def clear_fence(self, camera_id: str):
        """清除围栏配置"""
        self._fences.pop(camera_id, None)
        self._state_machines.pop(camera_id, None)
