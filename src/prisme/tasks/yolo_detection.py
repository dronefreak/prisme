"""
prisme/tasks/yolo_detection.py

Object detection using ultralytics YOLO.

Lighter and faster than RF-DETR — intended for deployment on embedded
hardware (Jetson Orin etc.) where RF-DETR is too slow for online AD.

Supports all YOLO detection model variants (v8, v9, v10, v11).
Weights auto-download from ultralytics on first use.

Input:  BGR uint8, any resolution
Output: BGR uint8 with bounding boxes and class labels at original resolution
"""

from __future__ import annotations

import cv2
import numpy as np

from prisme.base import BaseTask


# COCO 80-class colour palette — deterministic per class index
def _class_colour(class_id: int) -> tuple[int, int, int]:
    np.random.seed(class_id + 42)
    return tuple(int(x) for x in np.random.randint(80, 255, 3))


YOLO_DETECTION_MODELS = {
    # YOLOv8
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    # YOLOv9
    "yolov9c",
    "yolov9e",
    # YOLOv10
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov10l",
    "yolov10x",
    # YOLO11
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
}


class YOLODetectionTask(BaseTask):
    """
    Object detection using ultralytics YOLO.

    Args:
        model:          YOLO model variant (default: yolov8n).
        conf:           Detection confidence threshold.
        iou:            NMS IoU threshold.
        show_labels:    Draw class name on each box.
        show_conf:      Append confidence score to label.
        line_thickness: Bounding box line thickness in pixels.

    """

    def __init__(
        self,
        model: str = "yolov8n",
        conf: float = 0.25,
        iou: float = 0.45,
        show_labels: bool = True,
        show_conf: bool = True,
        line_thickness: int = 2,
    ) -> None:
        super().__init__(name="yolo_detection")
        model_key = model.removesuffix(".pt")
        if model_key not in YOLO_DETECTION_MODELS:
            raise ValueError(
                f"Unknown YOLO detection model '{model}'. "
                f"Available: {sorted(YOLO_DETECTION_MODELS)}"
            )
        self.model_name = model_key + ".pt"
        self.conf = conf
        self.iou = iou
        self.show_labels = show_labels
        self.show_conf = show_conf
        self.line_thickness = line_thickness
        self._model = None

    def _download_weights_if_missing(self) -> None:
        # ultralytics automatically downloads weights on first load, so we can skip this step
        pass

    def _load(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self.model_name)

    def infer(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._model.predict(
            rgb,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        return self._visualise(frame, results)

    def _visualise(self, frame: np.ndarray, results) -> np.ndarray:
        out = frame.copy()
        names = self._model.names  # id → class string

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                colour = _class_colour(cls_id)

                cv2.rectangle(out, (x1, y1), (x2, y2), colour, self.line_thickness)

                if self.show_labels:
                    label = names[cls_id]
                    if self.show_conf:
                        label = f"{label} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
                    cv2.putText(
                        out,
                        label,
                        (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
        return out
