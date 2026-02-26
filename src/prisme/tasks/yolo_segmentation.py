"""
prisme/tasks/yolo_segmentation.py

Instance segmentation using ultralytics YOLO-seg.

Produces per-instance coloured masks overlaid on the frame, with optional
bounding boxes and class labels. Unlike semantic segmentation (SegFormer),
this gives you individual object masks — every car, person, cyclist is a
separate coloured region.

Supports all YOLO -seg model variants (v8-seg, v11-seg).
Weights auto-download from ultralytics on first use.

Input:  BGR uint8, any resolution
Output: BGR uint8 with instance masks + boxes at original resolution
"""

from __future__ import annotations

import cv2
import numpy as np

from prisme.base import BaseTask


def _class_colour(class_id: int) -> tuple[int, int, int]:
    np.random.seed(class_id + 42)
    return tuple(int(x) for x in np.random.randint(80, 255, 3))


YOLO_SEG_MODELS = {
    # YOLOv8-seg
    "yolov8n-seg",
    "yolov8s-seg",
    "yolov8m-seg",
    "yolov8l-seg",
    "yolov8x-seg",
    # YOLO11-seg
    "yolo11n-seg",
    "yolo11s-seg",
    "yolo11m-seg",
    "yolo11l-seg",
    "yolo11x-seg",
}


class YOLOSegmentationTask(BaseTask):
    """
    Instance segmentation using ultralytics YOLO-seg.

    Args:
        model:          YOLO-seg model variant (default: yolov8n-seg).
        conf:           Detection confidence threshold.
        iou:            NMS IoU threshold.
        mask_alpha:     Opacity of the instance mask overlay (0–1).
        show_boxes:     Draw bounding boxes around instances.
        show_labels:    Draw class name on each instance.
        show_conf:      Append confidence score to label.
        line_thickness: Bounding box line thickness in pixels.

    """

    def __init__(
        self,
        model: str = "yolov8n-seg",
        conf: float = 0.25,
        iou: float = 0.45,
        mask_alpha: float = 0.45,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_conf: bool = True,
        line_thickness: int = 2,
    ) -> None:
        super().__init__(name="yolo_segmentation")
        model_key = model.removesuffix(".pt")
        if model_key not in YOLO_SEG_MODELS:
            raise ValueError(
                f"Unknown YOLO seg model '{model}'. "
                f"Available: {sorted(YOLO_SEG_MODELS)}"
            )
        self.model_name = model_key + ".pt"
        self.conf = conf
        self.iou = iou
        self.mask_alpha = mask_alpha
        self.show_boxes = show_boxes
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
        names = self._model.names

        for r in results:
            if r.masks is None:
                continue

            masks = r.masks.data.cpu().numpy()  # (N, H, W) — model resolution
            boxes = r.boxes
            ori_h, ori_w = frame.shape[:2]

            for i, (mask, box) in enumerate(zip(masks, boxes)):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                colour = _class_colour(cls_id)

                # Resize mask to original frame resolution
                mask_resized = cv2.resize(
                    mask, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                # Colour fill
                colour_layer = np.zeros_like(out)
                colour_layer[mask_resized] = colour
                out = cv2.addWeighted(out, 1.0, colour_layer, self.mask_alpha, 0)

                # Mask contour for crisp edges
                contours, _ = cv2.findContours(
                    mask_resized.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(out, contours, -1, colour, 1, cv2.LINE_AA)

                # Optional bounding box
                if self.show_boxes:
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    cv2.rectangle(out, (x1, y1), (x2, y2), colour, self.line_thickness)

                    if self.show_labels:
                        label = names[cls_id]
                        if self.show_conf:
                            label = f"{label} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        cv2.rectangle(
                            out, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1
                        )
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
