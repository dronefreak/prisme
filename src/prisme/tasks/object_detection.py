"""
Module for the ObjectDetectionTask.

Performs real-time object detection using RF-DETR (RFDETRBase),
filtering outputs to AD-relevant classes only.
"""

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETRNano, RFDETRSegMedium, RFDETRBase, RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES

from prisme.base import BaseTask

# AD-relevant COCO class names to keep
AD_CLASSES = {
    "car",
    "person",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "traffic light",
    "stop sign",
}

# Map class name → supervision Color for consistent AD palette
AD_COLOR_MAP = {
    "car": sv.Color.from_hex("#00B4D8"),  # blue
    "person": sv.Color.from_hex("#F72585"),  # pink
    "truck": sv.Color.from_hex("#4CC9F0"),  # light blue
    "bus": sv.Color.from_hex("#7209B7"),  # purple
    "bicycle": sv.Color.from_hex("#4CAF50"),  # green
    "motorcycle": sv.Color.from_hex("#FF9800"),  # orange
    "traffic light": sv.Color.from_hex("#FFD166"),  # yellow
    "stop sign": sv.Color.from_hex("#EF233C"),  # red
}

MODEL_REGISTRY = {
    "nano": RFDETRNano,
    "medium": RFDETRSegMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}


class ObjectDetectionTask(BaseTask):
    """
    Task for real-time object detection using RF-DETR.

    Only detects classes relevant to autonomous driving and annotates them on the frame.
    """

    def __init__(
        self,
        threshold: float = 0.4,
        resize_before_inference: int | None = None,
        rfdetr_model: str = "base",
    ) -> None:
        """
        Initialise the ObjectDetectionTask.

        Args:
            threshold: Confidence threshold for filtering detections.
            resize_before_inference: If set, resize the shorter edge to this value
                before inference to reduce VRAM usage on low-end GPUs.
            rfdetr_model: RF-DETR backbone variant to use (e.g., "base", "large").
                Controls model capacity, accuracy, and computational cost.

        """
        super().__init__(name="object_detection")
        self.threshold = threshold
        self.resize_before_inference = resize_before_inference
        self.rfdetr_model = rfdetr_model
        if rfdetr_model not in MODEL_REGISTRY:
            raise ValueError(
                f"Invalid RF-DETR model '{rfdetr_model}'."
                f"Valid options: {list(MODEL_REGISTRY.keys())}"
            )

    def _download_weights_if_missing(self) -> None:
        """RF-DETR auto-downloads weights on first use — nothing to do here."""
        pass

    def _load(self) -> None:
        """Load RFDETRBase and optimise for inference."""
        self.model = MODEL_REGISTRY[self.rfdetr_model]()
        self.model.optimize_for_inference()

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect AD-relevant objects in a BGR frame.

        Args:
            frame: Input BGR numpy array (H, W, 3).

        Returns:
            Annotated BGR numpy array with bounding boxes and labels.

        """
        self._ensure_model_loaded()

        original_h, original_w = frame.shape[:2]

        # Optionally resize for VRAM management
        if self.resize_before_inference is not None:
            scale = self.resize_before_inference / max(original_h, original_w)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            inference_frame = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            inference_frame = frame

        # RF-DETR expects a PIL Image in RGB
        pil_image = Image.fromarray(cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB))

        detections = self.model.predict(pil_image, threshold=self.threshold)

        # Filter to AD-relevant classes only
        if len(detections) > 0:
            ad_mask = np.array(
                [
                    COCO_CLASSES[class_id] in AD_CLASSES
                    for class_id in detections.class_id
                ]
            )
            detections = detections[ad_mask]

        # Annotate on a copy of the original (pre-resize) frame
        output = frame.copy()

        if len(detections) > 0:
            # Scale boxes back to original resolution if resized
            if self.resize_before_inference is not None:
                detections.xyxy = detections.xyxy / scale

            labels = [
                f"{COCO_CLASSES[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(
                    detections.class_id, detections.confidence
                )
            ]

            # Annotate — convert to PIL for supervision, then back to BGR numpy
            output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            output_pil = sv.BoxAnnotator().annotate(output_pil, detections)
            output_pil = sv.LabelAnnotator().annotate(output_pil, detections, labels)
            output = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)

        return output

    def _save(self, output: np.ndarray, output_path: str) -> None:
        """Save the annotated output frame to disk."""
        cv2.imwrite(output_path, output)
