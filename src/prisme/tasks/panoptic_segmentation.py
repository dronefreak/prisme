"""
Module for the PanopticSegmentationTask.

Performs panoptic segmentation using Mask2Former fine-tuned on Cityscapes.
Panoptic = semantic segmentation (stuff) + instance segmentation (things).

Weights are auto-downloaded from HuggingFace on first run.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from prisme.base import BaseTask

MODEL_REGISTRY = {
    "small": "facebook/mask2former-swin-small-cityscapes-panoptic",
    "base": "facebook/mask2former-swin-base-IN21k-cityscapes-panoptic",
    "large": "facebook/mask2former-swin-large-cityscapes-panoptic",
}

# Cityscapes stuff classes (background/amorphous regions) — fixed semantic colours (BGR)
# Indexed by label_id from the model's id2label
STUFF_COLOURS_BGR = {
    "road": (128, 64, 128),
    "sidewalk": (232, 35, 244),
    "building": (70, 70, 70),
    "wall": (156, 102, 102),
    "fence": (153, 153, 190),
    "pole": (153, 153, 153),
    "traffic light": (30, 170, 250),
    "traffic sign": (0, 60, 220),
    "vegetation": (35, 142, 107),
    "terrain": (152, 251, 152),
    "sky": (180, 130, 70),
}

# Distinct colours for thing instances (BGR) — cycles if more instances than colours
THING_COLOURS_BGR = [
    (60, 20, 220),  # person
    (0, 0, 255),  # rider
    (142, 0, 0),  # car
    (70, 0, 0),  # truck
    (100, 60, 0),  # bus
    (100, 80, 0),  # train
    (230, 0, 0),  # motorcycle
    (32, 11, 119),  # bicycle
    (0, 200, 200),
    (200, 0, 200),
    (0, 200, 0),
    (200, 200, 0),
]


# Cityscapes "things" — countable instances with distinct per-instance colours
# Fixed by the Cityscapes dataset definition, not the model config
CITYSCAPES_THING_CLASSES = {
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
}


def _random_colour_for_instance(instance_id: int) -> tuple:
    """Return a deterministic colour for a given instance id."""
    return THING_COLOURS_BGR[instance_id % len(THING_COLOURS_BGR)]


class PanopticSegmentationTask(BaseTask):
    """Task for panoptic segmentation using Mask2Former on Cityscapes."""

    def __init__(
        self,
        model_size: str = "large",
        alpha: float = 0.6,
        resize_before_inference: int | None = None,
    ) -> None:
        """
        Initialise the PanopticSegmentationTask.

        Args:
            model_size: One of 'small', 'base', 'large'. Defaults to 'large'.
            alpha: Blend weight for overlay (0.0 = original, 1.0 = full mask).
            resize_before_inference: If set, resize the longer edge to this value
                before inference to reduce VRAM usage.

        """
        super().__init__(name="panoptic_segmentation")

        if model_size not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_size '{model_size}'."
                f" Choose from {list(MODEL_REGISTRY.keys())}"
            )

        self.model_size = model_size
        self.alpha = alpha
        self.resize_before_inference = resize_before_inference
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _download_weights_if_missing(self) -> None:
        """HuggingFace auto-downloads weights on first use — nothing to do here."""
        pass

    def _load(self) -> None:
        """Load Mask2Former processor and model from HuggingFace."""
        model_id = MODEL_REGISTRY[self.model_size]  # nosec B615
        self.processor = AutoImageProcessor.from_pretrained(model_id)  # nosec B615
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)  # nosec B615
        self.model.to(self.device)
        self.model.eval()

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Run panoptic segmentation on a BGR frame.

        Args:
            frame: Input BGR numpy array (H, W, 3).

        Returns:
            Panoptic segmentation overlay as a BGR numpy array (H, W, 3).
            Stuff regions use fixed Cityscapes colours.
            Thing instances use per-instance colours.

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

        pil_image = Image.fromarray(cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB))

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        # Post-process panoptic segmentation
        # Returns list of dicts with 'segmentation' tensor and 'segments_info'
        result = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[(inference_frame.shape[0], inference_frame.shape[1])],
            label_ids_to_fuse=set(),
        )[0]

        seg_map = (
            result["segmentation"].cpu().numpy()
        )  # (H, W) — each pixel = segment_id
        segments_info = result["segments_info"]

        # Build colour canvas
        colour_map = np.zeros(
            (inference_frame.shape[0], inference_frame.shape[1], 3), dtype=np.uint8
        )

        for segment in segments_info:
            seg_id = segment["id"]
            label_id = segment["label_id"]
            label_name = self.model.config.id2label.get(label_id, "unknown")
            is_thing = label_name in CITYSCAPES_THING_CLASSES

            mask = seg_map == seg_id

            if is_thing:
                colour = _random_colour_for_instance(seg_id)
            else:
                colour = STUFF_COLOURS_BGR.get(label_name, (128, 128, 128))

            colour_map[mask] = colour

        # Blend with original frame for context
        overlay = cv2.addWeighted(
            inference_frame, 1 - self.alpha, colour_map, self.alpha, 0
        )

        # Resize back to original resolution
        if self.resize_before_inference is not None:
            overlay = cv2.resize(
                overlay, (original_w, original_h), interpolation=cv2.INTER_LINEAR
            )

        return overlay

    def _save(self, output: np.ndarray, output_path: str) -> None:
        """Save the panoptic segmentation output image to disk."""
        cv2.imwrite(output_path, output)
