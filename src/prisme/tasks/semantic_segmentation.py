"""
Module for the SemanticSegmentationTask.

Performs semantic segmentation using SegFormer-B2 fine-tuned on Cityscapes.
Weights are auto-downloaded from HuggingFace on first run.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from prisme.base import BaseTask

MODEL_ID = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"

# Standard Cityscapes 19-class palette (BGR for OpenCV)
# Order matches Cityscapes label IDs 0-18
CITYSCAPES_PALETTE_BGR = np.array(
    [
        [128, 64, 128],  # 0  road
        [232, 35, 244],  # 1  sidewalk
        [70, 70, 70],  # 2  building
        [156, 102, 102],  # 3  wall
        [153, 153, 190],  # 4  fence
        [153, 153, 153],  # 5  pole
        [30, 170, 250],  # 6  traffic light
        [0, 60, 220],  # 7  traffic sign
        [35, 142, 107],  # 8  vegetation
        [152, 251, 152],  # 9  terrain
        [180, 130, 70],  # 10 sky
        [60, 20, 220],  # 11 person
        [0, 0, 255],  # 12 rider
        [142, 0, 0],  # 13 car
        [70, 0, 0],  # 14 truck
        [100, 60, 0],  # 15 bus
        [100, 80, 0],  # 16 train
        [230, 0, 0],  # 17 motorcycle
        [32, 11, 119],  # 18 bicycle
    ],
    dtype=np.uint8,
)

# Module-level: just a string lookup, nothing loaded
MODEL_REGISTRY = {
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "b1": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
    "b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "b3": "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    "b4": "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}


class SemanticSegmentationTask(BaseTask):
    """Task for semantic segmentation using SegFormer-B2 on Cityscapes."""

    def __init__(
        self,
        resize_before_inference: int | None = None,
        segformer_model: str = "b2",
    ) -> None:
        """
        Initialise the SemanticSegmentationTask.

        Args:
            resize_before_inference: If set, resize the longer edge to this value
                before inference to reduce VRAM usage on low-end GPUs.
            segformer_model: SegFormer backbone variant to use (e.g., "b0", "b1",
                "b2"). Controls model capacity and performance/VRAM trade-off.

        """
        super().__init__(name="semantic_segmentation")
        self.resize_before_inference = resize_before_inference
        self.segformer_model = segformer_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _download_weights_if_missing(self) -> None:
        """HuggingFace auto-downloads weights on first use — nothing to do here."""
        pass

    def _load(self) -> None:
        """Load SegFormer processor and model from HuggingFace."""
        model_id = MODEL_REGISTRY[self.segformer_model]
        self.processor = SegformerImageProcessor.from_pretrained(model_id)  # nosec B615
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)  # nosec B615
        self.model.to(self.device)
        self.model.eval()

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment a BGR frame and return a colourised segmentation map.

        Args:
            frame: Input BGR numpy array (H, W, 3).

        Returns:
            Colourised segmentation map as a BGR numpy array (H, W, 3).

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

        # Convert BGR → RGB for the processor
        rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        # Upsample logits to inference frame size then argmax
        logits = outputs.logits  # (1, num_classes, H/4, W/4)
        upsampled = F.interpolate(
            logits,
            size=(inference_frame.shape[0], inference_frame.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        seg_map = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Map class IDs to Cityscapes BGR colours
        colour_map = CITYSCAPES_PALETTE_BGR[seg_map]  # (H, W, 3)

        # Resize back to original resolution
        if self.resize_before_inference is not None:
            colour_map = cv2.resize(
                colour_map, (original_w, original_h), interpolation=cv2.INTER_NEAREST
            )

        return colour_map

    def _save(self, output: np.ndarray, output_path: str) -> None:
        """Save the segmentation output image to disk."""
        cv2.imwrite(output_path, output)
