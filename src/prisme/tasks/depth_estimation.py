"""
Module for the DepthEstimationTask.

Performs monocular depth estimation using Depth Anything V2-Large.
Weights are auto-downloaded from HuggingFace on first run.
"""

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from prisme.base import BaseTask

MODEL_REGISTRY = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


class DepthEstimationTask(BaseTask):
    """Task for monocular depth estimation using Depth Anything V2."""

    def __init__(
        self,
        model_size: str = "large",
        colormap: int = cv2.COLORMAP_TURBO,
        resize_before_inference: int | None = None,
    ) -> None:
        """
        Initialise the DepthEstimationTask.

        Args:
            model_size: One of 'small', 'base', 'large'. Defaults to 'large'.
            colormap: OpenCV colormap applied to the normalised depth map.
                      Defaults to INFERNO (warm = close, cool = far).
            resize_before_inference: If set, resize the longer edge to this
                value before inference to reduce VRAM usage.

        """
        super().__init__(name="depth_estimation")

        if model_size not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_size '{model_size}'."
                f"Choose from {list(MODEL_REGISTRY.keys())}"
            )

        self.model_size = model_size
        self.colormap = colormap
        self.resize_before_inference = resize_before_inference
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _download_weights_if_missing(self) -> None:
        """HuggingFace auto-downloads weights on first use — nothing to do here."""
        pass

    def _load(self) -> None:
        """Load Depth Anything V2 processor and model from HuggingFace."""
        model_id = MODEL_REGISTRY[self.model_size]
        self.processor = AutoImageProcessor.from_pretrained(model_id)  # nosec B615
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)  # nosec B615
        self.model.to(self.device)
        self.model.eval()

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a BGR frame and return a colourised depth map.

        Args:
            frame: Input BGR numpy array (H, W, 3).

        Returns:
            Colourised depth map as a BGR numpy array (H, W, 3).
            Warm colours (yellow/white) = close, cool colours (dark) = far.

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

        # BGR → RGB for the processor
        from PIL import Image

        pil_image = Image.fromarray(cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB))

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            # predicted_depth: (1, H, W)
            depth = outputs.predicted_depth.squeeze(0).cpu().numpy()

        # Normalise to [0, 255]
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(
                np.uint8
            )
        else:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)

        # Apply colormap
        depth_coloured = cv2.applyColorMap(depth_norm, self.colormap)

        # Resize back to original resolution
        if self.resize_before_inference is not None:
            depth_coloured = cv2.resize(
                depth_coloured, (original_w, original_h), interpolation=cv2.INTER_LINEAR
            )

        return depth_coloured

    def _save(self, output: np.ndarray, output_path: str) -> None:
        """Save the depth estimation output image to disk."""
        cv2.imwrite(output_path, output)
