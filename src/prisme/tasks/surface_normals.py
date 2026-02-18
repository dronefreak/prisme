"""
Module for the SurfaceNormalsTask.

Predicts surface normals from images using a pre-trained model.
"""

import torch
import cv2
import numpy as np
from prisme.base import BaseTask


class SurfaceNormalsTask(BaseTask):
    """Task for predicting surface normals from images."""

    def __init__(self) -> None:
        """Initialize the SurfaceNormalsTask."""
        super().__init__(name="surface_normals")

    def _load(self) -> None:
        """Load the normal predictor model."""
        self.model = torch.hub.load(  # nosec: B614
            "hugoycj/DSINE-hub",
            "DSINE",
            local_file_path="./checkpoints/dsine.pt",
            trust_repo=True,
        )

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Infer the surface normals from the input image."""
        with torch.inference_mode():
            normal = self.model.infer_cv2(frame)[0]  # Output shape: (H, W, 3)
            normal = (normal + 1) / 2  # Convert values to the range [0, 1]
            normal = (normal * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
            normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
        return normal
