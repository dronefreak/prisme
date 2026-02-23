"""
Module for the SurfaceNormalsTask.

Predicts surface normals from images using the DSINE model.
Weights are auto-downloaded to ~/.prisme/weights/surface_normals/ on first run.
"""

import urllib.request

import cv2
import numpy as np
import torch

from prisme.base import BaseTask
from utils.generic_utils import RichConsoleManager

DSINE_WEIGHTS_URL = "https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt"


class SurfaceNormalsTask(BaseTask):
    """Task for predicting surface normals from images using DSINE."""

    def __init__(self) -> None:
        """Initialize the SurfaceNormalsTask."""
        super().__init__(name="surface_normals")
        self.weights_path = self.weights_dir / "dsine.pt"
        self.console = RichConsoleManager.get_console()

    def _download_weights_if_missing(self) -> None:
        """Download DSINE weights if not already present."""
        if self.weights_path.exists():
            return
        self.console.print(
            f"[prisme] Downloading DSINE weights to {self.weights_path} ...",
            style="bold cyan",
        )
        urllib.request.urlretrieve(DSINE_WEIGHTS_URL, self.weights_path)  # noqa: S310 # nosec: B310
        self.console.print("[prisme] Download complete.", style="bold green")

    def _load(self) -> None:
        """Load the DSINE normal predictor model."""
        self.model = torch.hub.load(  # nosec: B614
            "hugoycj/DSINE-hub",
            "DSINE",
            local_file_path=str(self.weights_path),
            trust_repo=True,
        )

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Infer surface normals from the input BGR frame.

        Args:
            frame: Input image as a BGR numpy array (H, W, 3).

        Returns:
            Surface normals visualised as a BGR numpy array (H, W, 3).

        """
        self._ensure_model_loaded()
        with torch.inference_mode():
            normal = self.model.infer_cv2(frame)[0]  # (3, H, W)
            normal = (normal + 1) / 2  # Remap [-1, 1] -> [0, 1]
            normal = (normal * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
            normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
        return normal

    def _save(self, output: np.ndarray, output_path: str) -> None:
        """Save the surface normals output image to disk."""
        cv2.imwrite(output_path, output)
        self.console.print(
            f"[prisme] Output saved to {output_path}", style="bold green"
        )
