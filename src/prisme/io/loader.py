"""
Module for loading images and videos as frame iterators.

Provides a unified interface regardless of input type — both images
and videos are iterated frame by frame.
"""

from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class Loader:
    """Loads an image or video file and exposes a unified frame iterator."""

    def __init__(self, path: str) -> None:
        """
        Initialize the Loader with a path to an image or video file.

        Args:
            path: Path to the input image or video file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.

        """
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Input file not found: {self.path}")

        ext = self.path.suffix.lower()
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            self._is_video = False
        elif ext in SUPPORTED_VIDEO_EXTENSIONS:
            self._is_video = True
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS}"
            )

    @property
    def is_video(self) -> bool:
        """Return True if the input is a video file."""
        return self._is_video

    def metadata(self) -> dict:
        """
        Return basic metadata about the input.

        For images: width, height.
        For videos: width, height, fps, frame_count.
        """
        if not self._is_video:
            frame = cv2.imread(str(self.path))
            h, w = frame.shape[:2]
            return {"width": w, "height": h, "fps": None, "frame_count": 1}

        cap = cv2.VideoCapture(str(self.path))
        meta = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        return meta

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Iterate over frames as (frame_index, bgr_frame) tuples.

        Yields:
            A tuple of (frame_index, frame) where frame is a BGR numpy array.

        Raises:
            RuntimeError: If the video file cannot be opened or a frame cannot be read.

        """
        if not self._is_video:
            frame = cv2.imread(str(self.path))
            if frame is None:
                raise RuntimeError(f"Failed to read image: {self.path}")
            yield 0, frame
            return

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame_idx, frame
                frame_idx += 1
        finally:
            cap.release()
