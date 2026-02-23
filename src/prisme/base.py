"""Base module for all prisme tasks."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

WEIGHTS_DIR = Path.home() / ".prisme" / "weights"


class BaseTask(ABC):
    """Base class for all tasks."""

    def __init__(self, name: str) -> None:
        """Initialize the task."""
        self.name = name
        self._model_loaded = False
        self.weights_dir = WEIGHTS_DIR / name
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self) -> None:
        """Download weights if missing, then load the model."""
        self._download_weights_if_missing()
        self._load()
        self._model_loaded = True

    def _download_weights_if_missing(self) -> None:
        """Download the weights if they are missing. Override in subclass if needed."""
        raise NotImplementedError

    @abstractmethod
    def _load(self) -> None:
        """Load the model. Must be implemented by subclass."""
        raise NotImplementedError

    @abstractmethod
    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Run inference on the input frame. Must be implemented by subclass."""
        raise NotImplementedError

    def _ensure_model_loaded(self) -> None:
        """Guard: raise if load_model() was never called."""
        if not self._model_loaded:
            raise RuntimeError(
                f"Model '{self.name}' is not loaded. Call load_model() before infer()."
            )

    def _save(self, output: np.ndarray, output_path: str) -> None:
        """Save the output to a file. Override in subclass if needed."""
        raise NotImplementedError
