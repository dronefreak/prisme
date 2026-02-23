"""
Module for compositing multiple task output frames into a tiled grid.

Grid layout is determined by total tile count (tasks + original frame):
    2 tiles → 1x2
    3 tiles → 1x3
    4 tiles → 2x2
    5 tiles → 2x3
    6 tiles → 2x3
    7 tiles → 2x4
    8 tiles → 2x4

The original input frame is always included as slot 0 by the runner.
Maximum supported tasks: 7 (8 total tiles).
"""

from typing import List, Tuple

import cv2
import numpy as np

# Total tiles (tasks + original frame) → (rows, cols)
_GRID_LAYOUTS: dict[int, Tuple[int, int]] = {
    2: (1, 2),
    3: (1, 3),
    4: (2, 2),
    5: (2, 3),
    6: (2, 3),
    7: (2, 4),
    8: (2, 4),
}

MAX_TASKS = 7


def _grid_shape(n_total: int) -> Tuple[int, int]:
    """
    Return (rows, cols) for a given total number of tiles.

    Args:
        n_total: Total number of tiles including the original frame.

    Returns:
        A (rows, cols) tuple.

    Raises:
        ValueError: If n_total exceeds the maximum supported tile count.

    """
    min_task_count = 1  # At least 1 task in addition to original frame
    if n_total > MAX_TASKS + 1:
        raise ValueError(
            f"Too many tasks: got {n_total - 1} tasks ({n_total} total tiles). "
            f"Maximum supported is {MAX_TASKS} tasks ({MAX_TASKS + 1} total tiles)."
        )
    if n_total < min_task_count + 1:  # +1 for the original frame
        raise ValueError(
            "At least 1 task is required in addition to the original frame."
        )
    return _GRID_LAYOUTS[n_total]


def _resize_to(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize a frame to exactly (width, height).

    Args:
        frame: Input BGR numpy array.
        width: Target width in pixels.
        height: Target height in pixels.

    Returns:
        Resized BGR numpy array.

    """
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def tile(frames: List[np.ndarray], tile_width: int, tile_height: int) -> np.ndarray:
    """
    Compose a list of frames into a tiled grid image.

    The first frame in the list is expected to be the original input frame.
    All frames are resized to (tile_width, tile_height) before compositing.
    Empty slots in the grid are filled with black.

    Args:
        frames: List of BGR numpy arrays. frames[0] must be the original input frame.
        tile_width: Width of each individual tile in pixels.
        tile_height: Height of each individual tile in pixels.

    Returns:
        A single BGR numpy array containing the composited grid.

    Raises:
        ValueError: If frames list is empty or exceeds maximum supported count.

    """
    if not frames:
        raise ValueError("No frames provided to tile.")

    n = len(frames)
    rows, cols = _grid_shape(n)

    resized = [_resize_to(f, tile_width, tile_height) for f in frames]

    # Pad with black frames to fill the grid rectangle
    total_slots = rows * cols
    black = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    resized += [black] * (total_slots - n)

    # Build row by row
    grid_rows = []
    for r in range(rows):
        row_frames = resized[r * cols : (r + 1) * cols]
        grid_rows.append(np.concatenate(row_frames, axis=1))

    return np.concatenate(grid_rows, axis=0)
