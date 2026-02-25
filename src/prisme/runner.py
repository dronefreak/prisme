"""
Module for orchestrating multi-task inference across image or video inputs.

The runner:
    1. Loads all tasks specified in the config and calls load_model() on each.
    2. Iterates over frames from the loader.
    3. Runs infer() on each task per frame.
    4. Prepends the original frame as slot 0.
    5. Tiles the outputs and writes the result to disk.
"""

from pathlib import Path
from typing import List

import cv2
from omegaconf import DictConfig, OmegaConf

from prisme.io.loader import Loader
from prisme.base import BaseTask
from prisme.viz.tiler import tile, _grid_shape


def _init_video_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    """
    Initialise a VideoWriter for the tiled output.

    Args:
        output_path: Path to the output video file.
        width: Total width of the tiled frame in pixels.
        height: Total height of the tiled frame in pixels.
        fps: Frames per second of the output video.

    Returns:
        An opened cv2.VideoWriter instance.

    Raises:
        RuntimeError: If the VideoWriter fails to open.

    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at {output_path}")
    return writer


def _resolve_tasks(cfg: DictConfig) -> List[BaseTask]:
    """
    Instantiate and load all tasks declared in the config.

    Each task entry in cfg.tasks must have a 'name' field matching a
    registered task class. Task-level params (everything except 'name')
    are forwarded as kwargs to the task constructor.

    Args:
        cfg: Hydra config object.

    Returns:
        List of loaded BaseTask instances.

    Raises:
        ValueError: If an unknown task name is provided.

    """
    from prisme.tasks.surface_normals import SurfaceNormalsTask
    from prisme.tasks.object_detection import ObjectDetectionTask
    from prisme.tasks.semantic_segmentation import SemanticSegmentationTask
    from prisme.tasks.depth_estimation import DepthEstimationTask
    from prisme.tasks.panoptic_segmentation import PanopticSegmentationTask

    TASK_REGISTRY = {
        "surface_normals": SurfaceNormalsTask,
        "object_detection": ObjectDetectionTask,
        "semantic_segmentation": SemanticSegmentationTask,
        "depth_estimation": DepthEstimationTask,
        "panoptic_segmentation": PanopticSegmentationTask,
    }
    tasks: List[BaseTask] = []
    for task_cfg in cfg.tasks:
        name = task_cfg.name
        if name not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{name}'. Available tasks: {list(TASK_REGISTRY.keys())}"
            )

        # Forward all task-level params except 'name' to the constructor
        task_kwargs = {
            k: v for k, v in OmegaConf.to_container(task_cfg).items() if k != "name"
        }

        task = TASK_REGISTRY[name](**task_kwargs)
        print(f"[prisme] Loading model for task: {name} | params: {task_kwargs}")
        task.load_model()
        tasks.append(task)

    return tasks


def run(cfg: DictConfig) -> None:
    """
    Run the full prisme pipeline from config.

    Args:
        cfg: Hydra config object containing input, output, and task definitions.

    """
    input_path = cfg.input
    output_path = Path(cfg.output)
    tile_width = cfg.get("tile_width", 640)
    tile_height = cfg.get("tile_height", 480)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input
    loader = Loader(input_path)
    meta = loader.metadata()
    print(f"[prisme] Input: {input_path} | {meta}")

    # Load tasks
    tasks = _resolve_tasks(cfg)
    n_tiles = len(tasks) + 1  # +1 for original frame

    # Compute final output dimensions from grid
    rows, cols = _grid_shape(n_tiles)
    out_width = cols * tile_width
    out_height = rows * tile_height

    # Set up writer
    writer = None
    if loader.is_video:
        fps = meta["fps"] or 30.0
        writer = _init_video_writer(output_path, out_width, out_height, fps)

    print(
        f"[prisme] Running {len(tasks)} task(s) on "
        f"{meta.get('frame_count', 1)} frame(s)..."
    )

    for frame_idx, frame in loader.frames():
        task_outputs = [task.infer(frame) for task in tasks]
        tiles = [frame] + task_outputs
        tiled_frame = tile(tiles, tile_width=tile_width, tile_height=tile_height)

        if writer is not None:
            writer.write(tiled_frame)
        else:
            # Image input — save single frame
            cv2.imwrite(str(output_path), tiled_frame)

        if frame_idx % 50 == 0:
            print(f"[prisme] Processed frame {frame_idx}")

    if writer is not None:
        writer.release()

    print(f"[prisme] Done. Output saved to {output_path}")
