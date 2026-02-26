"""
Module for orchestrating multi-task inference across image or video inputs.

The runner:
    1. Loads all tasks specified in the config and calls load_model() on each.
    2. Iterates over frames from the loader.
    3. Runs infer() on each task per frame.
    4. Prepends the original frame as slot 0.
    5. Tiles the outputs and writes the result to disk.
"""

import time
from pathlib import Path
from typing import List

import cv2
from omegaconf import DictConfig, OmegaConf
from rich.columns import Columns
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from prisme.io.loader import Loader
from prisme.base import BaseTask
from prisme.viz.tiler import tile, _grid_shape
from prisme.helpers.console_factory import RichConsoleManager

console = RichConsoleManager.get_console()


# ---------------------------------------------------------------------------
# Task colour map — one accent per task for the loading table
# ---------------------------------------------------------------------------
_TASK_COLOURS = {
    "surface_normals": "cyan",
    "object_detection": "bright_yellow",
    "semantic_segmentation": "bright_green",
    "depth_estimation": "bright_magenta",
    "panoptic_segmentation": "bright_cyan",
    "lane_detection": "bright_blue",
    "hybridnets": "bright_red",
    "pose_estimation": "bright_magenta",
}
_FALLBACK_COLOUR = "white"


def _task_colour(name: str) -> str:
    return _TASK_COLOURS.get(name, _FALLBACK_COLOUR)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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

    # from prisme.tasks.object_detection import ObjectDetectionTask
    from prisme.tasks.semantic_segmentation import SemanticSegmentationTask
    from prisme.tasks.depth_estimation import DepthEstimationTask
    from prisme.tasks.panoptic_segmentation import PanopticSegmentationTask
    from prisme.tasks.hybridnets import HybridNetsTask
    from prisme.tasks.pose_estimation import PoseEstimationTask

    TASK_REGISTRY = {
        "surface_normals": SurfaceNormalsTask,
        # "object_detection": ObjectDetectionTask,
        "semantic_segmentation": SemanticSegmentationTask,
        "depth_estimation": DepthEstimationTask,
        "panoptic_segmentation": PanopticSegmentationTask,
        "hybridnets": HybridNetsTask,
        "pose_estimation": PoseEstimationTask,
    }

    # ---- header -----------------------------------------------------------
    console.print(Rule("[bold]Loading Tasks[/bold]", style="dim"))

    tasks: List[BaseTask] = []
    for task_cfg in cfg.tasks:
        name = task_cfg.name
        if name not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{name}'. Available tasks: {list(TASK_REGISTRY.keys())}"
            )

        task_kwargs = {
            k: v for k, v in OmegaConf.to_container(task_cfg).items() if k != "name"
        }

        colour = _task_colour(name)
        # Params as a compact inline string
        params_str = "  ".join(
            f"[dim]{k}[/dim]=[bold]{v}[/bold]" for k, v in task_kwargs.items()
        )

        with console.status(
            f"[{colour}]Loading [bold]{name}[/bold]…[/{colour}]",
            spinner="dots",
        ):
            t0 = time.perf_counter()
            task = TASK_REGISTRY[name](**task_kwargs)
            task.load_model()
            elapsed = time.perf_counter() - t0

        console.print(
            f"  [{colour}]✔[/{colour}] [bold]{name}[/bold]"
            f"  [dim]({elapsed:.1f}s)[/dim]" + (f"  {params_str}" if params_str else "")
        )
        tasks.append(task)

    console.print()
    return tasks


def _print_run_summary(
    input_path: str,
    output_path: Path,
    meta: dict,
    tasks: List[BaseTask],
    out_width: int,
    out_height: int,
    rows: int,
    cols: int,
) -> None:
    """Print a Rich table summarising the run before frame processing starts."""
    # ---- Input / output panel -------------------------------------------
    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim", justify="right")
    info.add_column()

    info.add_row("Input", f"[bold path]{input_path}[/bold path]")
    info.add_row("Output", f"[bold path]{output_path}[/bold path]")

    if meta.get("frame_count"):
        info.add_row("Frames", f"[bold]{meta['frame_count']}[/bold]")
    if meta.get("fps"):
        info.add_row("FPS", f"[bold]{meta['fps']:.2f}[/bold]")
    if meta.get("width") and meta.get("height"):
        info.add_row(
            "Input res",
            f"[bold]{meta['width']} × {meta['height']}[/bold]",
        )

    info.add_row(
        "Output res",
        f"[bold]{out_width} × {out_height}[/bold]  [dim]({rows} × {cols} grid)[/dim]",
    )

    # ---- Task pills -------------------------------------------------------
    pills = []
    for t in tasks:
        name = t.name
        colour = _task_colour(name)
        pills.append(Text(f" {name} ", style=f"bold {colour} on grey15"))

    console.print(Rule("[bold]Pipeline Summary[/bold]", style="dim"))
    console.print(Panel(info, border_style="dim", padding=(0, 1)))
    console.print(Columns(pills, padding=(0, 1)))
    console.print()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


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

    # ---- Load input -------------------------------------------------------
    loader = Loader(input_path)
    meta = loader.metadata()

    # ---- Load tasks -------------------------------------------------------
    tasks = _resolve_tasks(cfg)
    n_tiles = len(tasks) + 1  # +1 for the original frame

    rows, cols = _grid_shape(n_tiles)
    out_width = cols * tile_width
    out_height = rows * tile_height

    _print_run_summary(
        input_path,
        output_path,
        meta,
        tasks,
        out_width,
        out_height,
        rows,
        cols,
    )

    # ---- Set up writer ----------------------------------------------------
    writer = None
    if loader.is_video:
        fps = meta.get("fps") or 30.0
        writer = _init_video_writer(output_path, out_width, out_height, fps)

    # ---- Frame loop with Rich progress bar --------------------------------
    frame_count = meta.get("frame_count") or 0  # 0 = unknown (image)
    total = frame_count if frame_count > 0 else None

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("[dim]eta[/dim]"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    console.print(Rule("[bold]Processing[/bold]", style="dim"))

    with progress:
        task_id = progress.add_task(
            "[bold bright_green]Rendering frames[/bold bright_green]",
            total=total,
        )

        for frame_idx, frame in loader.frames():
            task_outputs = [task.infer(frame) for task in tasks]
            tiles = [frame] + task_outputs
            tiled_frame = tile(tiles, tile_width=tile_width, tile_height=tile_height)

            if writer is not None:
                writer.write(tiled_frame)
            else:
                cv2.imwrite(str(output_path), tiled_frame)

            progress.advance(task_id)

    if writer is not None:
        writer.release()

    # ---- Done -------------------------------------------------------------
    console.print()
    console.print(
        Panel(
            f"[bold success]✔  Done[/bold success]  "
            f"[dim]→[/dim]  [bold path]{output_path}[/bold path]",
            border_style="success",
            padding=(0, 2),
        )
    )
