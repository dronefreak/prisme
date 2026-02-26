"""
prisme/benchmark.py

Benchmarks individual prisme tasks and the full multi-task stack on a
representative input (image or video).

Metrics reported (all AD-relevant):
  - Model load time          (seconds)
  - Warmup latency           (ms, first frame — JIT / CUDA graph overhead)
  - Mean / Median latency    (ms)
  - P95 / P99 latency        (ms)  ← tail latency matters for safety
  - Std-dev latency          (ms)  ← consistency / jitter
  - Throughput               (FPS)
  - Real-time capable        (FPS ≥ 30 Hz)
  - GPU VRAM — model load    (MB)  ← static model footprint
  - GPU VRAM — peak infer    (MB)  ← worst-case during a forward pass
  - CPU RSS delta            (MB)  ← host memory cost

Usage:
   python -m prisme.benchmark benchmark.runs=200
   python -m prisme.benchmark benchmark.random=true benchmark.random_width=3840 benchmark.random_height=2160
   python -m prisme.benchmark benchmark.tasks=[depth_estimation,lane_detection] benchmark.stack=true


Flags:
    benchmark.input     Path to video or image file (mutually exclusive with --random).
    benchmark.random    Generate random noise frames instead of loading from disk.
    benchmark.size      Frame size for random input, WxH (default: 1920x1080).
    benchmark.tasks     Space-separated task names, or 'all'.
    benchmark.warmup    Number of warmup frames before timing (default: 5).
    benchmark.runs      Number of timed frames per task (default: 50).
    benchmark.stack     Also benchmark the full multi-task stack together.
    benchmark.out       Optional path to write JSON results.
    benchmark.csv       Optional path to write CSV results.
    benchmark.no-table  Suppress Rich terminal table.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# ── Rich ───────────────────────────────────────────────────────────────────
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()

# ── Optional: psutil for CPU RAM ───────────────────────────────────────────
try:
    import psutil

    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# ── Task registry ──────────────────────────────────────────────────────────
ALL_TASKS = [
    "surface_normals",
    "object_detection",
    "semantic_segmentation",
    "depth_estimation",
    "panoptic_segmentation",
    "hybridnets",
    "pose_estimation",
]

_TASK_COLOURS = {
    "surface_normals": "cyan",
    "object_detection": "bright_yellow",
    "semantic_segmentation": "bright_green",
    "depth_estimation": "bright_magenta",
    "panoptic_segmentation": "bright_cyan",
    "hybridnets": "bright_blue",
    "pose_estimation": "bright_red",
    "FULL STACK": "bold white",
}


def _load_task(name: str, task_kwargs: dict | None = None):
    """Instantiate a task by name, forwarding kwargs from the pipeline config."""
    from prisme.tasks.surface_normals import SurfaceNormalsTask

    # from prisme.tasks.object_detection import ObjectDetectionTask
    from prisme.tasks.semantic_segmentation import SemanticSegmentationTask
    from prisme.tasks.depth_estimation import DepthEstimationTask
    from prisme.tasks.panoptic_segmentation import PanopticSegmentationTask
    from prisme.tasks.hybridnets import HybridNetsTask
    from prisme.tasks.pose_estimation import PoseEstimationTask

    registry = {
        "surface_normals": SurfaceNormalsTask,
        # "object_detection": ObjectDetectionTask,
        "semantic_segmentation": SemanticSegmentationTask,
        "depth_estimation": DepthEstimationTask,
        "panoptic_segmentation": PanopticSegmentationTask,
        "hybridnets": HybridNetsTask,
        "pose_estimation": PoseEstimationTask,
    }
    return registry[name](**(task_kwargs or {}))


# ── Frame loading ──────────────────────────────────────────────────────────


def _load_frames(input_path: str, n: int) -> list[np.ndarray]:
    """
    Load up to n BGR frames from a video or image.
    For video: sample evenly. For image: repeat.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(input_path)

    ext = p.suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Could not read image: {p}")
        return [img.copy() for _ in range(n)]

    # Video
    cap = cv2.VideoCapture(str(p))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // n)
    frames = []
    idx = 0
    while len(frames) < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += step
    cap.release()

    if not frames:
        raise ValueError("Could not read any frames from video.")
    # Pad by repeating last frame if video was shorter than n
    while len(frames) < n:
        frames.append(frames[-1].copy())
    return frames[:n]


def _generate_random_frames(width: int, height: int, n: int) -> list[np.ndarray]:
    """
    Generate n solid mid-grey BGR frames of the given size.

    Mid-grey (128, 128, 128) is a neutral input that avoids noise
    artifacts triggering spurious detections or unstable model outputs,
    giving a clean baseline for latency and VRAM measurement.

    All frames share the same underlying array — copies are intentionally
    avoided since we're measuring compute cost, not memory bandwidth.
    """
    frame = np.full((height, width, 3), 128, dtype=np.uint8)
    return [frame] * n


# ── GPU / CPU memory helpers ───────────────────────────────────────────────


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _vram_allocated_mb() -> float:
    if not _cuda_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def _vram_peak_mb() -> float:
    if not _cuda_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def _cpu_rss_mb() -> float:
    if not _PSUTIL:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def _cuda_sync():
    if _cuda_available():
        torch.cuda.synchronize()


def _reset_peak_vram():
    if _cuda_available():
        torch.cuda.reset_peak_memory_stats()


# ── Result dataclass ───────────────────────────────────────────────────────


@dataclass
class BenchResult:
    task: str
    load_time_s: float
    warmup_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    fps: float
    realtime: bool  # fps >= 30
    vram_model_mb: float  # VRAM after model load
    vram_peak_mb: float  # peak VRAM during timed inference
    cpu_rss_delta_mb: float
    n_runs: int
    errors: list[str] = field(default_factory=list)


# ── Core benchmark loop ────────────────────────────────────────────────────


def _benchmark_task(
    name: str,
    frames: list[np.ndarray],
    warmup: int,
    runs: int,
    task_kwargs: dict | None = None,
) -> BenchResult:
    """Benchmark a single task in isolation."""
    errors: list[str] = []

    # ── Baseline memory ────────────────────────────────────────────────────
    gc.collect()
    if _cuda_available():
        torch.cuda.empty_cache()
    _reset_peak_vram()

    cpu_before = _cpu_rss_mb()
    vram_before = _vram_allocated_mb()

    # ── Load model ─────────────────────────────────────────────────────────
    t_load_start = time.perf_counter()
    console.print(f"  [dim]Loading model for task '{name}'…[/dim]")
    console.print(f"  [dim]Task kwargs: {task_kwargs or {}}[/dim]")
    task = _load_task(name, task_kwargs)
    task.load_model()
    _cuda_sync()
    console.print("  [dim]Model loaded. Running inference to stabilise VRAM…[/dim]")
    load_time_s = time.perf_counter() - t_load_start

    vram_after_load = _vram_allocated_mb()
    vram_model_mb = max(0.0, vram_after_load - vram_before)

    # ── Warmup ─────────────────────────────────────────────────────────────
    warmup_ms = 0.0
    n_warmup = min(warmup, len(frames))
    for i in range(n_warmup):
        _cuda_sync()
        t0 = time.perf_counter()
        try:
            task.infer(frames[i % len(frames)])
        except Exception as e:
            errors.append(f"warmup[{i}]: {e}")
            break
        _cuda_sync()
        if i == 0:
            warmup_ms = (time.perf_counter() - t0) * 1000

    # ── Timed runs ─────────────────────────────────────────────────────────
    _reset_peak_vram()
    latencies_ms: list[float] = []
    n_runs = min(runs, len(frames))

    for i in range(n_runs):
        frame = frames[i % len(frames)]
        _cuda_sync()
        t0 = time.perf_counter()
        try:
            task.infer(frame)
        except Exception as e:
            errors.append(f"run[{i}]: {e}")
            continue
        _cuda_sync()
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    vram_peak = _vram_peak_mb()
    cpu_rss_delta = max(0.0, _cpu_rss_mb() - cpu_before)

    # ── Cleanup ────────────────────────────────────────────────────────────
    del task
    gc.collect()
    if _cuda_available():
        torch.cuda.empty_cache()

    # ── Stats ──────────────────────────────────────────────────────────────
    if not latencies_ms:
        # All runs failed
        return BenchResult(
            task=name,
            load_time_s=load_time_s,
            warmup_ms=warmup_ms,
            mean_ms=0,
            median_ms=0,
            p95_ms=0,
            p99_ms=0,
            std_ms=0,
            min_ms=0,
            max_ms=0,
            fps=0,
            realtime=False,
            vram_model_mb=vram_model_mb,
            vram_peak_mb=vram_peak,
            cpu_rss_delta_mb=cpu_rss_delta,
            n_runs=0,
            errors=errors,
        )

    arr = np.array(latencies_ms)
    mean_ms = float(arr.mean())
    median_ms = float(np.median(arr))
    p95_ms = float(np.percentile(arr, 95))
    p99_ms = float(np.percentile(arr, 99))
    std_ms = float(arr.std())
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return BenchResult(
        task=name,
        load_time_s=load_time_s,
        warmup_ms=warmup_ms,
        mean_ms=mean_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        std_ms=std_ms,
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        fps=fps,
        realtime=fps >= 30.0,
        vram_model_mb=vram_model_mb,
        vram_peak_mb=vram_peak,
        cpu_rss_delta_mb=cpu_rss_delta,
        n_runs=len(latencies_ms),
        errors=errors,
    )


def _benchmark_stack(
    task_names: list[str],
    frames: list[np.ndarray],
    warmup: int,
    runs: int,
    task_params: dict[str, dict] | None = None,
) -> BenchResult:
    """Benchmark all tasks running together (simulates real deployment)."""
    errors: list[str] = []
    task_params = task_params or {}

    gc.collect()
    if _cuda_available():
        torch.cuda.empty_cache()
    _reset_peak_vram()

    cpu_before = _cpu_rss_mb()
    vram_before = _vram_allocated_mb()

    # Load all tasks
    t_load = time.perf_counter()
    tasks = []
    for name in task_names:
        t = _load_task(name, task_params.get(name))
        t.load_model()
        tasks.append(t)
    _cuda_sync()
    load_time_s = time.perf_counter() - t_load

    vram_model_mb = max(0.0, _vram_allocated_mb() - vram_before)

    def _run_stack(frame):
        for t in tasks:
            t.infer(frame)

    # Warmup
    warmup_ms = 0.0
    for i in range(min(warmup, len(frames))):
        _cuda_sync()
        t0 = time.perf_counter()
        try:
            _run_stack(frames[i % len(frames)])
        except Exception as e:
            errors.append(f"warmup[{i}]: {e}")
            break
        _cuda_sync()
        if i == 0:
            warmup_ms = (time.perf_counter() - t0) * 1000

    _reset_peak_vram()
    latencies_ms: list[float] = []
    for i in range(min(runs, len(frames))):
        frame = frames[i % len(frames)]
        _cuda_sync()
        t0 = time.perf_counter()
        try:
            _run_stack(frame)
        except Exception as e:
            errors.append(f"run[{i}]: {e}")
            continue
        _cuda_sync()
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    vram_peak = _vram_peak_mb()
    cpu_rss_delta = max(0.0, _cpu_rss_mb() - cpu_before)

    del tasks
    gc.collect()
    if _cuda_available():
        torch.cuda.empty_cache()

    if not latencies_ms:
        return BenchResult(
            task="FULL STACK",
            load_time_s=load_time_s,
            warmup_ms=warmup_ms,
            mean_ms=0,
            median_ms=0,
            p95_ms=0,
            p99_ms=0,
            std_ms=0,
            min_ms=0,
            max_ms=0,
            fps=0,
            realtime=False,
            vram_model_mb=vram_model_mb,
            vram_peak_mb=vram_peak,
            cpu_rss_delta_mb=cpu_rss_delta,
            n_runs=0,
            errors=errors,
        )

    arr = np.array(latencies_ms)
    mean_ms = float(arr.mean())
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return BenchResult(
        task="FULL STACK",
        load_time_s=load_time_s,
        warmup_ms=warmup_ms,
        mean_ms=mean_ms,
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        std_ms=float(arr.std()),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        fps=fps,
        realtime=fps >= 30.0,
        vram_model_mb=vram_model_mb,
        vram_peak_mb=vram_peak,
        cpu_rss_delta_mb=cpu_rss_delta,
        n_runs=len(latencies_ms),
        errors=errors,
    )


# ── Rich output ────────────────────────────────────────────────────────────


def _fmt_ms(v: float) -> str:
    return f"{v:7.1f}"


def _fps_cell(r: BenchResult) -> Text:
    fps_str = f"{r.fps:6.1f}"
    rt_tag = (
        " [bold green]✔ RT[/bold green]" if r.realtime else " [bold red]✘[/bold red]"
    )
    return Text.from_markup(fps_str + rt_tag)


def _vram_cell(mb: float) -> str:
    if mb <= 0:
        return "[dim]—[/dim]"
    if mb >= 1024:
        return f"[bold]{mb / 1024:.2f} GB[/bold]"
    return f"{mb:.0f} MB"


def _print_table(results: list[BenchResult]) -> None:
    t = Table(
        title="[bold]prisme — Inference Benchmark[/bold]",
        caption="[dim]P95/P99 latency = tail latency under load (key for AD safety)[/dim]",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        row_styles=["", "dim"],
        expand=False,
    )

    t.add_column("Task", style="bold", no_wrap=True)
    t.add_column("Load (s)", justify="right")
    t.add_column("Warmup (ms)", justify="right")
    t.add_column("Mean (ms)", justify="right")
    t.add_column("Median (ms)", justify="right")
    t.add_column("P95 (ms)", justify="right", style="yellow")
    t.add_column("P99 (ms)", justify="right", style="red")
    t.add_column("Std (ms)", justify="right", style="dim")
    t.add_column("FPS / RT", justify="right")
    t.add_column("VRAM model", justify="right", style="cyan")
    t.add_column("VRAM peak", justify="right", style="bright_cyan")
    t.add_column("CPU RSS Δ", justify="right", style="dim")
    t.add_column("N", justify="right", style="dim")

    for r in results:
        colour = _TASK_COLOURS.get(r.task, "white")
        name_cell = f"[{colour}]{r.task}[/{colour}]"

        t.add_row(
            name_cell,
            f"{r.load_time_s:.2f}",
            _fmt_ms(r.warmup_ms),
            _fmt_ms(r.mean_ms),
            _fmt_ms(r.median_ms),
            _fmt_ms(r.p95_ms),
            _fmt_ms(r.p99_ms),
            _fmt_ms(r.std_ms),
            _fps_cell(r),
            _vram_cell(r.vram_model_mb),
            _vram_cell(r.vram_peak_mb),
            f"{r.cpu_rss_delta_mb:.0f} MB"
            if r.cpu_rss_delta_mb > 0
            else "[dim]—[/dim]",
            str(r.n_runs),
        )

    console.print(t)

    # Print any errors
    for r in results:
        if r.errors:
            console.print(f"[red]⚠  {r.task} errors:[/red]")
            for e in r.errors[:5]:
                console.print(f"   [dim]{e}[/dim]")


def _print_ad_notes(results: list[BenchResult]) -> None:
    """Print AD-specific observations after the table."""
    console.print()
    console.print(Rule("[bold]AD Suitability Notes[/bold]", style="dim"))
    for r in results:
        colour = _TASK_COLOURS.get(r.task, "white")
        issues = []

        if r.fps < 10:
            issues.append(
                f"[red]very slow ({r.fps:.1f} FPS) — unusable for online AD[/red]"
            )
        elif r.fps < 30:
            issues.append(
                f"[yellow]below real-time ({r.fps:.1f} FPS) — batch/async only[/yellow]"
            )

        # Tail latency jitter: p99/mean > 2x is a concern for deterministic systems
        if r.mean_ms > 0 and r.p99_ms / r.mean_ms > 2.5:
            issues.append(
                f"[yellow]high tail jitter (P99/mean = {r.p99_ms / r.mean_ms:.1f}×) "
                f"— unstable latency budget[/yellow]"
            )

        if r.vram_peak_mb > 4000:
            issues.append(
                f"[red]VRAM peak {r.vram_peak_mb:.0f} MB — may OOM in multi-task stack[/red]"
            )
        elif r.vram_peak_mb > 2000:
            issues.append(
                f"[yellow]VRAM peak {r.vram_peak_mb:.0f} MB — watch stack budget[/yellow]"
            )

        if not issues:
            line = (
                f"  [{colour}]✔ {r.task}[/{colour}]  [dim]looks good for AD stack[/dim]"
            )
        else:
            line = f"  [{colour}]{r.task}[/{colour}]  " + "  ".join(issues)

        console.print(line)


# ── Save helpers ───────────────────────────────────────────────────────────


def _save_json(results: list[BenchResult], path: str) -> None:
    data = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"[dim]JSON results → {path}[/dim]")


def _save_csv(results: list[BenchResult], path: str) -> None:
    if not results:
        return
    fieldnames = list(asdict(results[0]).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["errors"] = "; ".join(row["errors"])
            writer.writerow(row)
    console.print(f"[dim]CSV results  → {path}[/dim]")


# ── Entry point ────────────────────────────────────────────────────────────


@hydra.main(config_path="configs", config_name="example", version_base=None)
def main(cfg: DictConfig) -> None:
    bcfg = cfg.benchmark

    # ── Validate input source (mutual exclusivity enforced here, not in YAML) ─
    has_input = bcfg.input is not None
    use_random = bool(bcfg.random)
    if has_input and use_random:
        console.print(
            "[red]Error: set either benchmark.input or benchmark.random, not both.[/red]"
        )
        raise SystemExit(1)
    if not has_input and not use_random:
        console.print(
            "[red]Error: must set benchmark.input (path) or benchmark.random: true.[/red]"
        )
        raise SystemExit(1)

    # ── Resolve task list + params from pipeline config ────────────────────
    raw_tasks = OmegaConf.to_container(bcfg.tasks, resolve=True)
    task_names = ALL_TASKS if raw_tasks == ["all"] else list(raw_tasks)
    for t in task_names:
        if t not in ALL_TASKS:
            console.print(f"[red]Unknown task '{t}'. Available: {ALL_TASKS}[/red]")
            raise SystemExit(1)

    # Build name → kwargs map from the pipeline tasks: block so model size,
    # thresholds, etc. are respected exactly as configured for the pipeline.
    task_params: dict[str, dict] = {}
    if hasattr(cfg, "tasks"):
        for task_cfg in cfg.tasks:
            tc = OmegaConf.to_container(task_cfg, resolve=True)
            name = tc.pop("name")
            if name in task_names:
                task_params[name] = tc

    total_frames = bcfg.warmup + bcfg.runs

    # ── Print run header ───────────────────────────────────────────────────
    if use_random:
        w, h = int(bcfg.random_width), int(bcfg.random_height)
        input_label = f"[dim]solid grey[/dim] [bold]{w}×{h}[/bold]"
    else:
        input_label = f"[bold]{bcfg.input}[/bold]"

    console.print()
    console.print(Rule("[bold]prisme benchmark[/bold]"))
    console.print(f"  Input     : {input_label}")
    console.print(f"  Tasks     : [bold]{', '.join(task_names)}[/bold]")
    console.print(f"  Warmup    : [bold]{bcfg.warmup}[/bold] frames")
    console.print(f"  Timed runs: [bold]{bcfg.runs}[/bold] frames")
    console.print(f"  Stack     : [bold]{'yes' if bcfg.stack else 'no'}[/bold]")
    console.print(
        f"  GPU       : [bold]{'yes — ' + torch.cuda.get_device_name(0) if _cuda_available() else 'no (CPU only)'}[/bold]"
    )
    console.print(
        f"  psutil    : [bold]{'yes' if _PSUTIL else 'no (pip install psutil for CPU RAM)'}[/bold]"
    )
    console.print()

    # ── Load / generate frames ─────────────────────────────────────────────
    if use_random:
        with console.status(
            f"[dim]Generating {total_frames} solid-grey frames ({w}×{h})…[/dim]"
        ):
            frames = _generate_random_frames(w, h, total_frames)
        console.print(
            f"  [dim]Generated {len(frames)} frames "
            f"({frames[0].shape[1]}×{frames[0].shape[0]})[/dim]"
        )
    else:
        abs_input = to_absolute_path(bcfg.input)
        with console.status(f"[dim]Loading {total_frames} frames from input…[/dim]"):
            frames = _load_frames(abs_input, total_frames)
        console.print(
            f"  [dim]Loaded {len(frames)} frames "
            f"({frames[0].shape[1]}×{frames[0].shape[0]})[/dim]"
        )
    console.print()

    # ── Per-task benchmarks ────────────────────────────────────────────────
    results: list[BenchResult] = []

    for name in task_names:
        colour = _TASK_COLOURS.get(name, "white")
        with console.status(
            f"[{colour}]Benchmarking [bold]{name}[/bold]…[/{colour}]",
            spinner="dots",
        ):
            r = _benchmark_task(
                name,
                frames,
                warmup=bcfg.warmup,
                runs=bcfg.runs,
                task_kwargs=task_params.get(name),
            )

        rt_icon = (
            "[bold green]✔ RT[/bold green]" if r.realtime else "[bold red]✘[/bold red]"
        )
        console.print(
            f"  [{colour}]✔ {name}[/{colour}]  "
            f"mean=[bold]{r.mean_ms:.1f}ms[/bold]  "
            f"fps=[bold]{r.fps:.1f}[/bold] {rt_icon}  "
            f"vram_peak=[bold]{r.vram_peak_mb:.0f}MB[/bold]"
        )
        results.append(r)

    # ── Full stack benchmark ───────────────────────────────────────────────
    if bcfg.stack and len(task_names) > 1:
        with console.status(
            "[bold white]Benchmarking full stack…[/bold white]", spinner="dots"
        ):
            r = _benchmark_stack(
                task_names,
                frames,
                warmup=bcfg.warmup,
                runs=bcfg.runs,
                task_params=task_params,
            )
        console.print(
            f"  [bold white]✔ FULL STACK[/bold white]  "
            f"mean=[bold]{r.mean_ms:.1f}ms[/bold]  "
            f"fps=[bold]{r.fps:.1f}[/bold]  "
            f"vram_peak=[bold]{r.vram_peak_mb:.0f}MB[/bold]"
        )
        results.append(r)

    console.print()

    # ── Rich table + AD notes ──────────────────────────────────────────────
    if bcfg.show_table:
        _print_table(results)
        _print_ad_notes(results)

    # ── Save outputs ───────────────────────────────────────────────────────
    if bcfg.out:
        _save_json(results, to_absolute_path(bcfg.out))
    if bcfg.csv:
        _save_csv(results, to_absolute_path(bcfg.csv))

    console.print()


if __name__ == "__main__":
    main()
