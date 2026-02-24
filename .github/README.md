# prisme 🔷

> One input. Multiple perspectives.

**prisme** is a unified computer vision inference library for research and visualisation. Give it an image or video, configure your tasks in a single YAML file, and get a tiled output with all task results side by side — no stitching pipelines together manually.

Built for AD scene understanding but useful for any outdoor scene analysis.

---

## Demo

<!-- VIDEO PLACEHOLDER — replace with actual demo gif/video link -->
![Demo](assets/demo.gif)

---

## Installation

```bash
git clone https://github.com/yourusername/prisme.git
cd prisme
pip install -e .
```

Weights are downloaded automatically on first run to `~/.prisme/weights/`.

---

## Usage

Edit `src/prisme/configs/example.yaml`:

```yaml
input: /path/to/your/video.mp4
output: /path/to/output.mp4
tile_width: 640
tile_height: 480

tasks:
  - name: surface_normals
    resize_before_inference: 1280

  - name: object_detection
    threshold: 0.4
    resize_before_inference: 1280

  - name: semantic_segmentation
    segformer_model: b2
    resize_before_inference: 1280

  - name: depth_estimation
    model_size: large
    resize_before_inference: 1280
```

Then run:

```bash
prisme input=/path/to/video.mp4 output=/path/to/output.mp4
```

Any config value can be overridden inline via Hydra.

---

## Supported Tasks

| Task | Model | Source |
|------|-------|--------|
| Surface Normals | [DSINE](https://github.com/hugoycj/DSINE-hub) | `torch.hub` |
| Object Detection | [RF-DETR Base](https://github.com/roboflow/rf-detr) | `pip install rfdetr` |
| Semantic Segmentation | [SegFormer-B0 to B5](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024) — Cityscapes | HuggingFace |
| Depth Estimation | [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) | HuggingFace |

All models are inference-only. No training code.

---

## Output Grid Layout

The original frame always occupies slot 0. Tasks fill the remaining slots in config order.

| Tasks | Grid |
|-------|------|
| 1 | 1×2 |
| 2 | 1×3 |
| 3 | 2×2 |
| 4 | 2×3 |
| 5 | 2×3 |
| 6 | 2×4 |
| 7 | 2×4 |

---

## Adding a New Task

1. Create `src/prisme/tasks/your_task.py` subclassing `BaseTask`:

```python
from prisme.base import BaseTask
import numpy as np

class YourTask(BaseTask):
    def __init__(self, resize_before_inference=None):
        super().__init__(name="your_task")
        self.resize_before_inference = resize_before_inference

    def _download_weights_if_missing(self):
        pass  # or implement auto-download

    def _load(self):
        self.model = ...  # load your model here

    def infer(self, frame: np.ndarray) -> np.ndarray:
        self._ensure_model_loaded()
        # run inference, return BGR numpy array
        return result
```

2. Register it in `runner.py`:

```python
from prisme.tasks.your_task import YourTask

TASK_REGISTRY = {
    ...
    "your_task": YourTask,
}
```

3. Add it to your config:

```yaml
tasks:
  - name: your_task
    resize_before_inference: 1280
```

That's it.

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA recommended (4GB VRAM minimum with `resize_before_inference`)

---

## License

MIT

---

As always, Hare Krishna and happy coding! 🙏
