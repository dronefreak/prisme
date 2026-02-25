"""
prisme/tasks/hybridnets.py

Single-forward-pass task combining drivable area segmentation and lane
detection using HybridNets (EfficientNet-B3 backbone, BDD100K weights).

Replaces lane_detection (UFLDv2). One backbone, two outputs:
  - Green semi-transparent overlay for drivable area (seg class 1: road)
  - Yellow mask overlay for lane lines       (seg class 2: lane)

Detection head output is intentionally discarded — RF-DETR handles detection.

Model is loaded via torch.hub (datvuthanh/hybridnets), which clones the repo
on first use and caches it under torch hub's default directory.
Weights (~13 MB) are downloaded automatically by the hub entrypoint.

Input:  BGR uint8, any resolution
Output: BGR uint8 visualisation at original resolution
"""

from __future__ import annotations


import cv2
import numpy as np
import torch
import torch.nn.functional as F

from prisme.base import BaseTask

# ── Constants (BDD100K training config) ────────────────────────────────────
_INPUT_W = 640
_INPUT_H = 384
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Segmentation class indices in the HybridNets output
_SEG_ROAD = 1  # drivable area
_SEG_LANE = 2  # lane lines

# Visualisation colours (BGR)
_COL_ROAD = (0, 200, 0)  # green
_COL_LANE = (0, 220, 220)  # yellow


class HybridNetsTask(BaseTask):
    """
    Drivable area + lane detection using HybridNets.

    Args:
        road_alpha:     Blend weight for the drivable area overlay (0–1).
        lane_alpha:     Blend weight for the lane line overlay (0–1).
        conf_road:      Softmax probability threshold for road class.
        conf_lane:      Softmax probability threshold for lane class.
        resize_before_inference: Longer edge cap before inference (0 = no cap).

    """

    def __init__(
        self,
        road_alpha: float = 0.4,
        lane_alpha: float = 0.6,
        conf_road: float = 0.5,
        conf_lane: float = 0.5,
        resize_before_inference: int = 0,
    ) -> None:
        super().__init__(name="hybridnets")
        self.road_alpha = road_alpha
        self.lane_alpha = lane_alpha
        self.conf_road = conf_road
        self.conf_lane = conf_lane
        self.resize_cap = resize_before_inference
        self._model = None
        self._device = None

    # ── BaseTask interface ──────────────────────────────────────────────────

    def _download_weights_if_missing(self) -> None:
        """torch.hub auto-downloads HybridNets weights on first use — nothing to do here."""
        pass

    def _load(self) -> None:
        self._ensure_deps()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = torch.hub.load(
            "datvuthanh/hybridnets",
            "hybridnets",
            pretrained=True,
            verbose=False,
        )
        self._model.to(self._device).eval()

    def infer(self, frame: np.ndarray) -> np.ndarray:
        ori_h, ori_w = frame.shape[:2]

        # Optional resize guard
        work = self._maybe_resize(frame)
        inp = self._preprocess(work)

        with torch.no_grad():
            _, _, _, _, segmentation = self._model(inp)

        road_mask, lane_mask = self._decode_seg(segmentation, ori_h, ori_w)
        return self._visualise(frame, road_mask, lane_mask)

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _ensure_deps() -> None:
        """Install torch-hub requirements for HybridNets if absent."""
        try:
            import wget  # noqa: F401 — used by hubconf.py implicitly
        except ImportError:
            pass  # not strictly required for hub load with pretrained=True

    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:
        if self.resize_cap <= 0:
            return frame
        h, w = frame.shape[:2]
        longer = max(h, w)
        if longer <= self.resize_cap:
            return frame
        scale = self.resize_cap / longer
        return cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Resize to model input, convert BGR→RGB, normalise, to tensor."""
        resized = cv2.resize(
            frame, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normed = (rgb - _MEAN) / _STD
        tensor = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self._device)

    def _decode_seg(
        self,
        seg: torch.Tensor,
        ori_h: int,
        ori_w: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert raw segmentation logits to binary road/lane masks at
        original resolution.

        seg shape: (1, num_classes, H', W') — typically (1, 3, 12, 20) or
        similar depending on stride; we upsample to the model input size
        first, then rescale to original frame size.
        """
        # Upsample to model input resolution, then softmax over class dim
        seg_up = F.interpolate(
            seg, size=(_INPUT_H, _INPUT_W), mode="bilinear", align_corners=False
        )
        probs = torch.softmax(seg_up, dim=1)[0]  # (C, H, W)

        road_prob = probs[_SEG_ROAD].cpu().numpy()
        lane_prob = probs[_SEG_LANE].cpu().numpy()

        road_mask = (road_prob > self.conf_road).astype(np.uint8)
        lane_mask = (lane_prob > self.conf_lane).astype(np.uint8)

        # Resize binary masks to original frame resolution
        road_mask = cv2.resize(
            road_mask, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST
        )
        lane_mask = cv2.resize(
            lane_mask, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST
        )

        return road_mask, lane_mask

    def _visualise(
        self,
        frame: np.ndarray,
        road_mask: np.ndarray,
        lane_mask: np.ndarray,
    ) -> np.ndarray:
        out = frame.copy()

        # ── Drivable area overlay ───────────────────────────────────────────
        if road_mask.any():
            colour_layer = np.zeros_like(out)
            colour_layer[road_mask == 1] = _COL_ROAD
            out = cv2.addWeighted(out, 1.0, colour_layer, self.road_alpha, 0)

        # ── Lane line overlay ───────────────────────────────────────────────
        if lane_mask.any():
            # Slightly thicken thin lane pixels with a small dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            lane_dilated = cv2.dilate(lane_mask, kernel, iterations=1)
            colour_layer = np.zeros_like(out)
            colour_layer[lane_dilated == 1] = _COL_LANE
            out = cv2.addWeighted(out, 1.0, colour_layer, self.lane_alpha, 0)

        return out
