"""
prisme/tasks/pose_estimation.py

Top-down human pose estimation using ViTPose-B (COCO 17 keypoints).

Pipeline per frame:
  1. YOLOv8n (ultralytics, person class only) → bounding boxes
  2. Crop + pad each person to 192×256, normalise
  3. ViTPose-B (HuggingFace transformers) → heatmaps (17, 64, 48)
  4. Soft-argmax decode → keypoint coordinates in crop space
  5. Rescale coordinates back to original frame, draw COCO skeleton

The person detector is fully owned by this task — no coupling to the
object_detection task. This keeps tasks independently loadable and
benchmarkable.

Dependencies (pip):
    ultralytics       # YOLOv8n weights auto-download on first use (~6 MB)
    transformers      # ViTPose-B weights auto-download on first use (~330 MB)
    huggingface_hub   # pulled in by transformers

Input:  BGR uint8, any resolution
Output: BGR uint8 visualisation at original resolution
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from prisme.base import BaseTask
from ultralytics import YOLO
from transformers import AutoProcessor, VitPoseForPoseEstimation

# ── HuggingFace model identifier ───────────────────────────────────────────
_VITPOSE_MODEL_ID = "usyd-community/vitpose-base-simple"

# ── ViTPose input dimensions (COCO training config) ────────────────────────
_CROP_W = 192
_CROP_H = 256
_HEATMAP_W = 48
_HEATMAP_H = 64

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── COCO 17-keypoint skeleton ───────────────────────────────────────────────
# Each pair is (kp_index_a, kp_index_b)
_COCO_SKELETON = [
    (0, 1),
    (0, 2),  # nose → eyes
    (1, 3),
    (2, 4),  # eyes → ears
    (5, 6),  # shoulders
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (5, 11),
    (6, 12),  # shoulders → hips
    (11, 12),  # hips
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
]

# Colour per limb group (BGR): upper body warm, lower body cool
_LIMB_COLOURS = {
    (0, 1): (0, 215, 255),
    (0, 2): (0, 215, 255),
    (1, 3): (0, 165, 255),
    (2, 4): (0, 165, 255),
    (5, 6): (0, 255, 0),
    (5, 7): (255, 128, 0),
    (7, 9): (255, 200, 0),
    (6, 8): (0, 128, 255),
    (8, 10): (0, 200, 255),
    (5, 11): (200, 0, 200),
    (6, 12): (200, 0, 200),
    (11, 12): (180, 0, 255),
    (11, 13): (255, 0, 100),
    (13, 15): (255, 0, 180),
    (12, 14): (0, 100, 255),
    (14, 16): (0, 180, 255),
}
_KP_COLOUR = (255, 255, 255)


class PoseEstimationTask(BaseTask):
    """
    Top-down 2D human pose estimation using ViTPose-B + YOLOv8n detector.

    Args:
        det_conf:        YOLOv8n person detection confidence threshold.
        det_iou:         YOLOv8n NMS IoU threshold.
        kp_conf:         Minimum heatmap confidence to draw a keypoint.
        kp_radius:       Keypoint dot radius in pixels.
        limb_thickness:  Skeleton limb line thickness in pixels.
        bbox_pad:        Fractional padding added to each detected bbox
                         before cropping (helps include full body context).

    """

    def __init__(
        self,
        det_conf: float = 0.3,
        det_iou: float = 0.45,
        kp_conf: float = 0.3,
        kp_radius: int = 4,
        limb_thickness: int = 2,
        bbox_pad: float = 0.1,
    ) -> None:
        super().__init__(name="pose_estimation")
        self.det_conf = det_conf
        self.det_iou = det_iou
        self.kp_conf = kp_conf
        self.kp_radius = kp_radius
        self.limb_thickness = limb_thickness
        self.bbox_pad = bbox_pad

        self._detector = None
        self._pose = None
        self._processor = None
        self._device = None

    # ── BaseTask interface ──────────────────────────────────────────────────

    def _download_weights_if_missing(self) -> None:
        """HuggingFace auto-downloads ViTPose weights on first use — nothing to do here."""
        pass

    def _load(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Person detector — YOLOv8n, auto-downloads weights on first use
        self._detector = YOLO("yolov8n.pt")

        # ViTPose-B — weights auto-download from HuggingFace Hub (~330 MB)
        self._processor = AutoProcessor.from_pretrained(_VITPOSE_MODEL_ID)
        self._pose = VitPoseForPoseEstimation.from_pretrained(_VITPOSE_MODEL_ID)
        self._pose.to(self._device).eval()

    def infer(self, frame: np.ndarray) -> np.ndarray:
        bboxes = self._detect_persons(frame)
        if not bboxes:
            return frame.copy()

        keypoints_list = self._estimate_poses(frame, bboxes)
        return self._visualise(frame, bboxes, keypoints_list)

    # ── Detection ───────────────────────────────────────────────────────────

    def _detect_persons(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Run YOLOv8n on the frame, return list of person bboxes as
        (x1, y1, x2, y2) in pixel coords, padded by bbox_pad.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.predict(
            rgb,
            classes=[0],  # COCO class 0 = person
            conf=self.det_conf,
            iou=self.det_iou,
            verbose=False,
        )

        h, w = frame.shape[:2]
        bboxes = []
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box[:4]
                # Add padding
                bw = x2 - x1
                bh = y2 - y1
                x1 = max(0, int(x1 - bw * self.bbox_pad))
                y1 = max(0, int(y1 - bh * self.bbox_pad))
                x2 = min(w, int(x2 + bw * self.bbox_pad))
                y2 = min(h, int(y2 + bh * self.bbox_pad))
                if x2 > x1 and y2 > y1:
                    bboxes.append((x1, y1, x2, y2))
        return bboxes

    # ── Pose estimation ─────────────────────────────────────────────────────

    def _estimate_poses(
        self,
        frame: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
    ) -> list[np.ndarray]:
        """
        For each bbox, crop → preprocess → run ViTPose → decode heatmap.
        Returns list of (17, 3) arrays: [x, y, confidence] in frame space.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints_list = []

        for x1, y1, x2, y2 in bboxes:
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            inp = self._preprocess_crop(crop)
            heatmaps = self._run_vitpose(inp)  # (17, 64, 48)
            kps = self._decode_heatmaps(heatmaps, x1, y1, x2 - x1, y2 - y1)
            keypoints_list.append(kps)

        return keypoints_list

    def _preprocess_crop(self, crop_rgb: np.ndarray) -> torch.Tensor:
        """Resize crop to ViTPose input size, normalise, return tensor."""
        resized = cv2.resize(
            crop_rgb, (_CROP_W, _CROP_H), interpolation=cv2.INTER_LINEAR
        )
        norm = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self._device)

    def _run_vitpose(self, inp: torch.Tensor) -> np.ndarray:
        """Run ViTPose forward pass, return heatmaps as numpy (17, H, W)."""
        with torch.no_grad():
            out = self._pose(pixel_values=inp)
        # out.heatmaps: (1, 17, 64, 48)
        return out.heatmaps[0].cpu().numpy()

    def _decode_heatmaps(
        self,
        heatmaps: np.ndarray,
        x1: int,
        y1: int,
        crop_w: int,
        crop_h: int,
    ) -> np.ndarray:
        """
        Soft-argmax decode each heatmap channel to a (x, y) coordinate,
        then rescale from crop space back to original frame space.

        Returns (17, 3) float array: columns are [x_frame, y_frame, conf].
        """
        num_kp, hm_h, hm_w = heatmaps.shape
        result = np.zeros((num_kp, 3), dtype=np.float32)

        for k in range(num_kp):
            hm = heatmaps[k]
            conf = float(hm.max())
            # Flatten → argmax → 2D index
            idx = np.argmax(hm)
            hy = idx // hm_w
            hx = idx % hm_w
            # Rescale to crop pixel space
            cx = (hx + 0.5) / hm_w * crop_w
            cy = (hy + 0.5) / hm_h * crop_h
            # Rescale to frame space
            result[k] = [x1 + cx, y1 + cy, conf]

        return result

    # ── Visualisation ───────────────────────────────────────────────────────

    def _visualise(
        self,
        frame: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
        keypoints_list: list[np.ndarray],
    ) -> np.ndarray:
        out = frame.copy()

        for (x1, y1, x2, y2), kps in zip(bboxes, keypoints_list):
            # Faint bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (80, 80, 80), 1)

            # Skeleton limbs
            for ka, kb in _COCO_SKELETON:
                xa, ya, ca = kps[ka]
                xb, yb, cb = kps[kb]
                if ca < self.kp_conf or cb < self.kp_conf:
                    continue
                colour = _LIMB_COLOURS.get((ka, kb), (200, 200, 200))
                cv2.line(
                    out,
                    (int(xa), int(ya)),
                    (int(xb), int(yb)),
                    colour,
                    self.limb_thickness,
                    cv2.LINE_AA,
                )

            # Keypoint dots
            for x, y, conf in kps:
                if conf < self.kp_conf:
                    continue
                cv2.circle(
                    out, (int(x), int(y)), self.kp_radius, _KP_COLOUR, -1, cv2.LINE_AA
                )
                cv2.circle(
                    out, (int(x), int(y)), self.kp_radius, (0, 0, 0), 1, cv2.LINE_AA
                )

        return out
