#!/usr/bin/env python
"""
Retrain YOLOv8n on the live-robust dataset built by build_live_dataset.py.

Identical recipe to the notebook's model except for the data, so any difference is
attributable to the dataset fix (small-scale composites, white-balance jitter and
background negatives) rather than to tuning.
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "yolo_live" / "data.yaml"
RUNS = ROOT / "outputs" / "runs"
NAME = "finetune_yolov8n_live"
SEED = 42
NWORKERS = 0 if os.name == "nt" else 4


def patch_albumentations():
    """Same occlusion / noise / dirt pipeline the notebook injects."""
    import albumentations as A
    import ultralytics.data.augment as ua
    pipeline = A.Compose([
        A.CoarseDropout(num_holes_range=(1, 6),
                        hole_height_range=(0.08, 0.28), hole_width_range=(0.08, 0.28),
                        fill="random_uniform", p=0.45),
        A.GaussNoise(std_range=(0.03, 0.14), p=0.30),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.20),
        A.OneOf([A.MotionBlur(blur_limit=(3, 9)), A.Defocus(radius=(2, 6))], p=0.22),
        A.Spatter(mean=(0.6, 0.72), std=(0.25, 0.35),
                  cutout_threshold=(0.62, 0.70), mode="mud", p=0.25),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), p=0.25),
        A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.3),
                                   contrast_limit=(-0.3, 0.35), p=0.50),
        A.RandomGamma(gamma_limit=(65, 145), p=0.30),
        A.CLAHE(clip_limit=(1, 4), p=0.20),
        # white-balance style colour variation, on top of what the composites already carry
        A.RGBShift(r_shift_limit=18, g_shift_limit=10, b_shift_limit=18, p=0.35),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=28,
                             val_shift_limit=22, p=0.30),
        A.ToGray(p=0.05),
        A.ImageCompression(quality_range=(35, 90), p=0.30),
    ])

    def _init(self, p=1.0):
        self.p = p
        self.contains_spatial = False      # pixel-level only; boxes stay at full extent
        self.transform = pipeline
    ua.Albumentations.__init__ = _init
    print("custom albumentations pipeline installed")


def main():
    from ultralytics import YOLO
    patch_albumentations()
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    dev = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA), imgsz=640, epochs=100, batch=16,
        optimizer="auto", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3.0, cos_lr=True, patience=30,
        seed=SEED, deterministic=True, device=dev, workers=NWORKERS,
        pretrained=True, val=True, plots=True, save=True, exist_ok=True,
        project=str(RUNS), name=NAME,
        # geometry: scale is widened downward so small notes stay in distribution
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=25.0, translate=0.25, scale=0.9, shear=6.0, perspective=0.0008,
        flipud=0.3, fliplr=0.5, mosaic=1.0, close_mosaic=15, mixup=0.1, copy_paste=0.0,
    )

    best = RUNS / NAME / "weights" / "best.pt"
    out = ROOT / "weights" / "taka_yolov8n_live.pt"
    shutil.copy(best, out)
    print(f"\ndeployed -> {out}")


if __name__ == "__main__":
    main()
