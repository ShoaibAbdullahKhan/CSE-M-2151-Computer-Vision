#!/usr/bin/env python
"""
Build a second training set that fixes the three measured live-camera failures.

Diagnosis (outputs/results/live_diagnosis.csv and false_positives.csv):
  1. SCALE      - 0.0% of training notes are below 10% of the frame, and detection
                  collapses from 100% to 43.8% at 7% and to 0% at 1.5%. A note held
                  up to a webcam is usually under 10%.
  2. WHITE BAL. - 500 is green/teal and 1000 is purple, so the model leans on hue.
                  A warm +30% cast drops denomination accuracy to 50% (a coin flip).
  3. FALSE POS. - every one of the 290 images contains exactly one note and there are
                  ZERO background images, so 73.3% of note-free photographs still
                  produce a detection.

Fixes applied here:
  1. Composite masked notes onto varied backgrounds at 1.5-25% of the frame.
  2. Jitter per-channel gains (a white-balance model, not a hue rotation) so colour is
     learned relative to the scene.
  3. Add label-free background images so "no note" becomes a learnable state.

Notes are taken only from the split they belong to, and background images are disjoint
between train and val, so no leakage is introduced.
"""
from __future__ import annotations

import glob
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "yolo"
DST = ROOT / "data" / "yolo_live"
CACHE = ROOT / "data" / "mask_cache"
PETS = ROOT.parent / "10" / "data" / "images"
DIV2K = ROOT.parent / "02" / "data" / "DIV2K_train_HR"

N_COMPOSITE = {"train": 700, "val": 150}
N_BACKGROUND = {"train": 90, "val": 20}
CANVAS = (720, 1280)                     # webcam-shaped
SCALE_RANGE = (0.015, 0.25)              # the range the model currently fails in
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


# ----------------------------------------------------------------- note masking
def note_mask(rgb):
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = max(6, int(0.02 * min(h, w)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    d = np.sqrt(((lab - np.median(border, 0)) ** 2 * np.array([1., 2.5, 2.5])).sum(-1))
    dn = cv2.normalize(cv2.GaussianBlur(d, (0, 0), 3), None, 0, 255,
                       cv2.NORM_MINMAX).astype(np.uint8)
    t, _ = cv2.threshold(dn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strong, weak = (dn >= t).astype(np.uint8), (dn >= 0.45 * t).astype(np.uint8)
    n, lbl = cv2.connectedComponents(weak, 8)[:2]
    keep = np.unique(lbl[strong > 0]); keep = keep[keep > 0]
    m = cv2.morphologyEx(np.isin(lbl, keep).astype(np.uint8) * 255,
                         cv2.MORPH_CLOSE, KERNEL, iterations=2)
    n, lb, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    comp = (lb == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(comp)
    cv2.fillConvexPoly(out, cv2.convexHull(np.vstack(cnts)), 1)
    return out


def load_notes(split):
    """Cropped note + alpha mask + class id, for every image of a split."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{split}.npz"
    files = sorted(glob.glob(str(SRC / "images" / split / "*.jpg")))
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["notes"])
    notes = []
    for f in tqdm(files, desc=f"masking {split}", ncols=78):
        rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        m = note_mask(rgb)
        if m is None:
            continue
        ys, xs = np.where(m > 0)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        notes.append(dict(img=rgb[y0:y1+1, x0:x1+1].copy(),
                          mask=m[y0:y1+1, x0:x1+1].copy(),
                          cid=0 if Path(f).name.startswith("0_") else 1))
    np.savez_compressed(cache, notes=np.array(notes, dtype=object))
    return notes


# ----------------------------------------------------------------- backgrounds
def background_pool(seed=0):
    rng = random.Random(seed)
    pool = sorted(glob.glob(str(PETS / "*.jpg")))
    rng.shuffle(pool)
    extra = sorted(glob.glob(str(DIV2K / "*.png")))
    rng.shuffle(extra)
    return pool, extra


def synth_bg(rng, H, W):
    kind = rng.choice(["sheet", "wood", "cloth", "dark", "desk"])
    if kind == "sheet":
        bg = np.full((H, W, 3), rng.randint(215, 248), np.uint8)
    elif kind == "dark":
        bg = np.full((H, W, 3), rng.randint(28, 70), np.uint8)
    elif kind == "wood":
        bg = np.zeros((H, W, 3), np.uint8); bg[:] = (rng.randint(120, 170),
                                                     rng.randint(90, 125), rng.randint(55, 90))
        for _ in range(240):
            y = rng.randint(0, H - 1)
            cv2.line(bg, (0, y), (W, y),
                     tuple(int(max(0, min(255, c + rng.gauss(0, 15)))) for c in bg[y, 0]),
                     rng.randint(1, 3))
    elif kind == "cloth":
        small = np.random.default_rng(rng.randint(0, 10**6)).integers(
            70, 200, (H // 22, W // 22, 3)).astype(np.uint8)
        bg = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        bg = np.full((H, W, 3), rng.randint(150, 200), np.uint8)
    return (bg + np.random.default_rng(rng.randint(0, 10**6))
            .normal(0, 4, (H, W, 3))).clip(0, 255).astype(np.uint8)


def fit_bg(path, H, W):
    im = cv2.imread(str(path))
    if im is None:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    s = max(H / im.shape[0], W / im.shape[1])
    im = cv2.resize(im, (max(W, int(im.shape[1] * s)), max(H, int(im.shape[0] * s))))
    y = (im.shape[0] - H) // 2; x = (im.shape[1] - W) // 2
    return im[y:y+H, x:x+W]


# ----------------------------------------------------------------- compositing
def wb_jitter(img, rng):
    """Per-channel gains: a white-balance model, NOT a hue rotation."""
    g = np.array([rng.uniform(0.78, 1.28), rng.uniform(0.92, 1.08),
                  rng.uniform(0.78, 1.28)], np.float32)
    return (img.astype(np.float32) * g).clip(0, 255).astype(np.uint8)


def place(note, bg, rng):
    H, W = bg.shape[:2]
    crop, cm = note["img"], note["mask"]
    rel = rng.uniform(*SCALE_RANGE)
    ar = crop.shape[1] / crop.shape[0]
    tw = int(np.sqrt(rel * H * W * ar))
    tw = max(14, min(tw, W - 8))
    s = tw / crop.shape[1]
    crop = cv2.resize(crop, (max(3, int(crop.shape[1]*s)), max(3, int(crop.shape[0]*s))))
    cm = cv2.resize(cm, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)

    pad = int(max(crop.shape[:2]) * 0.6) + 2
    crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    cm = cv2.copyMakeBorder(cm, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), rng.uniform(-180, 180), 1.0)
    crop = cv2.warpAffine(crop, M, (w, h)); cm = cv2.warpAffine(cm, M, (w, h))
    d = rng.uniform(0, 0.28) * w
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[d, 0], [w - d, 0], [w, h], [0, h]])
    P = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(crop, P, (w, h)); cm = cv2.warpPerspective(cm, P, (w, h))

    ys, xs = np.where(cm > 0)
    if len(ys) < 25:
        return None
    crop = crop[ys.min():ys.max()+1, xs.min():xs.max()+1]
    cm = cm[ys.min():ys.max()+1, xs.min():xs.max()+1]
    ch, cw = crop.shape[:2]
    if ch >= H - 2 or cw >= W - 2:
        return None
    oy = rng.randint(0, H - ch - 1); ox = rng.randint(0, W - cw - 1)
    out = bg.copy()
    a = (cm > 0)[..., None].astype(np.float32)
    if rng.random() < 0.5:                                  # soften the cut edge
        a = cv2.GaussianBlur(a[..., 0], (0, 0), 0.8)[..., None]
    out[oy:oy+ch, ox:ox+cw] = (crop * a + out[oy:oy+ch, ox:ox+cw] * (1 - a)).astype(np.uint8)
    return out, (ox, oy, ox + cw, oy + ch), note["cid"]


def main():
    rng = random.Random(1234)
    if DST.exists():
        shutil.rmtree(DST)
    for s in ["train", "val", "test"]:
        (DST / "images" / s).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / s).mkdir(parents=True, exist_ok=True)

    # 1. copy the originals through unchanged
    counts = {}
    for s in ["train", "val", "test"]:
        n = 0
        for f in sorted(glob.glob(str(SRC / "images" / s / "*.jpg"))):
            p = Path(f)
            shutil.copy(p, DST / "images" / s / p.name)
            shutil.copy(SRC / "labels" / s / f"{p.stem}.txt", DST / "labels" / s / f"{p.stem}.txt")
            n += 1
        counts[s] = {"original": n, "composite": 0, "background": 0}

    pets, div = background_pool()
    cursor = {"train": 0, "val": len(pets) // 2}             # disjoint halves
    H, W = CANVAS

    # 2. small-scale composites
    for split in ["train", "val"]:
        notes = load_notes(split)
        if not notes:
            raise SystemExit(f"no masked notes for {split}")
        for i in tqdm(range(N_COMPOSITE[split]), desc=f"composite {split}", ncols=78):
            note = notes[rng.randrange(len(notes))]
            r = rng.random()
            if r < 0.45:
                bg = fit_bg(pets[cursor[split] % len(pets)], H, W); cursor[split] += 1
            elif r < 0.60 and div:
                bg = fit_bg(div[rng.randrange(len(div))], H, W)
            else:
                bg = synth_bg(rng, H, W)
            if bg is None:
                continue
            got = place(note, bg, rng)
            if got is None:
                continue
            img, (x1, y1, x2, y2), cid = got
            if rng.random() < 0.7:
                img = wb_jitter(img, rng)
            stem = f"syn_{split}_{i:04d}"
            cv2.imwrite(str(DST / "images" / split / f"{stem}.jpg"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            xc, yc = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            (DST / "labels" / split / f"{stem}.txt").write_text(
                f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            counts[split]["composite"] += 1

    # 3. background images with EMPTY labels -> "no note" becomes learnable
    for split in ["train", "val"]:
        for i in tqdm(range(N_BACKGROUND[split]), desc=f"background {split}", ncols=78):
            r = rng.random()
            if r < 0.75:
                bg = fit_bg(pets[cursor[split] % len(pets)], H, W); cursor[split] += 1
            elif div:
                bg = fit_bg(div[rng.randrange(len(div))], H, W)
            else:
                bg = synth_bg(rng, H, W)
            if bg is None:
                continue
            if rng.random() < 0.5:
                bg = wb_jitter(bg, rng)
            stem = f"bg_{split}_{i:04d}"
            cv2.imwrite(str(DST / "images" / split / f"{stem}.jpg"),
                        cv2.cvtColor(bg, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            (DST / "labels" / split / f"{stem}.txt").write_text("")   # empty = background
            counts[split]["background"] += 1

    (DST / "data.yaml").write_text(yaml.safe_dump({
        "path": str(DST.resolve()), "train": "images/train", "val": "images/val",
        "test": "images/test", "nc": 2, "names": {0: "500 taka", 1: "1000 taka"}},
        sort_keys=False))
    (ROOT / "outputs" / "results" / "live_dataset_counts.json").write_text(
        json.dumps(counts, indent=2))
    print("\n" + json.dumps(counts, indent=2))
    for s in ["train", "val", "test"]:
        print(f"  {s:5s}: {len(list((DST/'images'/s).glob('*.jpg')))} images")
    print(f"\nwrote {DST}")


if __name__ == "__main__":
    main()
