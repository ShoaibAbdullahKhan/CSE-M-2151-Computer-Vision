"""Shared pieces used by both systems: part ontology, colour naming, SAM 2 wrapper, metrics, drawing."""
import os, time, math, re
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"

# --------------------------------------------------------------------------------------------- ontology
SPECIES = ["bird", "cat", "cow", "dog", "horse", "sheep"]

# 15 merged part classes used everywhere (index = YOLO class id of the part model)
PARTS = ["head", "ear", "eye", "nose", "muzzle", "neck", "torso", "leg", "paw", "tail",
         "beak", "wing", "foot", "hoof", "horn"]

# which parts each species has in PASCAL-Part
SPECIES_PARTS = {
    "cat":   ["head", "ear", "eye", "nose", "neck", "torso", "leg", "paw", "tail"],
    "dog":   ["head", "ear", "eye", "nose", "muzzle", "neck", "torso", "leg", "paw", "tail"],
    "bird":  ["head", "eye", "beak", "neck", "torso", "wing", "leg", "foot", "tail"],
    "horse": ["head", "ear", "eye", "muzzle", "neck", "torso", "leg", "hoof", "tail"],
    "cow":   ["head", "ear", "eye", "muzzle", "neck", "torso", "leg", "horn", "tail"],
    "sheep": ["head", "ear", "eye", "muzzle", "neck", "torso", "leg", "horn", "tail"],
}

# plural / display names
PLURAL = {"head": "head", "ear": "ears", "eye": "eyes", "nose": "nose", "muzzle": "muzzle",
          "neck": "neck", "torso": "torso", "leg": "legs", "paw": "paws", "tail": "tail",
          "beak": "beak", "wing": "wings", "foot": "feet", "hoof": "hooves", "horn": "horns"}


def load_image(path_or_img):
    """RGB PIL image with EXIF orientation applied (phone photos are often stored rotated)."""
    im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    return ImageOps.exif_transpose(im).convert("RGB")


def limit_size(im, max_side=1600):
    """Large phone photos (4000 px) are shrunk once; every model resizes internally anyway."""
    s = max(im.size)
    if s <= max_side:
        return im, 1.0
    f = max_side / s
    return im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS), f


# --------------------------------------------------------------------------------------------- colour naming
# reference colours for animal fur / feathers, named in plain words
NAMED_COLOURS = {
    "black": (25, 25, 25), "dark grey": (80, 80, 80), "grey": (135, 135, 135), "light grey": (190, 190, 190),
    "white": (240, 240, 240), "cream": (235, 220, 185), "beige": (205, 180, 140), "tan": (195, 150, 100),
    "light brown": (165, 115, 70), "brown": (115, 75, 45), "dark brown": (65, 40, 25),
    "ginger": (210, 120, 50), "golden": (215, 170, 80), "reddish brown": (150, 65, 40),
    "pink": (225, 150, 150), "yellow": (230, 200, 60), "orange": (235, 140, 30), "red": (190, 40, 40),
    "green": (70, 130, 60), "blue": (60, 100, 180),
}
_NC_NAMES = list(NAMED_COLOURS)
_NC_LAB = cv2.cvtColor(np.uint8([[NAMED_COLOURS[n] for n in _NC_NAMES]]), cv2.COLOR_RGB2LAB)[0].astype(np.float32)


def colour_names(img_rgb, mask, k=3, min_share=0.2):
    """Dominant colour names of the masked pixels (k-means in Lab space). Returns [(name, share), ...]."""
    px = img_rgb[mask.astype(bool)]
    if len(px) < 20:
        return []
    if len(px) > 6000:
        px = px[np.random.default_rng(0).choice(len(px), 6000, replace=False)]
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    k = min(k, len(lab))
    _, lbl, cen = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3,
                             cv2.KMEANS_PP_CENTERS)
    out = {}
    for c in range(k):
        share = float((lbl == c).mean())
        name = _NC_NAMES[int(np.argmin(((_NC_LAB - cen[c]) ** 2).sum(1)))]
        out[name] = out.get(name, 0) + share
    return sorted([(n, s) for n, s in out.items() if s >= min_share], key=lambda t: -t[1])


# --------------------------------------------------------------------------------------------- SAM 2 wrapper
class SAM2Segmenter:
    """Box-prompted SAM 2.1 (ultralytics). Runs on a context crop around the boxes so that small parts
    in large photos are seen at a useful resolution. Shared by both systems."""

    def __init__(self, weights="sam2.1_b.pt", device="cuda"):
        from ultralytics import SAM
        self.model = SAM(str(WEIGHTS / weights) if (WEIGHTS / weights).exists() else weights)
        self.device = device

    def __call__(self, img_rgb, boxes, context=0.6, min_side=256):
        H, W = img_rgb.shape[:2]
        full = np.zeros((H, W), bool)
        if len(boxes) == 0:
            return full, []
        b = np.asarray(boxes, float)
        x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        cw, ch = x2 - x1, y2 - y1
        pad = max(context * max(cw, ch), (min_side - max(cw, ch)) / 2, 8)
        cx1, cy1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        cx2, cy2 = int(min(W, x2 + pad)), int(min(H, y2 + pad))
        crop = np.ascontiguousarray(img_rgb[cy1:cy2, cx1:cx2, ::-1])          # ultralytics wants BGR
        bb = (b - [cx1, cy1, cx1, cy1]).tolist()
        res = self.model(crop, bboxes=bb, verbose=False, device=self.device)[0]
        inst = []
        if res.masks is not None:
            for m in res.masks.data.cpu().numpy().astype(bool):
                if m.shape != crop.shape[:2]:
                    m = cv2.resize(m.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                g = np.zeros((H, W), bool)
                g[cy1:cy2, cx1:cx2] = m
                inst.append(g)
                full |= g
        return full, inst


# --------------------------------------------------------------------------------------------- metrics
def iou_dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    s = pred.sum() + gt.sum()
    iou = inter / union if union else (1.0 if s == 0 else 0.0)
    dice = 2 * inter / s if s else 1.0
    return float(iou), float(dice)


def box_iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms_boxes(boxes, scores, thr=0.5):
    order = np.argsort(scores)[::-1]
    keep = []
    for i in order:
        if all(box_iou(boxes[i], boxes[j]) < thr for j in keep):
            keep.append(i)
    return keep


# --------------------------------------------------------------------------------------------- drawing
def overlay(img_rgb, mask, colour=(255, 40, 120), alpha=0.5, boxes=()):
    out = img_rgb.copy().astype(np.float32)
    m = mask.astype(bool)
    out[m] = out[m] * (1 - alpha) + np.array(colour) * alpha
    out = out.astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lw = max(2, round(max(img_rgb.shape[:2]) / 400))
    cv2.drawContours(out, cnts, -1, colour, lw)
    for b in boxes:
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 215, 0), lw)
    return out


def isolate(img_rgb, mask, bg=255, pad=0.08):
    """The isolated body part: everything outside the mask blanked, cropped to the part."""
    m = mask.astype(bool)
    if not m.any():
        return np.full((64, 64, 3), bg, np.uint8)
    out = np.full_like(img_rgb, bg)
    out[m] = img_rgb[m]
    ys, xs = np.where(m)
    H, W = m.shape
    p = int(pad * max(np.ptp(ys) + 1, np.ptp(xs) + 1))
    return out[max(0, ys.min() - p):min(H, ys.max() + p + 1), max(0, xs.min() - p):min(W, xs.max() + p + 1)]


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _T:
            def __enter__(s):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                s.t0 = time.perf_counter()

            def __exit__(s, *a):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timer.t[name] = timer.t.get(name, 0) + time.perf_counter() - s.t0
        return _T()
