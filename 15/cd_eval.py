"""Metrics, speed/compute benchmark, corruptions, shortcut tests and occlusion-sensitivity maps (same for all models)."""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
import cd_data as D


def metrics(y, p, thr=0.5):
    """Dirty = positive class."""
    y, yhat = np.asarray(y), (np.asarray(p) >= thr).astype(int)
    tp, tn = int(((yhat == 1) & (y == 1)).sum()), int(((yhat == 0) & (y == 0)).sum())
    fp, fn = int(((yhat == 1) & (y == 0)).sum()), int(((yhat == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, p) if len(set(y)) == 2 else np.nan
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0, "AUC": auc, "TP": tp, "TN": tn, "FP": fp, "FN": fn}


# --------------------------------------------------------------------------------------------- compute
@torch.no_grad()
def benchmark(model, size=224, warmup=10, runs=50, batch=32):
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    x1 = torch.randn(1, 3, size, size, device="cuda")
    with FlopCounterMode(display=False) as fc:
        model(x1)
    gflops = fc.get_total_flops() / 2e9                         # multiply-accumulates count as one FLOP-pair -> GMACs
    for _ in range(warmup):
        model(x1)
    torch.cuda.synchronize()
    lat = []
    for _ in range(runs):
        t = time.perf_counter(); model(x1); torch.cuda.synchronize(); lat.append(time.perf_counter() - t)
    xb = torch.randn(batch, 3, size, size, device="cuda")
    for _ in range(3):
        model(xb)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for _ in range(10):
        model(xb)
    torch.cuda.synchronize()
    thr = 10 * batch / (time.perf_counter() - t)
    lat = np.array(lat) * 1000
    return {"GMACs": gflops, "latency_ms": float(lat.mean()), "latency_p95_ms": float(np.percentile(lat, 95)),
            "FPS_bs1": float(1000 / lat.mean()), f"FPS_bs{batch}": float(thr),
            "peak_inference_mem_GB": torch.cuda.max_memory_allocated() / 2**30}


# --------------------------------------------------------------------------------------------- corruptions & shortcuts
def corrupt(kind, sev, seed=0):
    rng = np.random.default_rng(seed)

    def f(im):
        a = np.asarray(im, np.float32)
        if kind == "dark":
            g = [1.6, 2.2, 3.0][sev - 1]
            return Image.fromarray((255 * (a / 255) ** g * [0.8, 0.6, 0.4][sev - 1]).clip(0, 255).astype(np.uint8))
        if kind == "overexposed":
            return Image.fromarray((a * [1.5, 1.9, 2.4][sev - 1] + 20).clip(0, 255).astype(np.uint8))
        if kind == "blur":
            return im.filter(ImageFilter.GaussianBlur([2, 4, 7][sev - 1]))
        if kind == "noise":
            return Image.fromarray((a + rng.normal(0, [12, 25, 40][sev - 1], a.shape)).clip(0, 255).astype(np.uint8))
        if kind == "haze":
            t = [0.35, 0.55, 0.7][sev - 1]
            return Image.fromarray((a * (1 - t) + 215 * t).clip(0, 255).astype(np.uint8))
        if kind == "occlusion":
            out = a.copy(); h, w = a.shape[:2]
            n, s = [3, 5, 8][sev - 1], [0.18, 0.22, 0.26][sev - 1]
            for _ in range(n):
                bw, bh = int(s * w), int(s * h)
                x0, y0 = rng.integers(0, w - bw), rng.integers(0, h - bh)
                out[y0:y0 + bh, x0:x0 + bw] = (128, 128, 128)
            return Image.fromarray(out.astype(np.uint8))
        raise ValueError(kind)
    return f


CORRUPTIONS = ["dark", "overexposed", "blur", "noise", "haze", "occlusion"]


def view_transform(kind):
    """Shortcut tests that change *what part of the scene* is shown.
    bottom: lower half only (removes the sky);  centre: central 50% zoom (close-up of the scene centre);
    top: upper half only (mostly sky / background for wide shots)."""
    def f(im):
        w, h = im.size
        if kind == "bottom":
            return im.crop((0, h // 2, w, h))
        if kind == "top":
            return im.crop((0, 0, w, h // 2))
        if kind == "centre":
            return im.crop((w // 4, h // 4, w - w // 4, h - h // 4))
        raise ValueError(kind)
    return f


def apply_to_cache(cache, files, fn):
    return {f: fn(cache[f]) for f in files}


# --------------------------------------------------------------------------------------------- explanation
@torch.no_grad()
def occlusion_map(model, im, size=224, patch=32, stride=16):
    """Model-agnostic saliency: drop in P(predicted class) when a grey patch hides each region (works for CNN,
    Transformer and Mamba alike)."""
    tf = D.make_transforms(size, train=False)
    x = tf(im)[None].cuda()
    with torch.autocast("cuda", torch.bfloat16):
        p0 = model(x).float().softmax(-1)[0]
    c = int(p0.argmax())
    xs, pos = [], []
    for yy in range(0, size - patch + 1, stride):
        for xx in range(0, size - patch + 1, stride):
            z = x.clone(); z[:, :, yy:yy + patch, xx:xx + patch] = 0.0
            xs.append(z); pos.append((yy, xx))
    drops = []
    for i in range(0, len(xs), 64):
        with torch.autocast("cuda", torch.bfloat16):
            drops.append(p0[c] - model(torch.cat(xs[i:i + 64])).float().softmax(-1)[:, c])
    drops = torch.cat(drops).cpu().numpy()
    heat, cnt = np.zeros((size, size)), np.zeros((size, size))
    for d, (yy, xx) in zip(drops, pos):
        heat[yy:yy + patch, xx:xx + patch] += d; cnt[yy:yy + patch, xx:xx + patch] += 1
    return heat / np.maximum(cnt, 1), c, float(p0[c])


def display_crop(im, size=224):
    """The exact centre crop the models see (for overlaying saliency maps)."""
    s = int(size / 0.875) / min(im.size)
    r = im.resize((round(im.width * s), round(im.height * s)))
    l, t = (r.width - size) // 2, (r.height - size) // 2
    return r.crop((l, t, l + size, t + size))
