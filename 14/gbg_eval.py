"""Prediction, COCO mAP, precision/recall, timing and synthetic-corruption helpers (identical for every model)."""
import time, json, io, contextlib
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from gbg_data import CACHE, INPUT, letterbox


def load_cached(files):
    return {f: np.asarray(Image.open(CACHE / f).convert("RGB")) for f in files}


@torch.no_grad()
def predict_images(model, images, size=INPUT, bs=8, k=100, transform=None):
    """images: {file: HxWx3 uint8}. Returns {file: (boxes Nx4 normalised xyxy, scores N, labels or None)}."""
    model.eval()
    out, files = {}, list(images)
    for i in range(0, len(files), bs):
        chunk = files[i:i + bs]
        lbs, valid = [], []
        for f in chunk:
            img = images[f] if transform is None else transform(images[f])
            lb, v = letterbox(img, size)
            lbs.append(lb); valid.append(v)
        x = torch.from_numpy(np.stack(lbs)).permute(0, 3, 1, 2).contiguous()
        preds = model.predict(x, valid, k=k)
        for f, p in zip(chunk, preds):
            out[f] = (p[0].float().cpu().numpy(), p[1].float().cpu().numpy(), p[2].cpu().numpy() if len(p) > 2 else None)
    return out


# --------------------------------------------------------------------------------------------- COCO mAP
def coco_eval(preds, gt_boxes, meta, files=None):
    """mAP@[.5:.95], AP50, AP75 with pycocotools, in original-image pixel coordinates."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    files = list(preds) if files is None else files
    m = meta.set_index("file")
    images, anns, dets = [], [], []
    for i, f in enumerate(files, 1):
        W, H = m.loc[f, "width"], m.loc[f, "height"]
        images.append({"id": i, "width": int(W), "height": int(H)})
        for b in gt_boxes[gt_boxes.file == f].itertuples():
            w, h = (b.x2 - b.x1) * W, (b.y2 - b.y1) * H
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1, "bbox": [b.x1 * W, b.y1 * H, w, h],
                         "area": w * h, "iscrowd": 0})
        bx, sc, _ = preds[f]
        for b, s in zip(bx, sc):
            dets.append({"image_id": i, "category_id": 1, "score": float(s),
                         "bbox": [float(b[0] * W), float(b[1] * H), float((b[2] - b[0]) * W), float((b[3] - b[1]) * H)]})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "piled garbage"}]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        if not dets:
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR100": 0.0}
        dt = gt.loadRes(dets)
        E = COCOeval(gt, dt, "bbox")
        E.evaluate(); E.accumulate(); E.summarize()
    s = E.stats
    return {"AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]), "AR100": float(s[8])}


# --------------------------------------------------------------------------------------------- P / R
def box_iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def nms(boxes, scores, thr=0.5):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        iou = box_iou(boxes[i], boxes[order[1:]])[0]
        order = order[1:][iou < thr]
    return np.array(keep, int)


def match_image(pb, ps, gb, thr_score, iou_thr=0.5):
    """Greedy one-to-one matching of kept detections (score >= thr, after NMS) to ground truth."""
    keep = ps >= thr_score
    pb, ps = pb[keep], ps[keep]
    if len(pb):
        k = nms(pb, ps, 0.5)
        pb, ps = pb[k], ps[k]
    iou = box_iou(pb, gb)
    tp, matched, best_iou = 0, set(), []
    for i in np.argsort(-ps):                       # highest-scoring detection first
        best_j, best = -1, iou_thr
        for j in range(iou.shape[1]):
            if j not in matched and iou[i, j] >= best:
                best_j, best = j, iou[i, j]
        if best_j >= 0:
            matched.add(best_j); tp += 1; best_iou.append(best)
    return {"tp": tp, "fp": len(pb) - tp, "fn": len(gb) - tp, "n_pred": len(pb),
            "max_iou": float(iou.max()) if iou.size else 0.0, "matched_iou": float(np.mean(best_iou)) if best_iou else np.nan}


def pr_at(preds, gt_boxes, thr, files=None, iou_thr=0.5):
    files = list(preds) if files is None else files
    rows = []
    for f in files:
        gb = gt_boxes[gt_boxes.file == f][["x1", "y1", "x2", "y2"]].values
        r = match_image(preds[f][0], preds[f][1], gb, thr, iou_thr)
        r["file"] = f
        rows.append(r)
    d = pd.DataFrame(rows)
    tp, fp, fn = d.tp.sum(), d.fp.sum(), d.fn.sum()
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": P, "recall": R, "F1": 2 * P * R / (P + R) if P + R else 0.0,
            "TP": int(tp), "FP": int(fp), "FN": int(fn)}, d


def best_f1_threshold(preds, gt_boxes, files=None, grid=np.arange(0.05, 0.96, 0.05)):
    res = [(t, pr_at(preds, gt_boxes, t, files)[0]["F1"]) for t in grid]
    return max(res, key=lambda r: (r[1], -abs(r[0] - 0.5)))[0]


# --------------------------------------------------------------------------------------------- timing
@torch.no_grad()
def benchmark(model, images, size=INPUT, warmup=10, runs=50, batch=8):
    """End-to-end latency for one image (letterbox + normalise + network + decoding), and batched throughput."""
    model.eval()
    imgs = list(images.values())
    for i in range(warmup):
        lb, v = letterbox(imgs[i % len(imgs)], size)
        model.predict(torch.from_numpy(lb[None]).permute(0, 3, 1, 2).contiguous(), [v])
    torch.cuda.synchronize()
    lat = []
    for i in range(runs):
        t = time.perf_counter()
        lb, v = letterbox(imgs[i % len(imgs)], size)
        p = model.predict(torch.from_numpy(lb[None]).permute(0, 3, 1, 2).contiguous(), [v])
        _ = p[0][1].cpu()
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t)
    lbs = [letterbox(imgs[i % len(imgs)], size) for i in range(batch)]
    x = torch.from_numpy(np.stack([l[0] for l in lbs])).permute(0, 3, 1, 2).contiguous()
    v = [l[1] for l in lbs]
    for _ in range(3):
        model.predict(x, v)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for _ in range(10):
        p = model.predict(x, v); _ = p[0][1].cpu()
    torch.cuda.synchronize()
    thr = 10 * batch / (time.perf_counter() - t)
    lat = np.array(lat) * 1000
    return {"latency_ms": float(lat.mean()), "latency_p95_ms": float(np.percentile(lat, 95)),
            "FPS_bs1": float(1000 / lat.mean()), f"FPS_bs{batch}": float(thr),
            "peak_mem_GB_bs8": torch.cuda.max_memory_allocated() / 2**30}


# --------------------------------------------------------------------------------------------- corruptions
def corrupt(kind, sev, seed=0):
    """Campus-condition corruptions applied to the photo before it reaches the model. sev in {1,2,3}."""
    rng = np.random.default_rng(seed)

    def f(img):
        x = img.astype(np.float32)
        if kind == "dark":                       # dusk / deep shade
            g = [1.6, 2.2, 3.0][sev - 1]
            return (255 * (x / 255) ** g * [0.8, 0.6, 0.4][sev - 1]).clip(0, 255).astype(np.uint8)
        if kind == "overexposed":                # harsh midday sun
            a = [1.5, 1.9, 2.4][sev - 1]
            return (x * a + 20).clip(0, 255).astype(np.uint8)
        if kind == "motion_blur":                # hand shake
            k = [9, 17, 27][sev - 1]
            ker = np.zeros((k, k), np.float32); ker[k // 2] = 1 / k
            return cv2.filter2D(img, -1, ker)
        if kind == "noise":                      # low-light sensor noise
            s = [12, 25, 40][sev - 1]
            return (x + rng.normal(0, s, x.shape)).clip(0, 255).astype(np.uint8)
        if kind == "haze":                       # fog / morning haze
            a = [0.35, 0.55, 0.7][sev - 1]
            return (x * (1 - a) + 215 * a).clip(0, 255).astype(np.uint8)
        if kind == "occlusion":                  # vegetation / objects covering parts of the scene
            out = img.copy(); h, w = img.shape[:2]
            n, s = [3, 5, 8][sev - 1], [0.18, 0.22, 0.26][sev - 1]
            for _ in range(n):
                bw, bh = int(s * w), int(s * h)
                x0, y0 = rng.integers(0, w - bw), rng.integers(0, h - bh)
                out[y0:y0 + bh, x0:x0 + bw] = (40, 90, 35)
            return out
        raise ValueError(kind)
    return f


CORRUPTIONS = ["dark", "overexposed", "motion_blur", "noise", "haze", "occlusion"]
