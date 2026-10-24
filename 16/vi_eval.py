"""Metrics, speed benchmark, temporal-order tests, timestamp localisation on spliced videos and the long-range
memory probe -- identical for every model."""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import vi_train as TR


def metrics(y, p, thr=0.5):
    """Dirty = positive class."""
    from sklearn.metrics import roc_auc_score
    y, yhat = np.asarray(y).astype(int), (np.asarray(p) >= thr).astype(int)
    tp, tn = int(((yhat == 1) & (y == 1)).sum()), int(((yhat == 0) & (y == 0)).sum())
    fp, fn = int(((yhat == 1) & (y == 0)).sum()), int(((yhat == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "AUC": roc_auc_score(y, p) if len(set(y)) == 2 else np.nan, "TP": tp, "TN": tn, "FP": fp, "FN": fn}


# ------------------------------------------------------------------------------------------------ speed
@torch.no_grad()
def latency(model, T=16, device="cuda", batch=1, runs=200, warmup=20, dim=2048):
    """Mean wall-clock time of one forward pass over a batch of T-frame feature sequences (ms)."""
    model = model.to(device).eval()
    x = torch.randn(batch, T, dim, device=device)
    L = torch.full((batch,), T)
    for _ in range(warmup):
        model(x, L)
    if device == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); model(x, L)
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    model.to("cuda")
    return 1000 * float(np.median(ts))


# ------------------------------------------------------------------------------------------------ temporal tests
def temporal_transform(kind, seed=0):
    """Changes to the frames of a clip that keep its content but destroy (part of) its order."""
    g = torch.Generator().manual_seed(seed)

    def f(x):
        n = len(x)
        if kind == "shuffle":
            return x[torch.randperm(n, generator=g).to(x.device)]
        if kind == "reverse":
            return x.flip(0)
        if kind == "middle_frame":            # one frame repeated: the clip carries no temporal information
            return x[n // 2].expand(n, -1)
        if kind == "drop_half":               # half the frames removed (= halved sampling rate, same span)
            return x[::2] if n > 1 else x
        return x
    return f


def temporal_saliency(model, x):
    """|d logit(dirty) / d frame_t| for every time step of equal-length clips x (B, T, D), averaged over the batch."""
    model.eval()
    x = x.clone().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):         # cuDNN RNN has no backward in eval mode
        L = torch.full((len(x),), x.shape[1])
        z = model(x, L)
        (z[:, 1] - z[:, 0]).sum().backward()
    return x.grad.norm(dim=-1).mean(0).cpu().numpy()


@torch.no_grad()
def prefix_probs(model, x):
    """P(dirty) after each time step for equal-length clips (B, T, D) -> (B, T)."""
    model.eval()
    return model.step_logits(x).softmax(-1)[..., 1].cpu().numpy()


# ------------------------------------------------------------------------------------------------ localisation
def sampled(store, v, fps, phase=0):
    idx = np.arange(phase, store.n(v), TR.step_of(fps))
    return store.f[v][torch.as_tensor(idx, device="cuda")], store.t[v][idx]


def make_composites(store, clean, dirty, n=40, fps=2.0, seed=0, clean_s=(8, 30), dirty_s=(4, 16)):
    """Spliced test streams  clean | dirty | clean (| dirty | clean)  built from held-out videos, with the true
    dirty intervals known. Every piece is a random contiguous part of one video."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        n_dirty = 1 if i % 2 == 0 else 2
        pattern = [0] + [1, 0] * n_dirty
        xs, ys, src = [], [], []
        for lab in pattern:
            v = rng.choice(clean if lab == 0 else dirty)
            f, _ = sampled(store, v, fps)
            want = int(rng.uniform(*(clean_s if lab == 0 else dirty_s)) * fps)
            want = min(want, len(f))
            s = rng.integers(len(f) - want + 1)
            xs.append(f[s:s + want]); ys += [lab] * want; src.append((v, want))
        y = np.array(ys)
        out.append({"x": torch.cat(xs), "y": y, "t": np.arange(len(y)) / fps, "sources": src})
    return out


@torch.no_grad()
def timeline(model, x, W, bs=512):
    """P(dirty) at every sampled frame from a window of W frames centred on it (shortened at the ends)."""
    model.eval()
    n = len(x)
    seqs = [x[max(0, t - W // 2): min(n, t - W // 2 + W)] for t in range(n)]
    seqs = [s if len(s) else x[t:t + 1] for t, s in enumerate(seqs)]
    p = []
    for i in range(0, n, bs):
        xb, L = TR.pad(seqs[i:i + bs])
        p.append(model(xb, L).softmax(-1)[:, 1].cpu())
    return torch.cat(p).numpy()


def segments(mask, t, min_len_s=1.0, merge_gap_s=1.0, dt=0.5):
    """Runs of True in a per-frame mask -> list of (start_s, end_s); short gaps merged, short runs dropped."""
    segs, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        if not m and start is not None:
            segs.append([t[start], t[i - 1] + dt]); start = None
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [tuple(s) for s in merged if s[1] - s[0] >= min_len_s]


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def localisation_scores(p, y, t, dt=0.5, thr=0.5):
    """Frame-level metrics plus event-level matching (IoU >= 0.5) and boundary error of matched events."""
    m = metrics(y, p, thr)
    gt = segments(y.astype(bool), t, min_len_s=0, merge_gap_s=0, dt=dt)
    pr = segments(p >= thr, t, dt=dt)
    used, matched, berr = set(), 0, []
    for g in gt:
        best = max(((i, _iou(g, q)) for i, q in enumerate(pr) if i not in used), key=lambda z: z[1], default=(None, 0))
        if best[1] >= 0.5:
            used.add(best[0]); matched += 1
            q = pr[best[0]]; berr += [abs(q[0] - g[0]), abs(q[1] - g[1])]
    ev_p = matched / len(pr) if pr else 0.0
    ev_r = matched / len(gt) if gt else 0.0
    return {"frame_acc": m["accuracy"], "frame_F1": m["F1"], "event_precision": ev_p, "event_recall": ev_r,
            "event_F1": 2 * ev_p * ev_r / (ev_p + ev_r) if ev_p + ev_r else 0.0,
            "boundary_err_s": float(np.mean(berr)) if berr else np.nan,
            "false_segments": len(pr) - matched, "switches": int(np.abs(np.diff((p >= thr).astype(int))).sum())}


# ------------------------------------------------------------------------------------------------ memory probe
class ProbeSampler:
    """'Does this stream contain a dirty moment?' Background = clean clips of 4-16 frames from random clean videos
    spliced together; one k-frame piece is inserted at position `pos`: from a dirty video (label 1) or from yet
    another clean video (label 0), so a splice alone is no evidence. Solving it needs memory of the inserted
    piece until the last frame."""
    def __init__(self, store, clean, dirty, fps=2.0):
        self.f = {v: sampled(store, v, fps)[0] for v in list(clean) + list(dirty)}
        self.clean, self.dirty = list(clean), list(dirty)

    def piece(self, vids, k, rng):
        v = vids[rng.integers(len(vids))]
        f = self.f[v]
        k = min(k, len(f)); s = rng.integers(len(f) - k + 1)
        return f[s:s + k]

    def one(self, L, k, label, pos, rng):
        bg, n = [], 0
        while n < L - k:
            p = self.piece(self.clean, int(rng.integers(4, 17)), rng); bg.append(p); n += len(p)
        bg = torch.cat(bg)[:L - k]
        ins = self.piece(self.dirty if label else self.clean, k, rng)
        if len(ins) < k:
            ins = ins.repeat(k // len(ins) + 1, 1)[:k]
        return torch.cat([bg[:pos], ins, bg[pos:]])

    def batch(self, B, L, k, rng, pos=None):
        y = rng.integers(0, 2, B)
        ps = rng.integers(0, L - k + 1, B) if pos is None else np.full(B, pos)
        x = torch.stack([self.one(L, k, int(yy), int(p), rng) for yy, p in zip(y, ps)])
        return x, torch.full((B,), L), torch.as_tensor(y, device="cuda")


def probe_run(cell, train_s, val_s, test_s, L, k=2, seed=0, epochs=40, steps=25, B=64, hidden=128, n_test=256, cfg=None,
              readout=None):
    """Trains one cell on the memory probe and returns accuracy as a function of the insert position."""
    import vi_models as M
    cfg = {**TR.DEFAULT, "label_smoothing": 0.0, **(cfg or {})}
    TR.set_seed(seed)
    model = M.build(cell, hidden=hidden, dropout=cfg["dropout"], **({"readout": readout} if readout else {})).cuda()
    model.norm.fit(torch.cat(list(train_s.f.values())))
    rng = np.random.default_rng(seed)
    vrng = np.random.default_rng(10_000 + seed)
    val = [val_s.batch(B, L, k, vrng) for _ in range(4)]

    def val_loss(m):
        m.eval()
        with torch.no_grad():
            ls = [(F.cross_entropy(m(x, l), y).item(), (m(x, l).argmax(-1) == y).float().mean().item()) for x, l, y in val]
        return float(np.mean([a for a, _ in ls])), float(np.mean([b for _, b in ls]))
    hist, best_ep, t = TR.train_model(model, lambda: train_s.batch(B, L, k, rng), val_loss, epochs, steps, cfg)
    model.eval()
    rows, trng = [], np.random.default_rng(20_000 + seed)
    for pos in np.unique(np.linspace(0, L - k, min(9, L - k + 1)).astype(int)):
        with torch.no_grad():
            x, l, y = test_s.batch(n_test, L, k, trng, pos=int(pos))
            pred = model(x, l).argmax(-1)
        yy = y.cpu().numpy(); pp = pred.cpu().numpy()
        rows.append({"cell": cell, "L": L, "seed": seed, "pos": int(pos), "frames_after": int(L - k - pos),
                     "accuracy": float((pp == yy).mean()), "recall": float(pp[yy == 1].mean()),
                     "specificity": float(1 - pp[yy == 0].mean())})
    return pd.DataFrame(rows), pd.DataFrame(hist), best_ep
