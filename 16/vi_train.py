"""Clip sampling, the training recipe shared by every model, and clip / video prediction. Runs are resumable:
a finished run is skipped and its saved result returned."""
import json, math, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import vi_data as D
import vi_models as M

RUNS = D.OUT / "runs"
DEFAULT = dict(hidden=128, layers=1, dropout=0.5, fps=2.0, T=16, eval_T=None, epochs=30, clips_per_video=16, batch=32, lr=3e-4,
               weight_decay=1e-2, label_smoothing=0.05, flip_p=0.5, seed=0)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Store:
    """Frame features of every video on the GPU: f (N, 2048), ff (flipped frames) and the frame times."""
    def __init__(self, vids, kind="none", subdir="resnet50"):
        self.f, self.ff, self.t = {}, {}, {}
        for v in vids:
            d = D.video_features(v, kind, subdir)
            self.f[v] = torch.from_numpy(d["f"].astype(np.float32)).cuda()
            self.ff[v] = torch.from_numpy(d["ff"].astype(np.float32)).cuda() if "ff" in d else self.f[v]
            self.t[v] = d["t"]

    def n(self, v):
        return len(self.f[v])


def step_of(fps):
    s = D.BASE_FPS / fps
    assert abs(s - round(s)) < 1e-6, "fps must divide the base rate"
    return int(round(s))


def eval_clips(n, fps, T, phase=0, stride=None):
    """Frame indices of the evaluation clips of one video: windows of T sampled frames, 50 % overlap,
    the last window aligned with the end; one shorter clip when the video has fewer than T sampled frames."""
    idx = np.arange(phase, n, step_of(fps))
    if len(idx) <= T:
        return [idx]
    stride = stride or max(1, T // 2)
    starts = list(range(0, len(idx) - T + 1, stride))
    if starts[-1] != len(idx) - T:
        starts.append(len(idx) - T)
    return [idx[s:s + T] for s in starts]


def random_clip(n, fps, T, rng):
    step = step_of(fps)
    idx = np.arange(rng.integers(step), n, step)       # random phase of the sampling grid
    if len(idx) <= T:
        return idx
    s = rng.integers(len(idx) - T + 1)
    return idx[s:s + T]


def pad(seqs):
    L = torch.tensor([len(s) for s in seqs])
    x = torch.zeros(len(seqs), int(L.max()), seqs[0].shape[-1], device=seqs[0].device)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
    return x, L


def fit_norm(model, store, vids):
    model.norm.fit(torch.cat([store.f[v] for v in vids]))


def train_model(model, sample_batch, val_loss, epochs, steps_per_epoch, cfg, class_weight=None):
    """Generic loop: AdamW + cosine schedule, best epoch chosen by validation loss."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    best, best_state, hist = (float("inf"), -1), None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        losses, correct, n = [], 0, 0
        for _ in range(steps_per_epoch):
            x, L, y = sample_batch()
            logits = model(x, L)
            loss = F.cross_entropy(logits, y, weight=class_weight, label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(loss.item()); correct += (logits.argmax(-1) == y).sum().item(); n += len(y)
        vl, vacc = val_loss(model)
        hist.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "train_acc": correct / n, "val_loss": vl, "val_acc": vacc})
        if vl < best[0] - 1e-6:
            best = (vl, ep + 1)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return hist, best[1], time.time() - t0


@torch.no_grad()
def predict_clips(model, store, vids, fps, T, phase=0, transform=None, bs=256):
    """P(dirty) of every evaluation clip. transform(x) may reorder / alter the frames of a clip."""
    model.eval()
    rows, seqs = [], []
    for v in vids:
        for k, idx in enumerate(eval_clips(store.n(v), fps, T, phase)):
            x = store.f[v][torch.as_tensor(idx, device="cuda")]
            seqs.append(transform(x) if transform else x)
            rows.append({"vid": v, "clip": k, "t_start": float(store.t[v][idx[0]]), "t_end": float(store.t[v][idx[-1]]), "n": len(idx)})
    p = []
    for i in range(0, len(seqs), bs):
        x, L = pad(seqs[i:i + bs])
        p.append(model(x, L).float().softmax(-1)[:, 1].cpu())
    df = pd.DataFrame(rows)
    df["p_dirty"] = torch.cat(p).numpy() if p else []
    return df


def video_scores(clips):
    """Video index score = mean clip P(dirty); also the fraction of clips called dirty."""
    g = clips.groupby("vid", sort=False)
    return pd.DataFrame({"p_dirty": g.p_dirty.mean(), "frac_dirty_clips": g.p_dirty.apply(lambda p: (p >= 0.5).mean()),
                         "n_clips": g.size()}).reset_index()


def clip_nll(model, store, vids, labels, fps, T):
    c = predict_clips(model, store, vids, fps, T)
    y = c.vid.map(labels).values
    p = np.clip(c.p_dirty.values, 1e-6, 1 - 1e-6)
    c["nll"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    c["ok"] = (p >= 0.5) == y
    per_vid = c.groupby("vid")[["nll", "ok"]].mean()          # every video counts equally
    return float(per_vid.nll.mean()), float(per_vid.ok.mean())


def train_run(name, cell, split, labels, store, cfg=None, save_weights=False, extra_eval=None):
    """split: {'train': [...], 'val': [...], 'test': [...]} video ids; labels: {vid: 0/1}.
    extra_eval: {tag: (Store, vids)} evaluated with the best checkpoint (e.g. perturbed copies of the test videos)."""
    cfg = {**DEFAULT, **(cfg or {})}
    eT = cfg["eval_T"] or cfg["T"]          # clip length for validation / test (the frame baseline trains on T=1)
    out = RUNS / name
    if (out / "result.json").exists():
        return load_result(name)
    set_seed(cfg["seed"])
    model = M.build(cell, hidden=cfg["hidden"], layers=cfg["layers"], dropout=cfg["dropout"], **readout_kw(cell, cfg)).cuda()
    fit_norm(model, store, split["train"])
    tr = split["train"]
    ytr = np.array([labels[v] for v in tr])
    w = torch.tensor([len(ytr) / (2 * (ytr == c).sum()) for c in (0, 1)], dtype=torch.float32).cuda()
    rng = np.random.default_rng(cfg["seed"])
    order = []

    def sample_batch():
        nonlocal order
        if len(order) < cfg["batch"]:                       # every training video contributes the same number of clips
            order += list(rng.permutation(np.repeat(np.arange(len(tr)), cfg["clips_per_video"])))
        pick, order = order[:cfg["batch"]], order[cfg["batch"]:]
        seqs = []
        for i in pick:
            v = tr[i]
            idx = torch.as_tensor(random_clip(store.n(v), cfg["fps"], cfg["T"], rng), device="cuda")
            src = store.ff[v] if rng.random() < cfg["flip_p"] else store.f[v]
            seqs.append(src[idx])
        x, L = pad(seqs)
        return x, L, torch.as_tensor(ytr[pick], device="cuda")

    steps = max(1, len(tr) * cfg["clips_per_video"] // cfg["batch"])
    hist, best_ep, t_train = train_model(model, sample_batch, lambda m: clip_nll(m, store, split["val"], labels, cfg["fps"], eT),
                                         cfg["epochs"], steps, cfg, w)
    out.mkdir(parents=True, exist_ok=True)
    clips = predict_clips(model, store, split["test"], cfg["fps"], eT)
    clips["label"] = clips.vid.map(labels)
    clips.to_csv(out / "test_clips.csv", index=False)
    vids = video_scores(clips); vids["label"] = vids.vid.map(labels)
    vids.to_csv(out / "test_videos.csv", index=False)
    for tag, (st, vv) in (extra_eval or {}).items():
        c = predict_clips(model, st, vv, cfg["fps"], eT); c["label"] = c.vid.map(labels)
        c.to_csv(out / f"extra_{tag}.csv", index=False)
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    if save_weights:
        torch.save({"cell": cell, "cfg": cfg, "state": model.state_dict()}, out / "best.pt")
    res = {"name": name, "cell": cell, "cfg": cfg, "best_epoch": best_ep, "train_time_s": t_train,
           "params": M.n_params(model), "split": split}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return load_result(name)


def readout_kw(cell, cfg):
    """Only recurrent cells take the readout option ('last' hidden state by default, 'mean' in the ablation)."""
    return {"readout": cfg["readout"]} if cell in M.CELLS and cfg.get("readout") else {}


def load_result(name):
    out = RUNS / name
    res = json.loads((out / "result.json").read_text())
    res["clips"] = pd.read_csv(out / "test_clips.csv")
    res["videos"] = pd.read_csv(out / "test_videos.csv")
    res["history"] = pd.read_csv(out / "history.csv")
    res["extra"] = {p.stem[6:]: pd.read_csv(p) for p in out.glob("extra_*.csv")}
    return res


def load_model(name):
    ck = torch.load(RUNS / name / "best.pt", map_location="cuda")
    c = ck["cfg"]
    m = M.build(ck["cell"], hidden=c["hidden"], layers=c["layers"], dropout=c["dropout"], **readout_kw(ck["cell"], c)).cuda()
    m.load_state_dict(ck["state"]); m.eval()
    return m, c
