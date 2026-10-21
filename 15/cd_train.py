"""Fine-tuning and evaluation loop shared by all backbones (identical recipe for every model). Resumable: a finished
run is skipped and its saved predictions are returned."""
import json, time, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cd_data as D
import cd_models as M

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "outputs" / "runs"

DEFAULT = dict(epochs=25, batch=16, lr=1e-4, head_lr=1e-3, weight_decay=0.05, warmup=2, label_smoothing=0.1,
               size=224, aug="standard", freeze_backbone=False, amp=True, seed=0)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, files, cache, size=224, bs=32, transform=None):
    """Returns P(dirty) for every file (test-time transform = resize + centre crop unless given)."""
    model.eval()
    tf = transform or D.make_transforms(size, train=False)
    out = []
    for i in range(0, len(files), bs):
        x = torch.stack([tf(cache[f]) for f in files[i:i + bs]]).cuda()
        with torch.autocast("cuda", torch.bfloat16):
            out.append(model(x).float().softmax(-1)[:, 1].cpu())
    return torch.cat(out).numpy()


def train_run(name, model_name, train_files, train_labels, val_files, val_labels, cache, cfg=None, eval_sets=None,
              log=print, save_weights=False):
    """eval_sets: {tag: list_of_files}; their P(dirty) under the best checkpoint is saved with the run."""
    cfg = {**DEFAULT, **(cfg or {})}
    out = RUNS / name
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    model = M.build(model_name).cuda()
    head = set(id(p) for p in model.head_params())
    if cfg["freeze_backbone"]:
        for p in model.parameters():
            if id(p) not in head:
                p.requires_grad_(False)
    groups = [{"params": [p for p in model.parameters() if id(p) in head], "lr": cfg["head_lr"]}]
    body = [p for p in model.parameters() if id(p) not in head and p.requires_grad]
    if body:
        groups.append({"params": body, "lr": cfg["lr"]})
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    tr = D.ImageSet(train_files, train_labels, D.make_transforms(cfg["size"], True, cfg["aug"]), cache)
    g = torch.Generator(); g.manual_seed(cfg["seed"])
    dl = torch.utils.data.DataLoader(tr, batch_size=cfg["batch"], shuffle=True, num_workers=0, drop_last=True, generator=g)
    steps = cfg["epochs"] * len(dl)
    warm = cfg["warmup"] * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    # class weights: the classes are slightly imbalanced (140 clean / 116 dirty)
    w = torch.tensor([len(train_labels) / (2 * (np.array(train_labels) == c).sum()) for c in (0, 1)], dtype=torch.float32).cuda()
    hist, best, best_state = [], (1e9, -1), None
    torch.cuda.reset_peak_memory_stats()
    t_train = 0.0
    for ep in range(cfg["epochs"]):
        model.train()
        t0, losses, correct, n = time.time(), [], 0, 0
        for x, y in dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16, enabled=cfg["amp"]):
                logits = model(x).float()
            loss = F.cross_entropy(logits, y, weight=w, label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(loss.item()); correct += (logits.argmax(-1) == y).sum().item(); n += len(y)
        torch.cuda.synchronize(); t_train += time.time() - t0
        pv = predict(model, val_files, cache, cfg["size"])
        yv = np.array(val_labels)
        vloss = float(F.binary_cross_entropy(torch.tensor(pv).clamp(1e-6, 1 - 1e-6), torch.tensor(yv, dtype=torch.float32)))
        vacc = float(((pv >= 0.5) == yv).mean())
        hist.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "train_acc": correct / n,
                     "val_loss": vloss, "val_acc": vacc, "epoch_time": time.time() - t0})
        if (vloss, -vacc) < (best[0], -best[1]):                       # model selection: lowest validation loss
            best = (vloss, vacc)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            hist[-1]["best"] = 1
    model.load_state_dict(best_state)
    res = {"model": model_name, "cfg": cfg, "train_time_s": t_train, "best_val_loss": best[0], "best_val_acc": best[1],
           "peak_train_mem_GB": torch.cuda.max_memory_allocated() / 2**30,
           "params": M.n_params(model), "trainable": M.n_params(model, trainable=True), "history": hist,
           "preds": {}}
    tf = D.make_transforms(cfg["size"], train=False)
    for tag, spec in (eval_sets or {}).items():          # spec = files  or  (files, PIL->PIL function)
        files, fn = (spec, None) if not isinstance(spec, tuple) else spec
        src = cache if fn is None else {f: fn(cache[f]) for f in files}
        res["preds"][tag] = dict(zip(files, predict(model, files, src, cfg["size"], transform=tf).tolist()))
    if save_weights:
        torch.save(model.state_dict(), out / "best.pt")
    (out / "result.json").write_text(json.dumps(res))
    del model, opt
    torch.cuda.empty_cache()
    return res


def load_trained(name, model_name):
    m = M.build(model_name).cuda()
    m.load_state_dict(torch.load(RUNS / name / "best.pt", map_location="cpu", weights_only=False))
    return m.eval()
