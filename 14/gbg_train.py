"""Fine-tuning loop shared by DETR, DINO, RT-DETR and Grounding DINO. Resumable: a killed run continues from
the last finished epoch; a finished run is skipped and its history returned."""
import json, time, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gbg_data as GD
import gbg_models as GM
import gbg_eval as GE

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "outputs" / "runs"

DEFAULT = dict(epochs=60, batch=4, accum=1, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, clip=0.1,
               lr_drop=0.75, aug="full", size=640, eval_every=2, seed=0)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def train(name, kind, train_files, val_files, boxes, meta, cfg=None, model_kw=None, log=print):
    cfg = {**DEFAULT, **(cfg or {})}
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"kind": kind, **cfg, **(model_kw or {})}, indent=1, default=str))
    if (out / "done.json").exists():
        return pd.read_csv(out / "history.csv"), json.loads((out / "done.json").read_text())

    set_seed(cfg["seed"])
    model = GM.build(kind, **(model_kw or {}))
    opt = torch.optim.AdamW(model.param_groups(cfg["lr"], cfg["lr_backbone"], cfg["weight_decay"]))
    drop_at = int(round(cfg["lr_drop"] * cfg["epochs"]))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [drop_at], 0.1)
    hist, start, best, train_time = [], 0, (-1.0, -1.0), 0.0
    if (out / "last.pt").exists():                                      # resume
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        hist, start, best, train_time = ck["hist"], ck["epoch"], tuple(ck["best"]), ck["train_time"]
        log(f"[{name}] resuming at epoch {start}")

    ds = GD.GarbageDataset(train_files, boxes, augment=GD.build_augment(cfg["aug"]), size=cfg["size"])
    g = torch.Generator(); g.manual_seed(cfg["seed"] + start)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch"] // cfg["accum"], shuffle=True, num_workers=0,
                                     collate_fn=GD.collate, generator=g, drop_last=True)
    val_imgs = GE.load_cached(val_files)
    torch.cuda.reset_peak_memory_stats()
    for ep in range(start, cfg["epochs"]):
        model.train()
        t0, logs, step = time.time(), [], 0
        opt.zero_grad(set_to_none=True)
        for imgs, bx, valid, _ in dl:
            loss, ld = model.loss(imgs, bx, valid)
            (loss / cfg["accum"]).backward()
            step += 1
            if step % cfg["accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
                opt.step(); opt.zero_grad(set_to_none=True)
            logs.append({"loss": float(loss), **ld})
        sched.step()
        torch.cuda.synchronize()
        ep_time = time.time() - t0
        train_time += ep_time
        row = {"epoch": ep + 1, "time_s": ep_time, **pd.DataFrame(logs).mean().to_dict()}
        if (ep + 1) % cfg["eval_every"] == 0 or ep + 1 == cfg["epochs"]:
            preds = GE.predict_images(model, val_imgs, size=cfg["size"])
            m = GE.coco_eval(preds, boxes, meta, list(val_imgs))
            row.update({f"val_{k}": v for k, v in m.items()})
            key = (m["AP"], m["AP50"])
            if key > best:
                best = key
                torch.save(model.state_dict(), out / "best.pt")
                row["saved_best"] = 1
            log(f"[{name}] ep {ep + 1:3d}/{cfg['epochs']} loss {row['loss']:.3f} | val AP {m['AP']:.3f} AP50 {m['AP50']:.3f}"
                f" | {ep_time:.1f}s/ep")
        hist.append(row)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "hist": hist,
                    "epoch": ep + 1, "best": best, "train_time": train_time}, out / "last.pt")
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    done = {"train_time_s": train_time, "best_val_AP": best[0], "best_val_AP50": best[1],
            "peak_train_mem_GB": torch.cuda.max_memory_allocated() / 2**30, "params": model.n_params(),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (out / "done.json").write_text(json.dumps(done, indent=1))
    (out / "last.pt").unlink(missing_ok=True)                           # keep only best.pt to save disk
    del model, opt
    torch.cuda.empty_cache()
    return pd.DataFrame(hist), done


def load_finetuned(name, kind, model_kw=None):
    m = GM.build(kind, **(model_kw or {}))
    m.load_state_dict(torch.load(RUNS / name / "best.pt", map_location="cpu", weights_only=False))
    m.eval()
    return m
