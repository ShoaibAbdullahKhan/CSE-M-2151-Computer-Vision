"""PASCAL-Part loading, part-class merging, YOLO-seg dataset building and test-set selection."""
import re, json, random
from pathlib import Path
import numpy as np
import cv2
import scipy.io as sio
from PIL import Image
from bp_common import PARTS, SPECIES, SPECIES_PARTS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ANNO = DATA / "pascal_part" / "Annotations_Part"
VOC = DATA / "VOCdevkit" / "VOC2010"


def merge_key(species, raw):
    """Raw PASCAL-Part part name -> (merged class, instance key). Left/right copies stay separate instances;
    upper+lower segments of the same leg are merged into one 'leg' instance."""
    if raw in ("head", "nose", "muzzle", "neck", "torso", "tail", "beak"):
        return raw, raw
    if raw in ("lear", "rear"):
        return "ear", raw
    if raw in ("leye", "reye"):
        return "eye", raw
    if raw in ("lhorn", "rhorn"):
        return "horn", raw
    if raw in ("lwing", "rwing"):
        return "wing", raw
    if raw in ("lfoot", "rfoot"):
        return "foot", raw
    if raw in ("lleg", "rleg"):                                  # bird legs
        return "leg", raw
    m = re.fullmatch(r"([lr][fb])(pa|ho|leg|lleg|uleg)", raw)    # e.g. lfpa, rbho, lfleg, rblleg
    if m:
        side, kind = m.groups()
        if kind == "pa":
            return "paw", side + "pa"
        if kind == "ho":
            return "hoof", side + "ho"
        return "leg", side + "leg"
    return None, None


def load_anno(img_id):
    """List of animal objects: {'species','mask','parts': {part: [instance masks]}}."""
    a = sio.loadmat(str(ANNO / f"{img_id}.mat"))["anno"][0, 0]["objects"]
    objs = []
    for o in a[0]:
        sp = str(o["class"][0])
        if sp not in SPECIES:
            continue
        inst = {}
        for p in (o["parts"][0] if o["parts"].size else []):
            cls, key = merge_key(sp, str(p["part_name"][0]))
            if cls is None:
                continue
            m = p["mask"].astype(bool)
            inst[(cls, key)] = inst.get((cls, key), np.zeros_like(m)) | m
        parts = {}
        for (cls, _), m in inst.items():
            parts.setdefault(cls, []).append(m)
        objs.append({"species": sp, "mask": o["mask"].astype(bool), "parts": parts})
    return objs


def voc_split(name):
    return [l.strip() for l in open(VOC / "ImageSets" / "Main" / f"{name}.txt") if l.strip()]


def image_path(img_id):
    return VOC / "JPEGImages" / f"{img_id}.jpg"


def animal_ids(ids):
    have = {p.stem for p in ANNO.glob("*.mat")}
    return [i for i in ids if i in have]


# --------------------------------------------------------------------------------------------- YOLO-seg crops
def _mask_to_polys(m, min_area=6):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2) for c in cnts if cv2.contourArea(c) >= min_area and len(c) >= 3]


def _merge_polys(polys):
    """Join several contours of one instance into a single polygon (ultralytics needs one polygon per line)."""
    if len(polys) == 1:
        return polys[0]
    from ultralytics.data.converter import merge_multi_segment
    return np.concatenate(merge_multi_segment([p.reshape(-1) for p in polys]), 0)


def build_yolo_crops(ids, out_dir, split, pad=0.15, min_side=48):
    """One training image per animal instance: the padded animal box, with every part polygon inside it."""
    img_dir, lbl_dir = Path(out_dir) / "images" / split, Path(out_dir) / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_id in ids:
        objs = load_anno(img_id)
        if not objs:
            continue
        img = cv2.imread(str(image_path(img_id)))
        H, W = img.shape[:2]
        for k, o in enumerate(objs):
            ys, xs = np.where(o["mask"])
            if len(xs) == 0 or not o["parts"]:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            if min(x2 - x1, y2 - y1) < min_side:
                continue
            pw, ph = pad * (x2 - x1), pad * (y2 - y1)
            cx1, cy1, cx2, cy2 = int(max(0, x1 - pw)), int(max(0, y1 - ph)), int(min(W, x2 + pw)), int(min(H, y2 + ph))
            cw, ch = cx2 - cx1, cy2 - cy1
            lines = []
            for oo in objs:                                   # parts of every animal visible in the crop
                for cls, masks in oo["parts"].items():
                    for m in masks:
                        polys = _mask_to_polys(m[cy1:cy2, cx1:cx2])
                        if not polys:
                            continue
                        poly = _merge_polys(polys).astype(np.float32)
                        poly[:, 0] /= cw
                        poly[:, 1] /= ch
                        lines.append(f"{PARTS.index(cls)} " + " ".join(f"{v:.5f}" for v in poly.clip(0, 1).reshape(-1)))
            if not lines:
                continue
            name = f"{img_id}_{k}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img[cy1:cy2, cx1:cx2], [cv2.IMWRITE_JPEG_QUALITY, 95])
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines))
            n += 1
    return n


def write_yaml(out_dir):
    p = Path(out_dir) / "parts.yaml"
    p.write_text(f"path: {Path(out_dir).as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
                 + "".join(f"  {i}: {n}\n" for i, n in enumerate(PARTS)))
    return p


# --------------------------------------------------------------------------------------------- test set
def select_test_images(ids, per_species, seed=0, min_frac=0.08):
    """Images with exactly one animal (so 'this cat' is unambiguous) covering >= min_frac of the image."""
    rng = random.Random(seed)
    pools = {s: [] for s in SPECIES}
    for img_id in ids:
        objs = load_anno(img_id)
        if len(objs) != 1 or len(objs[0]["parts"]) < 3:
            continue
        o = objs[0]
        if o["mask"].mean() >= min_frac:
            pools[o["species"]].append(img_id)
    out = []
    for s, n in per_species.items():
        rng.shuffle(pools[s])
        out += [(i, s) for i in sorted(pools[s][:n])]
    return out, {s: len(v) for s, v in pools.items()}
