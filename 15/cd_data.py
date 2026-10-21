"""Clean vs. Dirty dataset: metadata, near-duplicate groups, grouped split, transforms and in-memory datasets."""
import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None            # three clean photos are 12000 x 9000 (108 MP)
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data"
CACHE = ROOT / "cache" / "img"           # EXIF-corrected copies, shorter side 320 px
CLASSES = ["clean", "dirty"]             # label 0 = clean, 1 = dirty (the positive class)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# Scene type of every clean photo (index = position in sorted file order), labelled by hand from contact sheets.
CLEAN_SCENES = (
    [("park / vegetation", range(0, 2)), ("water / beach", range(2, 9)), ("sports field", range(9, 11)),
     ("street / path", range(11, 13)), ("street / path", range(13, 14)), ("water / beach", range(14, 20)),
     ("sports field", range(20, 55)), ("street / path", range(55, 56)), ("water / beach", range(56, 59)),
     ("park / vegetation", range(59, 67)), ("street / path", range(67, 69)), ("water / beach", range(69, 71)),
     ("building / campus", range(71, 73)), ("water / beach", range(73, 81)), ("building / campus", range(81, 84)),
     ("park / vegetation", range(84, 87)), ("park / vegetation", range(87, 91)), ("building / campus", range(91, 93)),
     ("street / path", range(93, 95)), ("building / campus", range(95, 104)), ("street / path", range(104, 106)),
     ("building / campus", range(106, 113)), ("water / beach", range(113, 116)), ("park / vegetation", range(116, 119)),
     ("street / path", range(119, 128)), ("park / vegetation", range(128, 130)), ("street / path", range(130, 132)),
     ("park / vegetation", range(132, 139)), ("water / beach", range(139, 140))])


def load_image(path, short=320):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = short / min(im.size)
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    return im


def build_metadata(a14_meta=ROOT / "annotations" / "dirty_metadata_from_a14.csv"):
    """One row per image with label, original size, near-duplicate group and descriptive attributes."""
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    scene = {}
    for name, rng in CLEAN_SCENES:
        for i in rng:
            scene[i] = name
    a14 = pd.read_csv(a14_meta, comment="#").set_index("file")          # the dirty photos are Assignment 14's photos
    for label, cls in enumerate(CLASSES):
        for i, f in enumerate(sorted((RAW / cls).glob("*.jpg"))):
            with Image.open(f) as im:
                exif = im.getexif()
                w, h = im.size
                orient = exif.get(274)
                make = exif.get(271)
            if orient in (5, 6, 7, 8):
                w, h = h, w
            out = CACHE / cls / f.name
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                load_image(f).save(out, quality=94)
            r = {"file": f"{cls}/{f.name}", "label": label, "cls": cls, "width": w, "height": h,
                 "orientation": "portrait" if h > w else "landscape", "camera": make or "unknown"}
            if cls == "clean":
                r.update(scene=scene[i], source="clean set")
            else:
                m = a14.loc[f.name]
                r.update(scene="garbage pile", source=f"dirty session {m['session']}", a14_group=m["group"],
                         composition=m["composition"], background=m["background"], occluded=int(m["occluded"]))
            rows.append(r)
    meta = pd.DataFrame(rows)
    cached = [np.asarray(Image.open(CACHE / f).convert("L"), np.float32) / 255 for f in meta.file]
    meta["brightness"] = [c.mean() for c in cached]
    meta["sky_fraction"] = [sky_fraction(Image.open(CACHE / f).convert("RGB")) for f in meta.file]
    return meta


def sky_fraction(im):
    """Share of the upper half of the image that looks like sky (bright and bluish or bright and unsaturated)."""
    a = np.asarray(im.resize((64, 64)), np.float32) / 255
    top = a[:32]
    r, g, b = top[..., 0], top[..., 1], top[..., 2]
    v = top.max(-1)
    s = (v - top.min(-1)) / (v + 1e-6)
    sky = (v > 0.55) & (((b >= r) & (b >= g * 0.95)) | (s < 0.12))
    return float(sky.mean())


def assign_groups(meta, feats_clean, link_next=0.5, link_any=0.8):
    """Near-duplicate groups. Dirty: the location groups of Assignment 14 (same photos). Clean: consecutive photos with
    cosine similarity >= link_next, or any two photos with similarity >= link_any, share a group (union-find)."""
    clean_idx = meta.index[meta.cls == "clean"].tolist()
    parent = list(range(len(clean_idx)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sim = feats_clean @ feats_clean.T
    n = len(clean_idx)
    for i in range(n):
        for j in range(i + 1, n):
            if (j == i + 1 and sim[i, j] >= link_next) or sim[i, j] >= link_any:
                parent[find(j)] = find(i)
    roots = {}
    groups = pd.Series(index=meta.index, dtype=object)
    for k, idx in enumerate(clean_idx):
        groups[idx] = "C%02d" % roots.setdefault(find(k), len(roots))
    for idx in meta.index[meta.cls == "dirty"]:
        groups[idx] = "D_" + str(meta.at[idx, "a14_group"])
    return groups


def grouped_split(meta, fractions=(0.70, 0.15, 0.15), seed=0, max_group_frac=0.5):
    """Whole groups go to one split; each class is split separately so the class balance is the same in every split."""
    rng = random.Random(seed)
    out = pd.Series(index=meta.index, dtype=object)
    for cls, g in meta.groupby("cls"):
        groups = g.group.value_counts()
        order = list(groups.index)
        rng.shuffle(order)
        n = len(g)
        tgt = {"test": fractions[2] * n, "val": fractions[1] * n}
        cnt = {"test": 0, "val": 0}
        for grp in order:
            size = groups[grp]
            dest = "train"
            for s in ("test", "val"):
                if size <= max_group_frac * tgt[s] + 1 and cnt[s] + size <= tgt[s] + 1:
                    dest = s
                    cnt[s] += size
                    break
            out[g.index[g.group == grp]] = dest
    return out


def grouped_folds(meta, k=5, seed=0):
    """Grouped, class-stratified k-fold assignment (every image is tested exactly once)."""
    rng = random.Random(seed)
    fold = pd.Series(0, index=meta.index, dtype=int)
    for cls, g in meta.groupby("cls"):
        groups = g.group.value_counts()
        order = list(groups.index)
        rng.shuffle(order)
        order.sort(key=lambda x: -groups[x])                     # largest first, then fill the emptiest fold
        load = [0] * k
        for grp in order:
            f = int(np.argmin(load))
            load[f] += groups[grp]
            fold[g.index[g.group == grp]] = f
    return fold


# --------------------------------------------------------------------------------------------- transforms
def make_transforms(size=224, train=True, strength="standard"):
    import torchvision.transforms as T
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if not train:
        return T.Compose([T.Resize(int(size / 0.875)), T.CenterCrop(size)] + norm)
    if strength == "none":
        return T.Compose([T.Resize(int(size / 0.875)), T.CenterCrop(size)] + norm)
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.35, 1.0), ratio=(0.75, 1.33)),   # zoom in: close-up and partial views
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.35, 0.35, 0.25, 0.03)], p=0.8),      # sun, shade, camera colour
        T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.2),              # hand shake / focus
        T.RandomGrayscale(p=0.05),
    ] + norm + [T.RandomErasing(p=0.25, scale=(0.02, 0.15))])              # partial occlusion


class ImageSet(torch.utils.data.Dataset):
    def __init__(self, files, labels, transform, cache=None):
        self.files, self.labels, self.transform = list(files), list(labels), transform
        self.imgs = cache if cache is not None else {f: Image.open(CACHE / f).convert("RGB") for f in self.files}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.transform(self.imgs[self.files[i]]), self.labels[i]


def load_cache(files):
    return {f: Image.open(CACHE / f).convert("RGB") for f in files}
