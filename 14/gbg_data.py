"""RU-campus piled-garbage dataset: annotation loading, COCO export, grouped split, augmentation, batching."""
import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data"                      # original smartphone photographs
CACHE = ROOT / "cache" / "img"           # EXIF-corrected copies, longest side 1280 px
ANN = ROOT / "annotations"
INPUT = 640                              # every model sees a 640x640 letterboxed input (same for all)
CLASS_NAME = "piled garbage"


# --------------------------------------------------------------------------------------------- loading
def cache_images(max_side=1280):
    """EXIF-correct every photo once and store a 1280-px copy (all models train/evaluate from these)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for f in sorted(RAW.glob("*.jpg")):
        im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        sizes[f.name] = im.size
        if not (CACHE / f.name).exists():
            c = im.copy()
            c.thumbnail((max_side, max_side), Image.LANCZOS)
            c.save(CACHE / f.name, quality=93)
    (ROOT / "cache" / "sizes.json").write_text(json.dumps(sizes))
    return sizes


def load_tables():
    boxes = pd.read_csv(ANN / "annotations_grid.csv", comment="#")
    meta = pd.read_csv(ANN / "image_metadata.csv", comment="#")
    sizes = json.loads((ROOT / "cache" / "sizes.json").read_text())
    meta["width"] = meta.file.map(lambda f: sizes[f][0])
    meta["height"] = meta.file.map(lambda f: sizes[f][1])
    for c in ("x1", "y1", "x2", "y2"):
        boxes[c] = boxes[c] / 10.0                       # grid units -> normalised [0, 1]
    return boxes, meta


def box_table(boxes, meta):
    """Normalised boxes + derived per-box attributes (relative area = pile size / distance proxy)."""
    b = boxes.merge(meta[["file", "width", "height"]], on="file")
    b["rel_area"] = (b.x2 - b.x1) * (b.y2 - b.y1)
    return b


# --------------------------------------------------------------------------------------------- split
def grouped_split(meta, fractions=(0.70, 0.15, 0.15), seed=7):
    """Location/near-duplicate groups never cross splits; each capture session is split separately so that
    train, val and test all contain photos from all three sessions."""
    rng = random.Random(seed)
    assign = {}
    for sess, g in meta.groupby("session"):
        groups = sorted(g.group.unique())
        rng.shuffle(groups)
        n = len(g)
        tgt_test, tgt_val = round(fractions[2] * n), round(fractions[1] * n)
        cnt = {"test": 0, "val": 0}
        for grp in groups:
            size = int((g.group == grp).sum())
            if cnt["test"] < tgt_test and cnt["test"] + size <= tgt_test + 2:
                assign[grp] = "test"; cnt["test"] += size
            elif cnt["val"] < tgt_val and cnt["val"] + size <= tgt_val + 2:
                assign[grp] = "val"; cnt["val"] += size
            else:
                assign[grp] = "train"
    return meta.group.map(assign)


def export_coco(boxes, meta, split_col="split", out_dir=ANN):
    """COCO-format annotation files (pixel coordinates of the EXIF-corrected original photos)."""
    out = {}
    for split in ["all", "train", "val", "test"]:
        m = meta if split == "all" else meta[meta[split_col] == split]
        images, anns = [], []
        for i, r in enumerate(m.itertuples(), 1):
            images.append({"id": i, "file_name": r.file, "width": int(r.width), "height": int(r.height),
                           "session": r.session, "group": r.group, "split": getattr(r, split_col)})
            for b in boxes[boxes.file == r.file].itertuples():
                x, y = b.x1 * r.width, b.y1 * r.height
                w, h = (b.x2 - b.x1) * r.width, (b.y2 - b.y1) * r.height
                anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1, "bbox": [x, y, w, h],
                             "area": w * h, "iscrowd": 0})
        coco = {"images": images, "annotations": anns,
                "categories": [{"id": 1, "name": CLASS_NAME, "supercategory": "garbage"}]}
        (out_dir / f"coco_{split}.json").write_text(json.dumps(coco))
        out[split] = coco
    return out


# --------------------------------------------------------------------------------------------- augmentation
def build_augment(strength="full"):
    """Bounding-box-aware augmentation (albumentations keeps boxes consistent with every geometric change)."""
    import albumentations as A
    if strength == "none":
        return None
    geo = [
        A.HorizontalFlip(p=0.5),
        A.BBoxSafeRandomCrop(erosion_rate=0.0, p=0.4),                       # crop without cutting any pile
        A.Affine(scale=(0.55, 1.1), translate_percent=(-0.12, 0.12), rotate=(-8, 8),
                 border_mode=cv2.BORDER_CONSTANT, fill=114, p=0.6),           # smaller / shifted / tilted piles
    ]
    photo = [
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.3, p=0.7),   # sun / shade / dusk
        A.RandomGamma(gamma_limit=(70, 140), p=0.3),
        A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=25, val_shift_limit=15, p=0.4),
        A.RandomShadow(p=0.2),                                                           # tree / building shadows
        A.OneOf([A.MotionBlur(blur_limit=(3, 9)), A.GaussianBlur(blur_limit=(3, 7))], p=0.2),  # hand shake
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),                                     # low-light sensor noise
        A.ImageCompression(quality_range=(40, 90), p=0.2),                               # WhatsApp re-compression
    ]
    return A.Compose(geo + photo, bbox_params=A.BboxParams(format="albumentations", label_fields=["labels"],
                                                           min_visibility=0.3, clip=True))


def letterbox(img, size=INPUT, fill=114):
    """Resize longest side to `size`, pad bottom/right. Returns padded image and the valid (w, h)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    out = np.full((size, size, 3), fill, np.uint8)
    out[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return out, (nw, nh)


class GarbageDataset(torch.utils.data.Dataset):
    """Returns (letterboxed uint8 HWC image, normalised xyxy boxes w.r.t. the *valid* region, valid size, file)."""

    def __init__(self, files, boxes, augment=None, size=INPUT):
        self.files, self.size, self.augment = list(files), size, augment
        self.imgs = {f: np.asarray(Image.open(CACHE / f).convert("RGB")) for f in self.files}
        self.boxes = {f: boxes[boxes.file == f][["x1", "y1", "x2", "y2"]].values.astype(np.float32) for f in self.files}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img, bx = self.imgs[f], self.boxes[f]
        if self.augment is not None:
            r = self.augment(image=img, bboxes=bx.clip(0, 1), labels=[0] * len(bx))
            img, bx = r["image"], np.asarray(r["bboxes"], np.float32).reshape(-1, 4)
        lb, valid = letterbox(img, self.size)
        return lb, bx, valid, f


def collate(batch):
    imgs = torch.from_numpy(np.stack([b[0] for b in batch])).permute(0, 3, 1, 2).contiguous()   # uint8 B3HW
    return imgs, [torch.from_numpy(b[1]) for b in batch], [b[2] for b in batch], [b[3] for b in batch]
