"""Video catalogue, grouped split, frame decoding, frame perturbations and ResNet-50 frame features.

Every expensive step writes to `outputs/` and is skipped when its output already exists."""
import hashlib, json, math, random, subprocess
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FRAMES = OUT / "frames"            # decoded frames, BASE_FPS, short side 320 px, JPEG
FEATS = OUT / "features"           # ResNet-50 features per video (original + horizontally flipped frame)
PREVIEWS = OUT / "previews"        # small H.264 copies for the retrieval interface
BASE_FPS = 8                       # every sampling rate used later (0.5 ... 8 fps) is a subsample of this grid
SHORT = 320
CLASSES = ["Clean", "Dirty"]
FOLDER_LABEL = {"clean": 0, "garbage": 1, "dirty": 1}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# Capture sessions of the clean phone videos (same day = same outing; those videos must stay in the same split).
# Each stock dirty clip comes from a different source and forms its own group.
SESSIONS = {"VID_20211030": "C-2021-10-30", "VID_20220831": "C-2022-08-31", "VID_20250526": "C-2025-05-26",
            "VID_20250114": "C-2025-01-beach", "VID_20250115": "C-2025-01-beach"}

# Visual description of every clip (written after watching them). capture: handheld / vehicle / static / drone.
SCENES = {
    "VID_20211030_081246_HSR_120.mp4": ("village path, trees", "sunny", "handheld walk"),
    "VID_20211030_091821.mp4": ("playground, slide tunnel", "sunny", "handheld walk"),
    "VID_20220510_125429_HSR_120.mp4": ("campus road, motorbike", "overcast", "static"),
    "VID_20220805_092010.mp4": ("river bank, greenery", "cloudy", "handheld pan"),
    "VID_20220831_143850.mp4": ("campus field, building", "sunny", "handheld pan"),
    "VID_20220831_144134_HSR_120.mp4": ("campus field, trees", "hazy sun", "handheld pan"),
    "VID_20240218_152823.mp4": ("sandy river bank", "backlit sun", "handheld walk"),
    "VID_20240322_164935.mp4": ("rural road, fields", "low evening sun", "vehicle"),
    "VID_20250105_093313.mp4": ("courtyard, person", "fog", "handheld"),
    "VID_20250114_102416.mp4": ("sea beach, people", "sunny", "handheld walk"),
    "VID_20250115_121605.mp4": ("sea beach from motorbike", "sunny", "vehicle"),
    "VID_20250526_143925.mp4": ("football field", "overcast", "handheld pan"),
    "VID_20250526_144159.mp4": ("football field", "overcast", "handheld pan"),
    "VID_20250526_144309.mp4": ("football field", "overcast", "handheld pan"),
    "VID_20250526_144331.mp4": ("football field", "overcast", "handheld pan"),
    "VID_20260611_154737.mp4": ("campus road, puddles", "rain", "handheld pan"),
    "VID_20260728_131314.mp4": ("campus road, wet", "rain", "handheld walk"),
    "12487565_1920_1080_30fps.mp4": ("burning roadside dump, smoke", "daylight", "static"),
    "2952132-uhd_4096_2160_24fps.mp4": ("litter on woodland floor", "daylight", "slider"),
    "3186589-uhd_3840_2160_30fps.mp4": ("landfill from above", "daylight", "drone"),
    "3695935-hd_1920_1080_24fps.mp4": ("dump with sticks", "bright", "static"),
    "3999624-uhd_3840_2160_24fps.mp4": ("litter at a drain grate", "daylight", "static"),
    "4279368-hd_1920_1080_30fps.mp4": ("pile of garbage bags", "daylight", "slow pan"),
    "4445847-uhd_3840_2160_30fps.mp4": ("landfill, top-down", "daylight", "drone"),
    "4513383-hd_1920_1080_30fps.mp4": ("garbage heap then waste land", "hazy", "drone"),
    "4855723-hd_1920_1080_25fps.mp4": ("polluted canal, bamboo", "daylight", "static"),
    "vecteezy_plastic-trash-waste-moving-on-a-conveyor-at-the-garbage_5485456.mp4": ("waste on a conveyor", "indoor", "static"),
    "vecteezy_pollution-concept-garbage-pile-in-trash-dump-or-landfill_12825825.mp4": ("garbage pile close-up", "daylight", "slow pan"),
    "vecteezy_river-garbage-pollution_2015570.mov": ("garbage floating in a river", "daylight", "static"),
}


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def probe(path):
    c = cv2.VideoCapture(str(path))
    fps = c.get(cv2.CAP_PROP_FPS); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, fr = c.read(); c.release()
    h, w = fr.shape[:2] if ok else (0, 0)   # OpenCV applies the rotation flag, so this is the displayed size
    return {"fps": fps, "n_frames": n, "duration_s": n / fps if fps else 0.0, "width": w, "height": h}


def catalogue(cache=OUT / "video_catalogue.csv"):
    """One row per unique video. Byte-identical copies (same MD5) are listed but marked duplicate=True."""
    if cache.exists():
        return pd.read_csv(cache)
    rows, seen = [], {}
    for folder in sorted(p for p in DATA.iterdir() if p.is_dir()):
        for f in sorted(folder.iterdir(), key=lambda p: ("(" in p.stem, p.name)):    # "x(1).mp4" after "x.mp4"
            if f.suffix.lower() not in VIDEO_EXT:
                continue
            h = md5(f)
            r = {"file": f"{folder.name}/{f.name}", "name": f.name, "folder": folder.name,
                 "label": FOLDER_LABEL[folder.name.lower()], "md5": h, "size_MB": f.stat().st_size / 1e6,
                 "duplicate_of": seen.get(h, ""), **probe(f)}
            seen.setdefault(h, r["file"])
            rows.append(r)
    df = pd.DataFrame(rows)
    df["duplicate"] = df.duplicate_of != ""
    df["class"] = df.label.map(dict(enumerate(CLASSES)))
    df["orientation"] = np.where(df.height > df.width, "portrait", "landscape")
    df["source"] = np.where(df.label == 0, "own phone", "stock (Pexels / Vecteezy)")
    df["group"] = [SESSIONS.get(n[:12], "C-" + n[4:19]) if l == 0 else "D-" + Path(n).stem[:28]
                   for n, l in zip(df.name, df.label)]
    for k, col in enumerate(["scene", "lighting", "capture"]):
        df[col] = [SCENES.get(n, ("", "", ""))[k] for n in df.name]
    # short ids: C01.. for clean, D01.. for dirty (duplicates get none)
    df["vid"] = ""
    for lab, p in [(0, "C"), (1, "D")]:
        m = (df.label == lab) & ~df.duplicate
        df.loc[m, "vid"] = [f"{p}{i + 1:02d}" for i in range(m.sum())]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


# ------------------------------------------------------------------------------------------------ split
def grouped_split(df, seed, n_test=(4, 3), n_val=(3, 2), max_group=2):
    """Picks whole groups for test and validation until each reaches (clean, dirty) video counts.
    Only groups with <= max_group videos go to val/test, so one big session cannot fill a held-out set."""
    rng = random.Random(seed)
    split = {v: "train" for v in df.vid}
    for lab in (0, 1):
        groups = df[df.label == lab].groupby("group").vid.apply(list).to_dict()
        names = [g for g in sorted(groups) if len(groups[g]) <= max_group]
        rng.shuffle(names)
        for part, need in (("test", n_test[lab]), ("val", n_val[lab])):
            got = 0
            for g in list(names):
                if got + len(groups[g]) <= need:
                    for v in groups[g]:
                        split[v] = part
                    got += len(groups[g]); names.remove(g)
                if got == need:
                    break
    return df.vid.map(split)


def grouped_folds(df, k=5, seed=0):
    """Grouped, class-stratified k folds: groups of each class are dealt to the currently smallest fold."""
    rng = random.Random(seed)
    fold = {}
    for lab in (0, 1):
        groups = df[df.label == lab].groupby("group").vid.apply(list).to_dict()
        names = sorted(groups, key=lambda g: (-len(groups[g]), rng.random()))
        size = [0] * k
        for g in names:
            f = min(range(k), key=lambda i: (size[i], rng.random()))
            for v in groups[g]:
                fold[v] = f
            size[f] += len(groups[g])
    return df.vid.map(fold)


# ------------------------------------------------------------------------------------------------ frames
def sample_indices(n_frames, fps, rate=BASE_FPS):
    """Source-frame index nearest to every time k / rate."""
    times = np.arange(0, n_frames / fps, 1.0 / rate)
    idx = np.minimum(np.round(times * fps).astype(int), n_frames - 1)
    return times, idx


def resize_short(img, short=SHORT):
    h, w = img.shape[:2]
    s = short / min(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def extract_frames(row, rate=BASE_FPS):
    """Decodes the video sequentially and stores every frame nearest to k/rate s (resized, JPEG q=92)."""
    out = FRAMES / row["vid"]
    meta = out / "frames.json"
    if meta.exists():
        return json.loads(meta.read_text())
    out.mkdir(parents=True, exist_ok=True)
    c = cv2.VideoCapture(str(DATA / row["file"]))
    times, idx = sample_indices(row["n_frames"], row["fps"], rate)
    want = {}
    for k, i in enumerate(idx):
        want.setdefault(int(i), []).append(k)
    saved = []
    for i in range(int(idx.max()) + 1):        # grab() skips a frame without converting it; read() only when needed
        if i in want:
            ok, fr = c.read()
            if not ok:
                break
            fr = resize_short(fr)
            for k in want[i]:
                cv2.imwrite(str(out / f"{k:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved.append(k)
        elif not c.grab():
            break
    c.release()
    info = {"vid": row["vid"], "rate": rate, "times": [float(times[k]) for k in saved], "files": [f"{k:05d}.jpg" for k in saved]}
    meta.write_text(json.dumps(info))
    return info


def load_frames(vid):
    info = json.loads((FRAMES / vid / "frames.json").read_text())
    return [cv2.cvtColor(cv2.imread(str(FRAMES / vid / f)), cv2.COLOR_BGR2RGB) for f in info["files"]], np.array(info["times"])


def make_preview(row, long_side=480):
    """Browser-playable H.264 copy (no audio) for the retrieval interface."""
    import imageio_ffmpeg
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    out = PREVIEWS / f"{row['vid']}.mp4"
    if not out.exists():
        vf = f"scale='if(gt(iw,ih),{long_side},-2)':'if(gt(iw,ih),-2,{long_side})'"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(DATA / row["file"]),
                        "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
                        "-an", "-movflags", "+faststart", str(out)], check=True)
    return out


# ------------------------------------------------------------------------------------------------ perturbations
def _clip(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def perturb_sequence(frames, kind, seed=0):
    """Applies one appearance / camera / environment change to a list of RGB uint8 frames.
    Temporal ones (flicker, shake) draw a new random value per frame."""
    rng = np.random.default_rng(seed)
    out = []
    h, w = frames[0].shape[:2]
    if kind == "rain":
        n = int(0.004 * h * w)
        layers = []
        for _ in range(len(frames)):
            m = np.zeros((h, w), np.float32)
            xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
            for x, y in zip(xs, ys):
                cv2.line(m, (int(x), int(y)), (int(x) - 4, int(y) + 18), 1.0, 1)
            layers.append(cv2.GaussianBlur(m, (3, 3), 0)[..., None])
    for t, f in enumerate(frames):
        x = f.astype(np.float32)
        if kind == "dark":            # dusk / poor light: gamma + exposure drop + sensor noise
            x = 255 * (x / 255) ** 1.8 * 0.45 + rng.normal(0, 6, x.shape)
        elif kind == "overexposed":
            x = x * 1.8 + 30
        elif kind == "low_contrast":
            x = (x - x.mean()) * 0.4 + x.mean()
        elif kind == "warm_cast":     # wrong white balance
            x = x * np.array([1.25, 1.0, 0.7])
        elif kind == "flicker":       # brightness changing from frame to frame
            x = x * rng.uniform(0.45, 1.55)
        elif kind == "shake":         # hand-held jitter: random shift / rotation / zoom + motion blur, per frame
            ang = rng.uniform(-8, 8); s = rng.uniform(1.05, 1.2)
            M = cv2.getRotationMatrix2D((w / 2 + rng.uniform(-0.08, 0.08) * w, h / 2 + rng.uniform(-0.08, 0.08) * h), ang, s)
            x = cv2.warpAffine(x, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            L = int(rng.integers(7, 17)); k = np.zeros((L, L), np.float32)
            th = rng.uniform(0, np.pi); c0 = (L - 1) / 2
            cv2.line(k, (int(c0 - c0 * np.cos(th)), int(c0 - c0 * np.sin(th))), (int(c0 + c0 * np.cos(th)), int(c0 + c0 * np.sin(th))), 1, 1)
            x = cv2.filter2D(x, -1, k / k.sum())
        elif kind == "defocus":
            x = cv2.GaussianBlur(x, (0, 0), 3.0)
        elif kind == "fog":           # haze thicker towards the top of the frame
            a = np.linspace(0.65, 0.35, h, dtype=np.float32)[:, None, None]
            x = x * (1 - a) + 215 * a
        elif kind == "rain":
            x = x * 0.8 * (1 - 0.7 * layers[t]) + 220 * 0.7 * layers[t]
            x = cv2.GaussianBlur(x, (0, 0), 0.8)
        elif kind == "noise":
            x = x + rng.normal(0, 25, x.shape)
        elif kind == "jpeg":
            ok, b = cv2.imencode(".jpg", _clip(x), [cv2.IMWRITE_JPEG_QUALITY, 8])
            x = cv2.imdecode(b, cv2.IMREAD_UNCHANGED).astype(np.float32)
        elif kind != "none":
            raise ValueError(kind)
        out.append(_clip(x))
    return out


PERTURBATIONS = {"lighting": ["dark", "overexposed", "low_contrast", "warm_cast", "flicker"],
                 "camera": ["shake", "defocus"],
                 "environment": ["fog", "rain", "noise", "jpeg"]}


# ------------------------------------------------------------------------------------------------ features
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(frames, size=224, resize=256):
    """Resize the short side to 256, centre-crop 224x224 (removes portrait / landscape shape), ImageNet normalise."""
    xs = []
    for f in frames:
        f = resize_short(f, resize)
        h, w = f.shape[:2]
        y0, x0 = (h - size) // 2, (w - size) // 2
        f = f[y0:y0 + size, x0:x0 + size].astype(np.float32) / 255.0
        xs.append(((f - _MEAN) / _STD).transpose(2, 0, 1))
    return torch.from_numpy(np.stack(xs))


_BACKBONE = None


def backbone():
    """ImageNet ResNet-50 without its classifier -> 2048-d global-average-pooled feature per frame."""
    global _BACKBONE
    if _BACKBONE is None:
        import torchvision
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
        _BACKBONE = m.eval().cuda()
    return _BACKBONE


@torch.no_grad()
def frame_features(frames, bs=64, flip=False):
    net = backbone()
    out = []
    for i in range(0, len(frames), bs):
        x = preprocess(frames[i:i + bs]).cuda()
        if flip:
            x = x.flip(-1)
        with torch.autocast("cuda", torch.bfloat16):
            out.append(net(x).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 2048), np.float32)


def video_features(vid, kind="none", subdir="resnet50"):
    """(N, 2048) features of the BASE_FPS frames (+ flipped copy for kind='none'); cached per video and perturbation."""
    path = FEATS / subdir / (f"{vid}.npz" if kind == "none" else f"{vid}__{kind}.npz")
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    path.parent.mkdir(parents=True, exist_ok=True)
    frames, times = load_frames(vid)
    if kind != "none":
        frames = perturb_sequence(frames, kind, seed=int(hashlib.md5(f"{vid}{kind}".encode()).hexdigest()[:8], 16))
    d = {"f": frame_features(frames).astype(np.float16), "t": times.astype(np.float32)}
    if kind == "none":
        d["ff"] = frame_features(frames, flip=True).astype(np.float16)
    np.savez(path, **d)
    return d


# ------------------------------------------------------------------------------------------------ external photos
def pseudo_video(img, n=16, seed=0):
    """Turns a still photo into an n-frame 'video' with a smooth simulated camera move (pan + zoom between two
    random crops covering 60-90 % of the side), like a slow hand-held sweep."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]

    def crop():
        s = rng.uniform(0.6, 0.9)
        ch, cw = int(h * s), int(w * s)
        return np.array([rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1), ch, cw], float)
    a, b = crop(), crop()
    out = []
    for k in np.linspace(0, 1, n):
        y, x, ch, cw = (a + (b - a) * k).astype(int)
        out.append(resize_short(img[y:y + ch, x:x + cw]))
    return out


def external_features(path, n=16, subdir="external"):
    """ResNet-50 features of the pseudo-video made from one photo (EXIF orientation applied); cached."""
    from PIL import ImageOps
    out = FEATS / subdir / (Path(path).parent.name + "__" + Path(path).stem + ".npz")
    if out.exists():
        return np.load(out)["f"]
    out.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(path); im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((1280, 1280))
    frames = pseudo_video(np.asarray(im), n, seed=int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16))
    f = frame_features(frames).astype(np.float16)
    np.savez(out, f=f)
    return f


# ------------------------------------------------------------------------------------------------ low-level cues
def frame_stats(frames):
    """Per-frame brightness, contrast, saturation, green / blue-sky fractions and motion (mean abs. difference to the
    previous sampled frame) -- used to describe videos and as a simple non-deep baseline."""
    rows, prev = [], None
    for f in frames:
        small = cv2.resize(f, (160, 160 * f.shape[0] // f.shape[1]), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32)
        g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hue, sat, val = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
        green = ((hue > 60) & (hue < 170) & (sat > 0.2)).mean()
        sky = ((hue > 180) & (hue < 260) & (val > 0.5)).mean() + ((sat < 0.12) & (val > 0.8)).mean()
        motion = np.abs(g - prev).mean() / 255 if prev is not None and prev.shape == g.shape else np.nan
        edges = cv2.Canny(g.astype(np.uint8), 80, 160).mean() / 255
        rows.append({"brightness": val.mean(), "contrast": g.std() / 255, "saturation": sat.mean(),
                     "green": green, "sky": sky, "edges": edges, "motion": motion})
        prev = g
    return pd.DataFrame(rows)
