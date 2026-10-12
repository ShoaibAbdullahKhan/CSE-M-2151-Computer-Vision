
"""
Real-time Bangladeshi banknote detection and taka counting.

Loads the YOLOv8 model fine-tuned in code.ipynb, reads frames from a camera (or a
video file), draws every detected 500/1000-taka note and reports the TOTAL amount
of money currently visible.

Usage
-----
    python detect_taka_live.py                         # default camera
    python detect_taka_live.py --source 1              # second camera
    python detect_taka_live.py --source clip.mp4       # a video file
    python detect_taka_live.py --imgsz 320 --device cpu
    python detect_taka_live.py --save out.mp4          # record the annotated output

Keys
----
    q / ESC   quit
    s         save a screenshot to outputs/live_snapshots/
    SPACE     freeze / unfreeze the frame
    h         toggle the help overlay
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
# The "_live" model is trained on the dataset built by build_live_dataset.py: small-scale
# composites, white-balance jitter and background negatives. On the webcam conditions that
# matter it is far better than the plain model (see outputs/results/live_diagnosis_*.csv):
#   note at 1.5% of frame ....... 0%    -> 81% detected
#   cool white balance .......... 56%   -> 100% correct denomination
#   false positives on no-note .. 73.3% -> 1.7% of frames
# Both score an identical 0.995 mAP@50 on the original held-out test set.
DEFAULT_WEIGHTS = ROOT / "weights" / "taka_yolov8n_live.pt"
FALLBACK_WEIGHTS = ROOT / "weights" / "taka_yolov8n.pt"

# class index -> (display name, taka value, BGR colour)
CLASS_INFO = {
    0: ("500 Taka", 500, (122, 175, 27)),      # aqua-green
    1: ("1000 Taka", 1000, (167, 58, 74)),     # violet
}


# --------------------------------------------------------------------------- utils
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                   help="path to the fine-tuned .pt model")
    p.add_argument("--source", default="0",
                   help="camera index (e.g. 0) or a path to a video file")
    p.add_argument("--imgsz", type=int, default=1000, help="inference size (320 is much faster)")
    p.add_argument("--conf", type=float, default=0.45, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--device", default=None, help="'cpu', '0', ... (default: auto)")
    p.add_argument("--smooth", type=int, default=9,
                   help="frames of temporal smoothing for the total (0 disables)")
    p.add_argument("--min-area", type=float, default=0.004,
                   help="ignore boxes smaller than this fraction of the frame")
    p.add_argument("--save", default=None, help="write the annotated video here")
    p.add_argument("--no-window", action="store_true",
                   help="do not open a window (headless; use with --save)")
    p.add_argument("--width", type=int, default=3840 , help="requested capture width")
    p.add_argument("--height", type=int, default=2160 , help="requested capture height")
    return p.parse_args()


def check_gui() -> bool:
    """
    True if this OpenCV build can open windows.

    `pip install opencv-python-headless` (often pulled in as a dependency) overwrites the
    normal build's files and silently removes highgui, so cv2.namedWindow then raises
    "The function is not implemented". Detect it up front and say exactly how to fix it.
    """
    info = cv2.getBuildInformation()
    for line in info.splitlines():
        s = line.strip()
        if s.startswith("GUI:"):
            return "NONE" not in s.upper()
    return True


def gui_hint() -> str:
    return (
        "\nThis OpenCV build has no GUI support (GUI: NONE), so no window can be opened.\n"
        "That happens when opencv-python-headless is installed alongside opencv-python -\n"
        "they share the same files and the headless one wins.\n\n"
        "Fix:\n"
        "    pip uninstall -y opencv-python-headless opencv-python opencv-contrib-python\n"
        "    pip install opencv-contrib-python\n\n"
        "Or run without a window and write the result to a file:\n"
        "    python detect_taka_live.py --source clip.mp4 --save out.mp4 --no-window\n")


def open_capture(source: str, width: int, height: int) -> cv2.VideoCapture:
    """Open a camera index or a video file, with a clear message if it fails."""
    if source.isdigit():
        idx = int(source)
        # CAP_DSHOW avoids the multi-second MSMF startup delay on Windows
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        for be in backends:
            cap = cv2.VideoCapture(idx, be)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                return cap
            cap.release()
        raise SystemExit(
            f"\nERROR: could not open camera index {idx}.\n"
            "  - Is a webcam connected and not in use by another application?\n"
            "  - Try a different index:  --source 1\n"
            "  - Or run on a video file: --source path/to/clip.mp4\n")
    path = Path(source)
    if not path.exists():
        raise SystemExit(f"\nERROR: video file not found: {path}\n")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"\nERROR: could not open video file: {path}\n")
    return cap


def draw_label(img, text, org, colour, scale=0.6, pad=5):
    """Filled label box with readable text, clamped to the image."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = int(org[0]), int(org[1])
    y = max(y, th + pad * 2)
    x = min(x, img.shape[1] - tw - pad * 2 - 1)
    x = max(x, 0)
    cv2.rectangle(img, (x, y - th - pad * 2), (x + tw + pad * 2, y), colour, -1)
    cv2.putText(img, text, (x + pad, y - pad), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)


def draw_panel(img, lines, origin=(14, 14), scale=0.72, alpha=0.78):
    """Translucent info panel in the top-left corner."""
    pad = 12
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale * s, 2)[0]
             for t, s, _ in lines]
    w = max(s[0] for s in sizes) + pad * 2
    h = sum(int(s[1] * 1.9) for s in sizes) + pad
    x0, y0 = origin
    roi = img[y0:y0 + h, x0:x0 + w]
    if roi.size:
        img[y0:y0 + h, x0:x0 + w] = cv2.addWeighted(
            roi, 1 - alpha, np.full_like(roi, 24), alpha, 0)
    y = y0 + pad
    for (text, s, colour), (tw, th) in zip(lines, sizes):
        y += int(th * 1.25)
        cv2.putText(img, text, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale * s, colour, 2, cv2.LINE_AA)
        y += int(th * 0.6)
    return h


def modal_counts(history: deque, settle: int = 3) -> Counter:
    """
    Stable reading of (c500, c1000) from a short history.

    Plain "mode over the window" absorbs single-frame flicker but keeps reporting the
    PREVIOUS scene for several frames after the notes on camera actually change --- in
    testing that was the sole cause of every wrong total, while the raw per-frame
    detections were correct. So a reading that has repeated for `settle` consecutive
    frames is adopted immediately, overriding the window mode.
    """
    if not history:
        return Counter()
    recent = list(history)[-settle:]
    if len(recent) == settle and len(set(recent)) == 1:
        key = recent[0]                       # scene has genuinely changed; trust it
    else:
        key, _ = Counter(history).most_common(1)[0]
    return Counter({0: key[0], 1: key[1]})


# --------------------------------------------------------------------------- main
def main() -> int:
    args = parse_args()

    wp = Path(args.weights)
    if not wp.exists() and wp == DEFAULT_WEIGHTS and FALLBACK_WEIGHTS.exists():
        print(f"note: {wp.name} not found, falling back to {FALLBACK_WEIGHTS.name}")
        wp = FALLBACK_WEIGHTS
    if not wp.exists():
        raise SystemExit(
            f"\nERROR: model not found at {wp}\n"
            "Run code.ipynb (exports taka_yolov8n.pt), then train_live_model.py\n"
            "(exports taka_yolov8n_live.pt, which is much better on a webcam).\n")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("\nERROR: ultralytics is not installed.  pip install ultralytics\n")

    print(f"loading {wp.name} ...")
    model = YOLO(str(wp))
    names = model.names
    print(f"classes: {names}")

    cap = open_capture(args.source, args.width, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("\nERROR: the capture device opened but returned no frames.\n")
    H, W = frame.shape[:2]
    print(f"stream: {W}x{H}   inference at {args.imgsz}px   conf>={args.conf}")

    writer = None
    if args.save:
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_out if fps_out > 1 else 25.0, (W, H))
        print(f"recording to {args.save}")

    snap_dir = ROOT / "outputs" / "live_snapshots"
    history: deque = deque(maxlen=max(1, args.smooth))
    fps_hist: deque = deque(maxlen=30)
    frozen, show_help, paused_frame = False, True, None
    show = not args.no_window
    if show and not check_gui():
        print(gui_hint())
        if not args.save:
            cap.release()
            return 2
        print("continuing headless because --save was given")
        show = False
    win = "Taka detector - press h for help, q to quit"
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(1280, W), int(min(1280, W) * H / W))

    while True:
        t0 = time.perf_counter()
        if frozen and paused_frame is not None:
            frame = paused_frame.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                print("stream ended")
                break
            paused_frame = frame.copy()

        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                            device=args.device, verbose=False)[0]

        frame_area = frame.shape[0] * frame.shape[1]
        live = Counter()
        for b in res.boxes:
            cid = int(b.cls.item())
            if cid not in CLASS_INFO:
                continue
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            if (x2 - x1) * (y2 - y1) < args.min_area * frame_area:
                continue          # reject specks that would corrupt the total
            conf = float(b.conf.item())
            label, value, colour = CLASS_INFO[cid]
            live[cid] += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
            draw_label(frame, f"{label}  {conf:.2f}", (x1, y1), colour, scale=0.62)

        history.append((live[0], live[1]))
        counts = modal_counts(history) if args.smooth > 0 else live
        total = counts[0] * 500 + counts[1] * 1000

        fps_hist.append(1.0 / max(time.perf_counter() - t0, 1e-6))
        fps = sum(fps_hist) / len(fps_hist)

        lines = [
            (f"TOTAL: {total:,} Taka", 1.5, (90, 230, 255)),
            (f"500 x {counts[0]}    1000 x {counts[1]}", 0.85, (235, 235, 235)),
            (f"{fps:5.1f} FPS   {args.imgsz}px   conf {args.conf:.2f}"
             + ("   [FROZEN]" if frozen else ""), 0.7, (170, 170, 170)),
        ]
        if show_help:
            lines.append(("q quit | s snapshot | SPACE freeze | h help", 0.62, (140, 140, 140)))
        draw_panel(frame, lines)

        if writer is not None:
            writer.write(frame)
        if not show:
            continue                      # headless: keep writing frames, no key handling
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            snap_dir.mkdir(parents=True, exist_ok=True)
            fp = snap_dir / f"taka_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(fp), frame)
            print(f"saved {fp}")
        if key == ord(" "):
            frozen = not frozen
        if key == ord("h"):
            show_help = not show_help
        if show and cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"wrote {args.save}")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
