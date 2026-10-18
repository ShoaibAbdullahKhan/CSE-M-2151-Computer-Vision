"""Fine-tune YOLO11s-seg on PASCAL-Part animal-part crops (System A's part detector).
Run as a script (not inside the notebook) so data-loader workers work on Windows; resumes if interrupted."""
import sys, shutil
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data" / "yolo_parts" / "parts.yaml"
RUN = ROOT / "outputs" / "runs" / "parts_yolo11s_seg"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60

if __name__ == "__main__":
    last = RUN / "weights" / "last.pt"
    if last.exists():
        try:
            YOLO(str(last)).train(resume=True)          # continues an interrupted run
        except AssertionError as e:                     # "training to N epochs is finished"
            print("already finished:", e)
    else:
        w = ROOT / "weights" / "yolo11s-seg.pt"
        YOLO(str(w) if w.exists() else "yolo11s-seg.pt").train(
            data=str(DATA_YAML), epochs=EPOCHS, imgsz=512, batch=32, workers=6, seed=0,
            project=str(RUN.parent), name=RUN.name, exist_ok=True, patience=25,
            cos_lr=True, close_mosaic=10, degrees=10, plots=True, verbose=False)
    best = RUN / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, ROOT / "weights" / "parts_yolo11s_seg.pt")
        print("DONE ->", ROOT / "weights" / "parts_yolo11s_seg.pt")
