# Assignment 14 — Piled-garbage detection on the Rajshahi University campus

Four transformer detectors — **DETR**, **DINO**, **RT-DETR** and **Grounding DINO** — are fine-tuned to detect
*piled garbage* in 116 smartphone photographs taken across the RU campus, compared for accuracy, speed and robustness,
and Grounding DINO is also used zero-shot with text prompts and in an interactive app.

## Contents

| path | what it is |
|---|---|
| `code.ipynb` | complete executed notebook: dataset, augmentation, training, evaluation, robustness, prompts, UI |
| `data/` | the 116 original photographs |
| `annotations/annotations_grid.csv` | bounding boxes (grid units 0–10 = normalised ×10, EXIF-corrected image) |
| `annotations/image_metadata.csv` | session, location group, composition, background, occlusion per image |
| `annotations/coco_{all,train,val,test}.json`, `split.csv` | COCO-format annotations and the train/val/test split |
| `gbg_data.py` | loading, grouped split, COCO export, box-aware augmentation, letterboxing |
| `gbg_models.py` | one interface for DETR, DINO, RT-DETR and Grounding DINO |
| `gbg_train.py` | resumable fine-tuning loop shared by all models |
| `gbg_eval.py` | COCO mAP, precision/recall, latency/FPS, corruptions |
| `app.py` | Gradio app: text-prompt garbage detection with Grounding DINO |
| `outputs/runs/<model>_s0/best.pt`, `config.json`, `history.csv` | fine-tuned weights, training configuration and curves |
| `outputs/results/*.csv`, `outputs/figures/*.png` | all quantitative results and figures |
| `third_party/DINO` | official DINO repository (2-file patch: pure-PyTorch deformable attention fallback) |

## Setup

```bash
pip install "transformers>=4.49,<4.56" timm albumentations pycocotools addict yapf "gradio<6" opencv-python
git clone https://github.com/IDEA-Research/DINO third_party/DINO
gdown 1eeAHgu-fzp28PGdIjeLe-pzGPMG2r2G_ -O weights/dino_r50_4scale_12ep.pth
jupyter nbconvert --to notebook --execute code.ipynb      # or run cell by cell
python app.py                                              # UI at http://127.0.0.1:7860
```

The DINO patch makes `models/dino/ops/functions/ms_deform_attn_func.py` import the compiled
`MultiScaleDeformableAttention` op only if available and makes `models/dino/ops/modules/ms_deform_attn.py` fall back to
the repository's own `ms_deform_attn_core_pytorch`, so no CUDA compilation is needed.

Hardware used: one NVIDIA RTX 5070 Ti (16 GB), PyTorch 2.7.1, transformers 4.55.4.
