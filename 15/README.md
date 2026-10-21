# Assignment 15: Clean vs. dirty place classification (CNN vs. Transformer vs. Mamba)

Md. Shoaib Abdullah Khan · 2010976135

Six ImageNet-pretrained backbones are fine-tuned on 256 mobile-phone photos (140 clean, 116 dirty):

| family | models |
|---|---|
| CNN | ResNet-50, ConvNeXt-T |
| Transformer | ViT-S/16, Swin-T |
| Mamba | MambaVision-T, MambaVision-T2 |

## Layout

| path | content |
|---|---|
| `code.ipynb` | the complete pipeline: data, baselines, training, evaluation, figures |
| `data/clean`, `data/dirty` | the photographs (label = folder) |
| `annotations/dataset.csv` | one row per photo: label, near-duplicate group, split, CV fold, scene, capture attributes |
| `annotations/dirty_metadata_from_a14.csv` | location groups and attributes of the dirty photos (from Assignment 14) |
| `cd_data.py` | metadata, grouping, grouped split / folds, augmentations |
| `cd_models.py` | the six backbones with a new 2-way head |
| `cd_train.py` | fine-tuning recipe (identical for every model); resumable runs |
| `cd_eval.py` | metrics, compute benchmark, corruptions, crop tests, occlusion maps |
| `third_party/mamba_ssm` | pure-PyTorch `selective_scan_fn` (MambaVision needs it; the CUDA package cannot be built on Windows) |
| `outputs/runs/split_*_s0/best.pt` | trained weights (seed 0) of each model |
| `outputs/results/*.csv` | all result and comparison tables |
| `outputs/figures/*.png` | all figures |

## Running

```
pip install torch torchvision timm "transformers>=4.49,<4.56" einops scikit-learn pandas matplotlib
jupyter nbconvert --to notebook --execute --inplace code.ipynb
```

Finished training runs are cached in `outputs/runs/<run>/result.json`. Delete that folder to retrain.
