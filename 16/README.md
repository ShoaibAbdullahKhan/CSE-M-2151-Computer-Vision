# Assignment 16: RNN vs. LSTM vs. GRU video indexing of clean and dirty places

Md. Shoaib Abdullah Khan · 2010976135

29 unique videos (17 clean phone clips, 12 dirty stock clips; one byte-identical copy in `data/garbage` is detected by
MD5 and skipped) are indexed as **Clean** or **Dirty**. Frames sampled at a fixed rate are encoded by a frozen
ImageNet ResNet-50, and three recurrent heads read the feature sequence:

| model | cell | trainable parameters (hidden 128) |
|---|---|---|
| RNN | tanh recurrence | 0.28 M |
| LSTM | input / forget / output gates + cell state | 1.12 M |
| GRU | update / reset gates | 0.84 M |

Baselines: single-frame MLP and mean-pooling MLP (same features, no recurrence).

## Layout

| path | content |
|---|---|
| `code.ipynb` | the complete pipeline: dataset, preprocessing, training, evaluation, analysis, index, figures |
| `data/clean`, `data/garbage` | the videos (label = folder) |
| `annotations/dataset.csv` | one row per unique video: label, capture group, split, CV fold, scene, lighting, camera motion |
| `vi_data.py` | catalogue + duplicate check, grouped split / folds, frame decoding, image changes, ResNet-50 features |
| `vi_models.py` | RNN / LSTM / GRU indexing heads and the two baselines |
| `vi_train.py` | clip sampling and the training recipe shared by every model (resumable runs) |
| `vi_eval.py` | metrics, latency, frame-order tests, timestamp localisation, memory probe |
| `app.py` | Gradio search / retrieval interface for the video index (+ indexing a new upload) |
| `outputs/runs/<run>/` | trained weights (`best.pt`) and test predictions of each run |
| `outputs/results/*.csv` | all result tables (`comparative_table.csv` = the comparison of the three models) |
| `outputs/figures/*.png` | all figures |
| `outputs/index/` | the video index (`video_index.csv`), dirty timelines, thumbnails |
| `outputs/previews/` | small H.264 copies of the videos for the interface |

## Running

```
pip install torch torchvision scikit-learn pandas matplotlib opencv-python imageio-ffmpeg "gradio<6"
jupyter nbconvert --to notebook --execute --inplace code.ipynb     # cached runs are reused
python app.py                                                       # opens the retrieval interface
```

Decoded frames and features are cached in `outputs/frames` and `outputs/features`. A finished training run is cached in
`outputs/runs/<run>/result.json`; delete that folder to retrain it. The external test in section 9 reads the
Assignment 15 photos from `../15/data` and is skipped when they are missing.
