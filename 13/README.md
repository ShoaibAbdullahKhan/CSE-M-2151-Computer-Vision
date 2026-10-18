# Assignment 13 — Prompt-guided animal body-part segmentation (with and without a VLM)

Upload an animal photo, type a request such as *"Show the ears of this cat"*, and the system segments that body part,
shows it isolated, and describes its appearance and biological function. Two systems are implemented and compared:

| | System A — no VLM | System B — with a VLM |
|---|---|---|
| Prompt understanding | keyword lexicon (synonyms, plurals, typos) + text-only MiniLM sentence embedding | **Qwen2.5-VL-3B-Instruct** reads image + prompt (JSON: animal, part, possible, visible) |
| Localisation | **YOLO11m** (COCO) finds the animal → **YOLO11s-seg fine-tuned on PASCAL-Part** finds the part | Qwen2.5-VL grounding boxes, refined by **Grounding DINO** |
| Segmentation | **SAM 2.1** from the part boxes | **SAM 2.1** from the fused boxes |
| Description | predefined text per (species, part) + colour, count and size measured on the mask | Qwen2.5-VL, given the image and the isolated part |

## Files

| file | purpose |
|---|---|
| `code.ipynb` | the complete, executed notebook (data, training, both systems, UI test, Part C evaluation) |
| `system_a.py`, `system_b.py` | the two systems (`SystemA()(image, prompt)`, `SystemB()(image, prompt)`) |
| `bp_common.py` | shared part vocabulary, SAM 2 wrapper, colour naming, metrics, drawing |
| `bp_data.py` | PASCAL-Part loading, part merging, YOLO training-set builder, test-set selection |
| `bp_eval.py` | test prompts, resumable evaluation runner, Part C metrics |
| `train_parts.py` | fine-tunes the YOLO11s-seg part detector (run as a script) |
| `app.py` | Gradio user interface with both systems side by side |
| `phone image/` | sample input photos (3 dogs, 2 cats) |
| `outputs/results/phone_prompts.csv`, `test_prompts.csv` | sample text prompts |
| `outputs/predictions/phone/` | isolated body parts produced by both systems for every phone prompt |
| `outputs/results/phone_outputs.csv` | the generated descriptions of both systems for every phone prompt |
| `outputs/results/*.csv`, `outputs/figures/*.png` | comparative results and figures |

## Running

```bash
pip install ultralytics "transformers>=4.49,<4.56" sentence-transformers "gradio<6" accelerate scipy opencv-python
python app.py            # UI at http://127.0.0.1:7860
```

The notebook downloads PASCAL-Part and VOC 2010 itself, trains the part detector, and caches every expensive step,
so re-running it continues where it stopped. Model weights (YOLO11, SAM 2.1, Qwen2.5-VL-3B, Grounding DINO) are
downloaded automatically on first use. The fine-tuned part detector is saved as `weights/parts_yolo11s_seg.pt`.
System B needs a GPU with about 10 GB of free memory (Qwen2.5-VL-3B in bfloat16).

`gradio` must stay below version 6: Gradio 6 installs `huggingface_hub` 2.x, which breaks `transformers` 4.55.
