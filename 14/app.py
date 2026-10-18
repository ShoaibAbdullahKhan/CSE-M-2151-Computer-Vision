"""Text-prompt garbage detection on RU-campus photos with Grounding DINO.  Run:  python app.py  (http://127.0.0.1:7860)"""
import os
import numpy as np
import gradio as gr
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFont
from pathlib import Path
from gbg_data import letterbox

ROOT = Path(__file__).resolve().parent
FT = ROOT / "outputs" / "runs" / "gdino_s0" / "best.pt"
CHOICES = {"Grounding DINO base (zero-shot)": ("IDEA-Research/grounding-dino-base", None),
           "Grounding DINO tiny (zero-shot)": ("IDEA-Research/grounding-dino-tiny", None),
           "Grounding DINO tiny (fine-tuned on RU campus)": ("IDEA-Research/grounding-dino-tiny", FT)}
_MODELS = {}


def get_model(choice):
    if choice not in _MODELS:
        from gbg_models import GroundingDINO
        ckpt, weights = CHOICES[choice]
        m = GroundingDINO(ckpt=ckpt)
        if weights is not None and Path(weights).exists():
            m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False))
        m.eval()
        _MODELS[choice] = m
    return _MODELS[choice]


def draw(img, boxes, scores, colour=(255, 40, 40)):
    im = img.copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    lw = max(3, round(max(W, H) / 250))
    try:
        font = ImageFont.truetype("arial.ttf", max(16, round(max(W, H) / 45)))
    except OSError:
        font = ImageFont.load_default()
    for b, s in zip(boxes, scores):
        x1, y1, x2, y2 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
        d.rectangle([x1, y1, x2, y2], outline=colour, width=lw)
        t = f"{s:.2f}"
        tw, th = d.textbbox((0, 0), t, font=font)[2:]
        d.rectangle([x1, max(0, y1 - th - 6), x1 + tw + 8, max(th + 6, y1)], fill=colour)
        d.text((x1 + 4, max(0, y1 - th - 4)), t, fill="white", font=font)
    return im


def detect(image, prompt, choice, threshold, model=None):
    from gbg_eval import nms
    if image is None or not prompt.strip():
        return None, "Upload an image and type a prompt."
    img = ImageOps.exif_transpose(image).convert("RGB")
    img.thumbnail((1280, 1280))
    m = model or get_model(choice)
    m.set_prompt(prompt)
    lb, v = letterbox(np.asarray(img))
    boxes, scores = m.predict(torch.from_numpy(lb[None]).permute(0, 3, 1, 2).contiguous(), [v])[0]
    boxes, scores = boxes.cpu().numpy(), scores.cpu().numpy()
    keep = scores >= threshold
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes):
        k = nms(boxes, scores, 0.5)
        boxes, scores = boxes[k], scores[k]
    rows = "\n".join(f"| {i + 1} | {s:.3f} | {b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f} |" for i, (b, s) in enumerate(zip(boxes, scores)))
    md = (f"**Prompt:** `{m.prompt}` — **{len(boxes)} garbage pile(s)** with confidence ≥ {threshold:.2f}\n\n"
          + ("| # | confidence | box (x1, y1, x2, y2), normalised |\n|---|---|---|\n" + rows if len(boxes) else ""))
    return draw(img, boxes, scores), md


def build_demo():
    ex_dir = ROOT / "cache" / "img"
    examples = [[str(ex_dir / f), p] for f, p in [
        ("IMG_20261012_121408.jpg", "a pile of garbage"),
        ("garbase_054.jpg", "garbage dumped beside a campus road"),
        ("garbase_062.jpg", "a pile of plastic waste"),
        ("IMG-20261013-WA0034.jpg", "food waste"),
        ("IMG_20261012_121834.jpg", "a pile of paper waste")] if (ex_dir / f).exists()]
    with gr.Blocks(title="RU campus garbage detection") as demo:
        gr.Markdown("# Text-prompt garbage detection — Rajshahi University campus\n"
                    "Upload a campus photo, describe the garbage you are looking for, and Grounding DINO draws a box "
                    "around every matching pile with its confidence score.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Campus image", height=380)
                prompt = gr.Textbox(label="Prompt", value="a pile of garbage")
                choice = gr.Dropdown(list(CHOICES), value=list(CHOICES)[0], label="Model")
                thr = gr.Slider(0.05, 0.9, value=0.3, step=0.05, label="Confidence threshold")
                btn = gr.Button("Detect garbage", variant="primary")
                gr.Examples(examples, [image, prompt])
            with gr.Column():
                out = gr.Image(label="Detections", height=420)
                txt = gr.Markdown()
        btn.click(detect, [image, prompt, choice, thr], [out, txt])
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)))
