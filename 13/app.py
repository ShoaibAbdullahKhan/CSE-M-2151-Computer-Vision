"""Interactive UI for both systems.  Run:  python app.py   (then open http://127.0.0.1:7860)"""
import os, sys, time
import numpy as np
import gradio as gr
from bp_common import ROOT, overlay, isolate, SAM2Segmenter

_SYS = {}


def get_systems(which):
    """Load lazily so the UI starts fast; both systems share one SAM 2 model."""
    if "sam" not in _SYS:
        _SYS["sam"] = SAM2Segmenter()
    if which == "A" and "A" not in _SYS:
        from system_a import SystemA
        _SYS["A"] = SystemA(sam=_SYS["sam"])
    if which == "B" and "B" not in _SYS:
        from system_b import SystemB
        _SYS["B"] = SystemB(sam=_SYS["sam"])
    return _SYS[which]


def run_one(which, image, prompt):
    if image is None or not prompt.strip():
        return None, None, "Please upload an image and type a prompt."
    t0 = time.time()
    res = get_systems(which)(image, prompt)
    dt = time.time() - t0
    img, mask = res["image"], res["mask"]
    ov = overlay(img, mask, boxes=res["boxes"])
    part = isolate(img, mask) if mask.any() else None
    q = res.get("parse", {})
    if which == "A":
        understood = f"species = **{res.get('species')}**, part = **{', '.join(res.get('parts', []) or ['-'])}** ({q.get('method')})"
    else:
        understood = f"animal = **{q.get('animal')}**, part = **{q.get('body_part')}**, possible = {q.get('possible')}"
    md = (f"**Prompt understood as:** {understood}  \n**Status:** {res['status']}  \n"
          f"**Time:** {dt:.2f} s\n\n{res.get('description', '')}")
    return ov, part, md


def build_demo():
    examples_dir = ROOT / "phone image"
    examples = [[str(examples_dir / f), p] for f, p in
                [("4.jpg", "Identify the ears of this cat."), ("2.jpg", "Show the nose of this dog."),
                 ("3.jpg", "Show me the part this dog wags."), ("5.jpg", "Show the eyes of the kitten."),
                 ("1.jpg", "Show the wings of this dog.")] if (examples_dir / f).exists()]
    with gr.Blocks(title="Prompt-guided animal body-part segmentation") as demo:
        gr.Markdown("# Prompt-guided animal body-part segmentation\n"
                    "Upload an animal photo, type what you want to see (e.g. *Show the ears of this cat*), and "
                    "compare **System A** (YOLO + SAM 2, no VLM) with **System B** (Qwen2.5-VL + Grounding DINO + SAM 2).")
        with gr.Row():
            image = gr.Image(type="pil", label="Animal image", height=360)
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Show the wings of this bird")
                with gr.Row():
                    btn_a = gr.Button("Run System A (no VLM)")
                    btn_b = gr.Button("Run System B (VLM)", variant="primary")
                    btn_ab = gr.Button("Compare both")
                gr.Examples(examples, [image, prompt])
        with gr.Row():
            with gr.Column():
                gr.Markdown("### System A — YOLO + SAM 2 + predefined description")
                ov_a = gr.Image(label="Segmentation", height=300)
                part_a = gr.Image(label="Isolated body part", height=220)
                txt_a = gr.Markdown()
            with gr.Column():
                gr.Markdown("### System B — Qwen2.5-VL + Grounding DINO + SAM 2")
                ov_b = gr.Image(label="Segmentation", height=300)
                part_b = gr.Image(label="Isolated body part", height=220)
                txt_b = gr.Markdown()
        btn_a.click(lambda i, p: run_one("A", i, p), [image, prompt], [ov_a, part_a, txt_a])
        btn_b.click(lambda i, p: run_one("B", i, p), [image, prompt], [ov_b, part_b, txt_b])
        btn_ab.click(lambda i, p: run_one("A", i, p) + run_one("B", i, p), [image, prompt],
                     [ov_a, part_a, txt_a, ov_b, part_b, txt_b])
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)))
