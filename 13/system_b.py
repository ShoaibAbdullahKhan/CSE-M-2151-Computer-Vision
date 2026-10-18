"""System B -- prompt-guided body-part segmentation WITH a vision-language model.

image + prompt --Qwen2.5-VL--> {animal, body part, possible?, visible?}           (prompt understanding)
               --Qwen2.5-VL grounding + Grounding DINO--> fused part boxes         (localisation)
               --SAM 2--> mask                                                     (segmentation)
image + isolated part --Qwen2.5-VL--> context-aware description                   (description)
"""
import json, re
import numpy as np
import torch
from PIL import Image
from bp_common import (WEIGHTS, SAM2Segmenter, load_image, limit_size, isolate, box_iou, nms_boxes, Timer)

UNDERSTAND = (
    'A user asked about this image: "{prompt}"\n'
    "Reply with JSON only, using these keys:\n"
    '  "animal": the species of the main animal in the image (one or two words),\n'
    '  "body_part": the body part the user is asking for, in singular form. If the user names the part, keep the '
    "user's own word; if the user describes the part without naming it, give its usual anatomical name,\n"
    '  "possible": true if this kind of animal has this body part, otherwise false,\n'
    '  "visible": true if that body part can be seen in the image, otherwise false.')
GROUND = "Locate every {part} of the {animal} in the image, and output the bbox coordinates in JSON format."
DESCRIBE = (
    "The first image shows a {animal}; the second image is a close-up of its {part}. "
    "In two or three short sentences, describe how this {animal}'s {part} looks here (colour, shape, size or any "
    "notable detail) and state the biological function of the {part}. Write about the animal directly and do not "
    "mention images, cut-outs or segmentation. Do not use lists.")


def _pl(word):
    return word if word.endswith("s") else word + "s"


def _json(text):
    """Extract the first JSON object/array from a model reply."""
    text = text.replace("```json", "").replace("```", "")
    for pat in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pat, text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


class SystemB:
    name = "System B (Qwen2.5-VL + Grounding DINO + SAM 2)"

    def __init__(self, vlm="Qwen/Qwen2.5-VL-3B-Instruct", detector="IDEA-Research/grounding-dino-base",
                 device="cuda", localizer="fused", sam=None, max_pixels=1024 * 28 * 28):
        from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                                  AutoModelForZeroShotObjectDetection)
        self.device, self.localizer = device, localizer
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa").eval()
        for k in ("temperature", "top_p", "top_k"):                 # greedy decoding; silence sampling warnings
            setattr(self.vlm.generation_config, k, None)
        self.proc = AutoProcessor.from_pretrained(vlm, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
        self.gd_proc = AutoProcessor.from_pretrained(detector)
        self.gd = AutoModelForZeroShotObjectDetection.from_pretrained(detector).to(device).eval()
        self.sam = sam if sam is not None else SAM2Segmenter(device=device)

    # ------------------------------------------------------------------ VLM chat
    @torch.no_grad()
    def chat(self, images, text, max_new_tokens=128):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        msgs = [{"role": "user", "content": content}]
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[prompt], images=images or None, return_tensors="pt").to(self.device)
        gen = self.vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        reply = self.proc.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return reply.strip(), inputs.get("image_grid_thw")

    def says_absent(self, animal, part):
        """Text-only anatomy check by the VLM's language model: 'Does a cat have wings?'"""
        reply, _ = self.chat([], f"Does a {animal} have {_pl(part)}? Answer only yes or no.", 3)
        return reply.strip().lower().startswith("no")

    def understand(self, pil, prompt):
        reply, _ = self.chat([pil], UNDERSTAND.format(prompt=prompt.replace('"', "'")), 96)
        j = _json(reply) or {}
        if isinstance(j, list):
            j = j[0] if j and isinstance(j[0], dict) else {}
        as_bool = lambda v: v if isinstance(v, bool) else str(v).strip().lower() in ("true", "yes", "1")
        return {"animal": str(j.get("animal", "animal")).strip().lower() or "animal",
                "body_part": str(j.get("body_part", "")).strip().lower(),
                "possible": as_bool(j.get("possible", True)), "visible": as_bool(j.get("visible", True)),
                "raw": reply}

    def ground_vlm(self, pil, animal, part):
        reply, thw = self.chat([pil], GROUND.format(part=part, animal=animal), 256)
        j = _json(reply)
        if isinstance(j, dict):
            j = [j]
        h_in, w_in = int(thw[0][1]) * 14, int(thw[0][2]) * 14     # Qwen2.5-VL boxes are in resized-input pixels
        sx, sy = pil.width / w_in, pil.height / h_in
        boxes = []
        for d in (j or []):
            b = d.get("bbox_2d") if isinstance(d, dict) else None
            if b and len(b) == 4:
                x1, y1, x2, y2 = [float(v) for v in b]
                bb = [max(0, min(x1, x2) * sx), max(0, min(y1, y2) * sy), min(pil.width, max(x1, x2) * sx),
                      min(pil.height, max(y1, y2) * sy)]
                if bb[2] - bb[0] > 2 and bb[3] - bb[1] > 2:
                    boxes.append(bb)
        return boxes, reply

    @torch.no_grad()
    def ground_dino(self, pil, animal, part, thr=0.30):
        text = f"{animal} {part}."
        inputs = self.gd_proc(images=pil, text=text, return_tensors="pt").to(self.device)
        outputs = self.gd(**inputs)
        r = self.gd_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=thr, text_threshold=0.25, target_sizes=[(pil.height, pil.width)])[0]
        boxes, scores = r["boxes"].cpu().numpy().tolist(), r["scores"].cpu().numpy().tolist()
        # Grounding DINO often also returns the whole animal for "dog ear": drop any box that contains a much
        # smaller detected box (that box is the animal, not the part), and boxes covering most of the image
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        def contains(a, b):
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            return ix >= 0.9 * area(b) and area(a) > 2.5 * area(b)
        keep = [i for i, b in enumerate(boxes) if area(b) < 0.6 * pil.width * pil.height
                and not any(j != i and contains(b, boxes[j]) for j in range(len(boxes)))]
        boxes, scores = [boxes[i] for i in keep], [scores[i] for i in keep]
        k = nms_boxes(boxes, np.array(scores), 0.5) if boxes else []
        return [boxes[i] for i in k], [scores[i] for i in k]

    @staticmethod
    def fuse(vb, gb, gs, add_thr=0.45):
        """VLM boxes are trusted for *which* part; Grounding DINO refines their extent and adds missed copies."""
        if not vb:
            return [b for b, s in zip(gb, gs) if s >= 0.35]
        fused = []
        for v in vb:
            best = max(range(len(gb)), key=lambda i: box_iou(v, gb[i]), default=None)
            if best is not None and box_iou(v, gb[best]) >= 0.3:
                w = gs[best]
                fused.append([(1 - w) * a + w * b for a, b in zip(v, gb[best])])
            else:
                fused.append(v)
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in vb]
        lo, hi = 0.25 * min(areas), 4 * max(areas)
        for b, s in zip(gb, gs):
            a = (b[2] - b[0]) * (b[3] - b[1])
            if s >= add_thr and lo <= a <= hi and all(box_iou(b, f) < 0.2 for f in fused):
                fused.append(b)
        return fused

    def describe(self, pil, img, mask, animal, part):
        cut = Image.fromarray(isolate(img, mask))
        if max(cut.size) < 112:
            f = 112 / max(cut.size)
            cut = cut.resize((max(28, round(cut.width * f)), max(28, round(cut.height * f))))
        reply, _ = self.chat([pil, cut], DESCRIBE.format(animal=animal, part=part), 120)
        return reply

    # ------------------------------------------------------------------ full pipeline
    def __call__(self, image, prompt, describe=True, localizer=None, reject_policy="combined"):
        localizer = localizer or self.localizer
        tm = Timer()
        pil, _ = limit_size(load_image(image))
        img = np.asarray(pil)
        H, W = img.shape[:2]
        out = {"system": "B", "prompt": prompt, "mask": np.zeros((H, W), bool), "boxes": [], "image": img,
               "status": "ok"}
        with tm("understand"):
            q = self.understand(pil, prompt)
        out["parse"] = q
        animal, part = q["animal"], q["body_part"]
        out.update(species=animal, parts=[part])
        if not part:
            out.update(status="not_understood", description="The model could not identify a body part in the request.")
            out["time"] = tm.t
            return out
        if not q["possible"]:
            out.update(status="impossible", description=f"A {animal} does not have {_pl(part)}, so nothing was segmented.")
            out["time"] = tm.t
            return out
        # The 3B model often answers possible=true for impossible requests (e.g. wings on a cat). Two weaker signals
        # are combined: its 'visible' flag and a text-only anatomy question. Measured on VOC *train*-split prompts
        # (not the test set): rejecting when both say no loses 7.5% of real parts and catches 90% of impossible
        # requests; 'visible' alone loses 17.5% (catches 95%); the 'possible' flag alone catches only 42.5%.
        with tm("understand"):
            q["text_absent"] = self.says_absent(animal, part)
        rule = {"lenient": False, "strict": not q["visible"],
                "combined": (not q["visible"]) and q["text_absent"]}[reject_policy]
        if rule:
            out.update(status="impossible" if q["text_absent"] else "not_visible",
                       description=f"No {part} of the {animal} can be seen in this image (a {animal} may not have "
                                   f"{_pl(part)} at all), so nothing was segmented.")
            out["time"] = tm.t
            return out
        vb, gb, gs = [], [], []
        if localizer in ("vlm", "fused"):
            with tm("ground_vlm"):
                vb, out["ground_raw"] = self.ground_vlm(pil, animal, part)
        if localizer in ("dino", "fused"):
            with tm("ground_dino"):
                gb, gs = self.ground_dino(pil, animal, part)
        boxes = vb if localizer == "vlm" else (gb if localizer == "dino" else self.fuse(vb, gb, gs))
        out.update(boxes=boxes, vlm_boxes=vb, dino_boxes=gb, dino_scores=gs)
        if not boxes:
            out.update(status="not_found",
                       description=f"No {part} of the {animal} could be located in this image.")
            out["time"] = tm.t
            return out
        with tm("segment"):
            mask, _ = self.sam(img, boxes)
        out["mask"] = mask
        if describe:
            with tm("describe"):
                out["description"] = self.describe(pil, img, mask, animal, part)
        out["time"] = tm.t
        return out
