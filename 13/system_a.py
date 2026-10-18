"""System A -- prompt-guided body-part segmentation WITHOUT a vision-language model.

prompt --(keyword lexicon + text-only sentence embedding)--> (species, part)
image  --YOLO11 (COCO)--> animal box --YOLO11-seg fine-tuned on PASCAL-Part--> part boxes --SAM 2--> mask
description = predefined text for (species, part) + colour / count / size measured from the mask
"""
import re, difflib
import numpy as np
import torch
from bp_common import (PARTS, SPECIES, SPECIES_PARTS, PLURAL, WEIGHTS, SAM2Segmenter, colour_names,
                       load_image, limit_size, Timer)

# --------------------------------------------------------------------------------------------- lexicon
SPECIES_WORDS = {
    "cat": ["cat", "kitten", "kitty", "feline", "kitties", "tomcat"],
    "dog": ["dog", "puppy", "pup", "doggy", "doggo", "canine", "hound", "pooch"],
    "bird": ["bird", "birdie", "chick", "parrot", "pigeon", "sparrow", "duck", "hen", "eagle", "seagull"],
    "horse": ["horse", "pony", "foal", "mare", "stallion", "colt"],
    "cow": ["cow", "cattle", "calf", "bull", "ox", "oxen"],
    "sheep": ["sheep", "lamb", "ram", "ewe"],
}
# noun synonyms -> canonical part. "@feet" etc. are resolved per species below.
PART_WORDS = {
    "head": ["head", "face", "skull"],
    "ear": ["ear", "earflap", "pinna"],
    "eye": ["eye", "eyeball"],
    "nose": ["nose", "nostril"],
    "muzzle": ["muzzle", "snout", "mouth", "jaw", "lip"],
    "neck": ["neck", "throat", "nape"],
    "torso": ["torso", "body", "belly", "trunk", "chest", "abdomen", "stomach", "back", "flank"],
    "leg": ["leg", "limb", "thigh"],
    "paw": ["paw", "pad"],
    "@feet": ["foot", "feet", "toe"],
    "tail": ["tail"],
    "beak": ["beak", "bill"],
    "wing": ["wing", "feather"],
    "foot": ["talon", "claw"],
    "hoof": ["hoof", "hooves"],
    "horn": ["horn", "antler"],
}
# dictionary-style definitions, used only when no keyword matches (text-only semantic fallback)
PART_GLOSS = {
    "head": "the head, the top front part of the body containing the brain, face, eyes and mouth",
    "ear": "the ear, the organ used for hearing and listening to sounds",
    "eye": "the eye, the organ used for seeing and vision",
    "nose": "the nose, the organ used for smelling and sniffing and breathing",
    "muzzle": "the muzzle or snout, the projecting front part of the face with the mouth and jaws",
    "neck": "the neck, the part connecting the head to the body",
    "torso": "the torso, the main central trunk of the body, the chest, back and belly",
    "leg": "the legs, the limbs used for standing, walking and running",
    "paw": "the paws, the soft padded feet at the ends of the legs used for walking",
    "tail": "the tail, the flexible appendage at the rear end that wags, swishes or balances",
    "beak": "the beak, the hard pointed mouth of a bird used for pecking and eating",
    "wing": "the wings, the feathered limbs a bird uses for flying",
    "foot": "the bird's feet, used for perching and gripping",
    "hoof": "the hooves, the hard horny tips of the legs",
    "horn": "the horns, the hard pointed growths on the head",
}


def _norm(word):
    w = word.lower()
    for a, b in (("hooves", "hoof"), ("feet", "feet"), ("teeth", "tooth"), ("calves", "calf"),
                 ("puppies", "puppy"), ("kitties", "kitty"), ("oxen", "ox")):
        if w == a:
            return b
    if len(w) > 3 and w.endswith("es") and w[:-2] in _ALL_WORDS:
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


_ALL_WORDS = {w for ws in list(SPECIES_WORDS.values()) + list(PART_WORDS.values()) for w in ws}


def resolve_part(key, species):
    """Map a lexicon key to a part that exists for this species (e.g. 'feet' -> paw / foot / hoof)."""
    avail = SPECIES_PARTS.get(species, PARTS)
    if key == "@feet":
        for p in ("paw", "foot", "hoof", "leg"):
            if p in avail:
                return p
    if key == "nose" and "nose" not in avail and "muzzle" in avail:
        return "muzzle"
    if key == "muzzle" and "muzzle" not in avail and "nose" in avail:
        return "nose"
    if key == "foot" and "foot" not in avail:
        return resolve_part("@feet", species)
    return key


class PromptParser:
    """Rule-based parser with a text-only sentence-embedding fallback (no image input)."""

    def __init__(self, use_embeddings=True, sim_threshold=0.35):
        self.sim_threshold = sim_threshold
        self.emb = None
        if use_embeddings:
            from sentence_transformers import SentenceTransformer
            self.emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            self.gloss_keys = list(PART_GLOSS)
            self.gloss_vec = self.emb.encode([PART_GLOSS[k] for k in self.gloss_keys], normalize_embeddings=True)

    def __call__(self, prompt):
        toks = [_norm(t) for t in re.findall(r"[a-zA-Z]+", prompt)]
        species, keys = None, []
        vocab = {w: ("S", s) for s, ws in SPECIES_WORDS.items() for w in ws}
        vocab.update({w: ("P", p) for p, ws in PART_WORDS.items() for w in ws})
        for t in toks:
            hit = vocab.get(t)
            if hit is None and len(t) >= 4:                      # tolerate small typos ("tial", "eares")
                close = difflib.get_close_matches(t, vocab.keys(), n=1, cutoff=0.85)
                hit = vocab[close[0]] if close else None
            if hit is None:
                continue
            kind, val = hit
            if kind == "S" and species is None:
                species = val
            elif kind == "P" and val not in keys:
                keys.append(val)
        method = "keyword"
        if not keys and self.emb is not None:                   # e.g. "the part it hears with"
            v = self.emb.encode([prompt], normalize_embeddings=True)[0]
            sims = self.gloss_vec @ v
            j = int(np.argmax(sims))
            if sims[j] >= self.sim_threshold:
                keys, method = [self.gloss_keys[j]], f"embedding (cos={sims[j]:.2f})"
        if not keys:
            return {"species": species, "keys": [], "method": "none", "ok": False}
        return {"species": species, "keys": keys, "method": method, "ok": True}


# --------------------------------------------------------------------------------------------- descriptions
APPEARANCE = {
    "head": "a rounded head that carries the face, eyes, ears and mouth",
    "ear": "ears that are thin flaps of skin and cartilage covered in short fur",
    "eye": "forward-facing eyes with a dark pupil inside a coloured iris",
    "nose": "a small, moist, hairless nose leather with two nostrils",
    "muzzle": "an elongated muzzle that holds the jaws, teeth and nose",
    "neck": "a muscular neck that joins the head to the body",
    "torso": "a torso that forms the main bulk of the body, covered by the coat",
    "leg": "slender, muscular legs",
    "paw": "paws with soft pads on the underside and short claws",
    "tail": "a flexible, fur-covered tail",
    "beak": "a hard, keratin-covered beak",
    "wing": "feathered wings folded along the sides of the body",
    "foot": "scaly feet with clawed toes",
    "hoof": "hard keratin hooves at the tips of the legs",
    "horn": "curved horns made of bone covered by keratin",
}
FUNCTION = {
    "head": "It houses the brain and the main sense organs, so it is the centre of perception and feeding.",
    "ear": "Ears collect sound and turn towards it; they also help with balance and show the animal's mood.",
    "eye": "Eyes provide vision, including good motion detection and sight in dim light.",
    "nose": "The nose is used for breathing and for a very sensitive sense of smell.",
    "muzzle": "The muzzle holds the teeth for biting and chewing and the nose for smelling.",
    "neck": "The neck supports and moves the head and protects the airway, food pipe and major blood vessels.",
    "torso": "The torso protects the vital organs such as the heart, lungs and stomach.",
    "leg": "Legs support the body's weight and allow walking, running and jumping.",
    "paw": "Paws cushion each step, give grip and silent movement, and help with grooming and handling objects.",
    "tail": "The tail helps with balance and is used for communication, such as signalling mood.",
    "beak": "The beak is used for feeding, preening feathers, building nests and defence.",
    "wing": "Wings generate lift for flight and are also used for display and balance.",
    "foot": "Feet are used for perching, walking and gripping food or branches.",
    "hoof": "Hooves bear the animal's weight and protect the toes on hard ground.",
    "horn": "Horns are used for defence, fighting rivals and display.",
}
SPECIES_APPEARANCE = {
    ("cat", "ear"): "triangular, upright ears that can rotate independently",
    ("dog", "ear"): "ears that may be erect or floppy depending on the breed",
    ("cat", "eye"): "large eyes with vertical slit pupils",
    ("dog", "eye"): "round eyes with a dark iris",
    ("cat", "nose"): "a small triangular pink or dark nose",
    ("dog", "nose"): "a wet, dark nose with large nostrils",
    ("cat", "tail"): "a long, slender and flexible tail",
    ("dog", "tail"): "a tail whose shape varies from straight to curled",
    ("cat", "paw"): "small rounded paws with retractable claws",
    ("dog", "paw"): "paws with thick pads and non-retractable claws",
    ("horse", "muzzle"): "a soft muzzle with large nostrils and sensitive whiskers",
}
SPECIES_FUNCTION = {
    ("dog", "tail"): "Dogs wag the tail to communicate excitement, friendliness or nervousness, and use it for balance.",
    ("cat", "tail"): "Cats use the tail for balance when jumping and climbing and to signal their mood.",
    ("dog", "nose"): "A dog's nose is extremely sensitive, with hundreds of millions of scent receptors.",
    ("cat", "ear"): "Cat ears hear very high-pitched sounds, which helps in hunting small prey.",
    ("cat", "eye"): "Cat eyes see well in very low light, which suits hunting at dawn and dusk.",
}


def describe_a(species, part, img_rgb, mask, n_inst, animal_area):
    sp = species or "animal"
    app = SPECIES_APPEARANCE.get((sp, part), APPEARANCE[part])
    fun = SPECIES_FUNCTION.get((sp, part), FUNCTION[part])
    cols = [c for c, _ in colour_names(img_rgb, mask)][:2]
    col_txt = " and ".join(cols) if cols else "of unclear colour"
    share = 100 * mask.sum() / max(animal_area, 1)
    n_txt = f"{n_inst} {part} region{'s' if n_inst > 1 else ''}"
    return (f"The {sp}'s {PLURAL[part]}: a {sp} typically has {app}. "
            f"In this image the system found {n_txt}, mainly {col_txt}, covering about {share:.1f}% of the "
            f"{sp}'s detected area. Function: {fun}")


# --------------------------------------------------------------------------------------------- system
COCO_ANIMALS = {14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow"}


class SystemA:
    name = "System A (YOLO + SAM 2, no VLM)"

    def __init__(self, part_weights=None, det_weights="yolo11m.pt", device="cuda", use_sam=True,
                 sam=None, parser=None, part_conf=0.25, imgsz=512):
        from ultralytics import YOLO
        part_weights = part_weights or (WEIGHTS / "parts_yolo11s_seg.pt")
        det = WEIGHTS / det_weights
        self.det = YOLO(str(det) if det.exists() else det_weights)
        self.parts = YOLO(str(part_weights))
        self.sam = sam if sam is not None else (SAM2Segmenter(device=device) if use_sam else None)
        self.use_sam = use_sam
        self.parser = parser or PromptParser()
        self.device, self.part_conf, self.imgsz = device, part_conf, imgsz

    def detect_animal(self, img_rgb, species):
        r = self.det(img_rgb[:, :, ::-1], conf=0.25, classes=list(COCO_ANIMALS), verbose=False, device=self.device)[0]
        dets = [(COCO_ANIMALS[int(c)], float(s), b.tolist())
                for b, s, c in zip(r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu())]
        if not dets:
            return None
        same = [d for d in dets if d[0] == species]
        pool = same if same else dets
        return max(pool, key=lambda d: d[1] * (1 + 1e-6 * (d[2][2] - d[2][0]) * (d[2][3] - d[2][1])))

    def __call__(self, image, prompt, describe=True):
        tm = Timer()
        pil, _ = limit_size(load_image(image))
        img = np.asarray(pil)
        H, W = img.shape[:2]
        out = {"system": "A", "prompt": prompt, "mask": np.zeros((H, W), bool), "boxes": [], "image": img,
               "status": "ok"}
        with tm("parse"):
            q = self.parser(prompt)
        out["parse"] = q
        with tm("detect_animal"):
            det = self.detect_animal(img, q["species"])
        # the user's word for the animal wins; the detector supplies the location (and the species if none is given)
        species = q["species"] or (det[0] if det else None)
        if q["species"] and det and det[0] != q["species"]:
            out["note"] = f"prompt says {q['species']} but the detector labelled the animal a {det[0]}"
        abox = det[2] if det else [0, 0, W, H]
        out.update(species=species, animal_box=abox)
        if not q["ok"]:
            out.update(status="not_understood", description="Sorry, the requested body part could not be recognised in the prompt.")
            out["time"] = tm.t
            return out
        parts = [resolve_part(k, species) for k in q["keys"]]
        out["parts"] = parts
        valid = [p for p in parts if species is None or p in SPECIES_PARTS.get(species, PARTS)]
        if not valid:
            out.update(status="impossible",
                       description=f"A {species} does not have {PLURAL[parts[0]]}, so nothing was segmented.")
            out["time"] = tm.t
            return out
        # ---- part detection on a padded crop around the animal
        x1, y1, x2, y2 = abox
        pw, ph = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
        cx1, cy1, cx2, cy2 = int(max(0, x1 - pw)), int(max(0, y1 - ph)), int(min(W, x2 + pw)), int(min(H, y2 + ph))
        crop = np.ascontiguousarray(img[cy1:cy2, cx1:cx2])
        with tm("detect_parts"):
            r = self.parts(crop[:, :, ::-1], conf=self.part_conf, imgsz=self.imgsz, verbose=False, device=self.device)[0]
        want = {PARTS.index(p) for p in valid}
        boxes, scores, polys, kept_cls = [], [], [], []
        for i, (b, s, c) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())):
            if int(c) in want:
                boxes.append((b + [cx1, cy1, cx1, cy1]).tolist())
                scores.append(float(s))
                kept_cls.append(PARTS[int(c)])
                polys.append(r.masks.xy[i] + [cx1, cy1] if r.masks is not None else None)
        out.update(boxes=boxes, scores=scores)
        if not boxes:
            out.update(status="not_found",
                       description=f"No {PLURAL[valid[0]]} could be found on the {species or 'animal'} in this image.")
            out["time"] = tm.t
            return out
        # ---- segmentation: SAM 2 from the part boxes (or the YOLO-seg masks for the ablation)
        with tm("segment"):
            if self.use_sam:
                mask, inst = self.sam(img, boxes)
            else:
                import cv2
                mask = np.zeros((H, W), np.uint8)
                for p in polys:
                    if p is not None and len(p) >= 3:
                        cv2.fillPoly(mask, [p.astype(np.int32)], 1)
                mask, inst = mask.astype(bool), boxes
        out["mask"] = mask
        if describe:
            with tm("describe"):
                aarea = (abox[2] - abox[0]) * (abox[3] - abox[1])
                out["description"] = " ".join(describe_a(species, p, img, mask, kept_cls.count(p), aarea)
                                              for p in valid if p in kept_cls)
        out["time"] = tm.t
        return out
