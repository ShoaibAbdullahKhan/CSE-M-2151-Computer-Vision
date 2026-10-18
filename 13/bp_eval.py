"""Test prompts, a resumable evaluation runner, and the Part C metrics."""
import json, random, re, time
from pathlib import Path
import numpy as np
from bp_common import PARTS, SPECIES_PARTS, PLURAL, iou_dice, colour_names

# --------------------------------------------------------------------------------------------- prompts
SPECIES_SYN = {"cat": ["kitten", "kitty"], "dog": ["puppy", "pooch"], "bird": ["birdie"], "horse": ["pony"],
               "cow": ["calf", "cattle animal"], "sheep": ["lamb"]}
PART_SYN = {"head": ["face"], "nose": ["nostrils"], "muzzle": ["snout", "mouth area"], "neck": ["throat"],
            "torso": ["body", "belly"], "leg": ["limbs"], "paw": ["feet"], "beak": ["bill"],
            "foot": ["feet", "talons"], "hoof": ["feet"], "horn": ["horns"], "ear": ["ear flaps"],
            "eye": ["eyeballs"], "tail": ["tail"], "wing": ["wings"]}
INDIRECT = {
    "head": "the part of this {s} that contains its brain and face",
    "ear": "the part this {s} uses to hear",
    "eye": "the part this {s} sees with",
    "nose": "the part this {s} uses to smell",
    "muzzle": "the front part of this {s}'s face where its mouth and jaws are",
    "neck": "the part that joins this {s}'s head to its body",
    "torso": "the main trunk of this {s}'s body",
    "leg": "the limbs this {s} stands and runs on",
    "paw": "the soft padded parts this {s} walks on",
    "tail": "the part at the back end that this {s} wags or swishes",
    "beak": "the part this {s} pecks with",
    "wing": "the parts this {s} flies with",
    "foot": "the parts this {s} perches with",
    "hoof": "the hard tips of this {s}'s legs",
    "horn": "the hard pointed growths on this {s}'s head",
}
DIRECT = ["Show the {p} of this {s}.", "Identify the {p} of this {s}.", "Segment the {s}'s {p}.",
          "Can you highlight the {p} of the {s}?"]
IMPOSSIBLE = {"cat": ["wing", "beak", "hoof", "horn"], "dog": ["wing", "beak", "hoof", "horn"],
              "bird": ["horn", "hoof", "paw"], "horse": ["wing", "beak", "horn", "paw"],
              "cow": ["wing", "beak", "paw"], "sheep": ["wing", "beak", "paw"]}


def make_prompt(species, part, style, rng):
    if style == "direct":
        return rng.choice(DIRECT).format(p=PLURAL[part], s=species)
    if style == "synonym":
        return rng.choice(DIRECT).format(p=rng.choice(PART_SYN[part]), s=rng.choice(SPECIES_SYN[species]))
    if style == "indirect":
        return "Show me " + INDIRECT[part].format(s=species) + "."
    raise ValueError(style)


def build_prompts(test_items, load_anno, n_pos=3, neg_every=3, min_px=80, seed=0):
    """test_items: [(img_id, species)]. Every image gets up to n_pos prompts for annotated parts, cycling the
    three styles; every neg_every-th image also gets one impossible prompt."""
    rng = random.Random(seed)
    styles = ["direct", "synonym", "indirect"]
    rows, k = [], 0
    part_count = {p: 0 for p in PARTS}
    for j, (img_id, sp) in enumerate(test_items):
        o = load_anno(img_id)[0]
        avail = [p for p, ms in o["parts"].items() if sum(m.sum() for m in ms) >= min_px]
        avail.sort(key=lambda p: (part_count[p], rng.random()))          # balance parts over the test set
        for p in avail[:n_pos]:
            part_count[p] += 1
            st = styles[k % 3]
            k += 1
            rows.append({"img_id": img_id, "species": sp, "part": p, "style": st, "negative": False,
                         "prompt": make_prompt(sp, p, st, rng)})
        if j % neg_every == 0:
            p = rng.choice(IMPOSSIBLE[sp])
            rows.append({"img_id": img_id, "species": sp, "part": p, "style": "impossible", "negative": True,
                         "prompt": rng.choice(DIRECT).format(p=PLURAL[p], s=sp)})
    for i, r in enumerate(rows):
        r["qid"] = i
    return rows


# --------------------------------------------------------------------------------------------- runner
def run_resumable(system, rows, image_fn, out_jsonl, describe_qids=(), extra=None, log_every=25):
    """Run system(image, prompt) for every row; append one JSON line per prompt so a killed run resumes."""
    out_jsonl = Path(out_jsonl)
    done = {}
    if out_jsonl.exists():
        for l in out_jsonl.read_text().splitlines():
            d = json.loads(l)
            done[d["qid"]] = d
    masks_dir = out_jsonl.with_suffix("")
    masks_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_jsonl, "a") as f:
        for n, r in enumerate(rows):
            if r["qid"] in done:
                continue
            res = system(image_fn(r["img_id"]), r["prompt"], describe=r["qid"] in describe_qids)
            res["_qid"] = r["qid"]
            np.save(masks_dir / f"{r['qid']}.npy", np.packbits(res["mask"]))
            rec = {"qid": r["qid"], "status": res["status"], "parse": {k: v for k, v in res.get("parse", {}).items() if k != "raw"},
                   "species_pred": res.get("species"), "parts_pred": res.get("parts"), "boxes": res.get("boxes"),
                   "description": res.get("description"), "time": res.get("time"), "shape": list(res["mask"].shape)}
            if extra:
                rec.update(extra(res))
            f.write(json.dumps(rec, default=float) + "\n")
            f.flush()
            done[r["qid"]] = rec
            if log_every and len(done) % log_every == 0:
                print(f"  {len(done)}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)
    return [done[r["qid"]] for r in rows]


def load_mask(out_jsonl, qid, shape):
    bits = np.load(Path(out_jsonl).with_suffix("") / f"{qid}.npy")
    return np.unpackbits(bits)[: shape[0] * shape[1]].reshape(shape).astype(bool)


# --------------------------------------------------------------------------------------------- metrics
def gt_part_mask(obj, part, shape):
    ms = obj["parts"].get(part, [])
    m = np.zeros(obj["mask"].shape, bool)
    for x in ms:
        m |= x
    return _resize_mask(m, shape)


def _resize_mask(m, shape):
    if m.shape == tuple(shape):
        return m
    import cv2
    return cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def correct_part(pred, obj, part):
    """Which annotated part does the predicted mask match best? PASCAL-Part masks are nested (the head contains the
    ears, eyes and nose; the leg contains the paw; the torso the neck), so a pixel vote would always favour the larger
    part. The match is therefore the part with the highest IoU, which normalises by part size."""
    if not pred.any():
        return False, None
    ious = {p: iou_dice(pred, _resize_mask(np.any(ms, 0), pred.shape))[0] for p, ms in obj["parts"].items()}
    best = max(ious, key=ious.get)
    return (ious[best] > 0 and best == part), (best if ious[best] > 0 else None)


COLOUR_FAMILY = {
    "black": ["black", "dark grey", "dark", "charcoal"],
    "white": ["white", "cream", "light grey", "pale", "ivory"],
    "grey": ["grey", "gray", "dark grey", "light grey", "silver"],
    "brown": ["brown", "light brown", "dark brown", "reddish brown", "tan", "beige", "chestnut", "chocolate", "fawn"],
    "orange": ["ginger", "orange", "golden", "tan", "tawny", "rust", "caramel", "reddish brown"],
    "yellow": ["yellow", "golden", "cream", "blonde"],
    "pink": ["pink", "rosy"],
    "red": ["red", "reddish"],
}
_FAM_OF = {}
for fam, names in COLOUR_FAMILY.items():
    for n in names:
        _FAM_OF.setdefault(n, set()).add(fam)
COLOUR_WORDS = sorted({n for ns in COLOUR_FAMILY.values() for n in ns} | {"gray"}, key=len, reverse=True)
FUNCTION_RE = re.compile(r"\b(used (for|to)|helps?|help(ing)? (it|the|them)|allows?|enables?|function|serves?|"
                         r"important for|responsible for|for (hearing|seeing|vision|smell|balance|walking|flying)|"
                         r"protects?|support)", re.I)
SPECIES_TERMS = {"cat": ["cat", "kitten", "kitty", "feline"], "dog": ["dog", "puppy", "canine", "pooch"],
                 "bird": ["bird"], "horse": ["horse", "pony", "equine"], "cow": ["cow", "cattle", "calf", "bovine"],
                 "sheep": ["sheep", "lamb", "ewe"]}
PART_TERMS = {"head": ["head", "face"], "ear": ["ear"], "eye": ["eye"], "nose": ["nose", "nostril"],
              "muzzle": ["muzzle", "snout", "mouth", "jaw"], "neck": ["neck"], "torso": ["torso", "body", "belly", "chest", "back"],
              "leg": ["leg", "limb"], "paw": ["paw", "feet", "foot", "pad"], "tail": ["tail"], "beak": ["beak", "bill"],
              "wing": ["wing"], "foot": ["feet", "foot", "talon", "claw"], "hoof": ["hoof", "hooves"], "horn": ["horn"]}


def colours_in(text):
    t = text.lower()
    found = []
    for w in COLOUR_WORDS:
        if re.search(rf"\b{w}\b", t):
            found.append(w)
            t = re.sub(rf"\b{w}\b", " ", t)
    return found


def description_scores(text, species, part, img_rgb, gt_mask):
    t = (text or "").lower()
    words = re.findall(r"[a-z']+", t)
    ref = {f for n, _ in colour_names(img_rgb, gt_mask, k=3, min_share=0.15) for f in _FAM_OF.get(n, {n})}
    said = colours_in(t)
    said_f = {f for w in said for f in _FAM_OF.get(w, {w})}
    other = [s for s in SPECIES_TERMS if s != species and any(re.search(rf"\b{w}s?\b", t) for w in SPECIES_TERMS[s])]
    return {
        "mentions_part": any(w in t for w in PART_TERMS[part]),
        "mentions_species": any(re.search(rf"\b{w}", t) for w in SPECIES_TERMS[species]),
        "states_function": bool(FUNCTION_RE.search(t)),
        "colour_correct": bool(said_f & ref) if said_f else False,
        "concise": 15 <= len(words) <= 90,
        "wrong_species": bool(other),
        "n_words": len(words),
    }
