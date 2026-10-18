"""One interface for DETR, DINO, RT-DETR and Grounding DINO.

Every wrapper takes a uint8 batch (B,3,640,640, letterboxed, padding bottom/right) plus the valid (w,h) of each image
and returns boxes normalised to the *valid* image region, so all models are evaluated identically.
  * mask-based models (DETR, DINO, Grounding DINO) get a padding mask and natively predict in valid-region coords;
  * RT-DETR has no mask and predicts in canvas coords -> converted here.
"""
import sys, math
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DINO_REPO = ROOT / "third_party" / "DINO"
WEIGHTS = ROOT / "weights"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], -1)


def valid_mask(valid, size, device):
    """True on real pixels, False on padding (B,H,W)."""
    m = torch.zeros(len(valid), size, size, dtype=torch.bool, device=device)
    for i, (w, h) in enumerate(valid):
        m[i, :h, :w] = True
    return m


class Detector:
    name = "detector"
    single_class = True                          # False for COCO-pretrained (class-agnostic evaluation)

    def __init__(self, device="cuda"):
        self.device = device

    def to_input(self, imgs):
        return imgs.to(self.device, non_blocking=True).float().div_(255)

    def parameters(self):
        return self.model.parameters()

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)

    def param_groups(self, lr, lr_backbone, wd):
        bb, rest = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (bb if self.is_backbone(n) else rest).append(p)
        return [{"params": rest, "lr": lr, "weight_decay": wd}, {"params": bb, "lr": lr_backbone, "weight_decay": wd}]

    @staticmethod
    def _topk(scores, boxes, k=100):
        k = min(k, scores.shape[-1])
        s, i = scores.topk(k, dim=-1)
        return [(boxes[b, i[b]].clamp(0, 1), s[b]) for b in range(len(s))]


# --------------------------------------------------------------------------------------------- DETR
class DETR(Detector):
    name = "DETR"

    def __init__(self, pretrained_coco=False, device="cuda", ckpt="facebook/detr-resnet-50"):
        super().__init__(device)
        from transformers import DetrForObjectDetection
        if pretrained_coco:
            self.model = DetrForObjectDetection.from_pretrained(ckpt)
            self.single_class = False
        else:
            self.model = DetrForObjectDetection.from_pretrained(ckpt, num_labels=1, ignore_mismatched_sizes=True)
        self.model.to(device)

    @staticmethod
    def is_backbone(n):
        return "backbone" in n

    def _forward(self, imgs, valid, labels=None):
        x = (self.to_input(imgs) - MEAN.to(self.device)) / STD.to(self.device)
        m = valid_mask(valid, x.shape[-1], self.device).long()
        return self.model(pixel_values=x, pixel_mask=m, labels=labels)

    def loss(self, imgs, boxes, valid):
        labels = [{"class_labels": torch.zeros(len(b), dtype=torch.long, device=self.device),
                   "boxes": xyxy_to_cxcywh(b.to(self.device))} for b in boxes]
        out = self._forward(imgs, valid, labels)
        return out.loss, {k: float(v) for k, v in out.loss_dict.items()}

    @torch.no_grad()
    def predict(self, imgs, valid, k=100):
        out = self._forward(imgs, valid)
        prob = out.logits.softmax(-1)[..., :-1]                   # drop the "no object" class
        scores, labels = prob.max(-1)
        res = self._topk(scores, cxcywh_to_xyxy(out.pred_boxes), k)
        if not self.single_class:
            return [(b, s, labels[i][s.argsort(descending=True)]) for i, (b, s) in enumerate(res)]
        return res


# --------------------------------------------------------------------------------------------- RT-DETR
class RTDETR(Detector):
    name = "RT-DETR"

    def __init__(self, pretrained_coco=False, device="cuda", ckpt="PekingU/rtdetr_r50vd"):
        super().__init__(device)
        from transformers import RTDetrForObjectDetection
        if pretrained_coco:
            self.model = RTDetrForObjectDetection.from_pretrained(ckpt)
            self.single_class = False
        else:
            self.model = RTDetrForObjectDetection.from_pretrained(ckpt, num_labels=1, ignore_mismatched_sizes=True)
        self.model.to(device)

    @staticmethod
    def is_backbone(n):
        return "backbone" in n

    @staticmethod
    def _scale(valid, size, device):
        return torch.tensor([[w / size, h / size, w / size, h / size] for w, h in valid], device=device)

    def loss(self, imgs, boxes, valid):
        sc = self._scale(valid, imgs.shape[-1], self.device)
        labels = [{"class_labels": torch.zeros(len(b), dtype=torch.long, device=self.device),
                   "boxes": xyxy_to_cxcywh(b.to(self.device) * sc[i])} for i, b in enumerate(boxes)]
        out = self.model(pixel_values=self.to_input(imgs), labels=labels)
        return out.loss, {k: float(v) for k, v in out.loss_dict.items()}

    @torch.no_grad()
    def predict(self, imgs, valid, k=100):
        out = self.model(pixel_values=self.to_input(imgs))
        sc = self._scale(valid, imgs.shape[-1], self.device)
        boxes = cxcywh_to_xyxy(out.pred_boxes) / sc[:, None, :]
        prob = out.logits.sigmoid()
        scores, labels = prob.max(-1)
        res = self._topk(scores, boxes, k)
        if not self.single_class:
            return [(b, s, labels[i][s.argsort(descending=True)]) for i, (b, s) in enumerate(res)]
        return res


# --------------------------------------------------------------------------------------------- DINO (official repo)
def _dino_imports():
    if str(DINO_REPO) not in sys.path:
        sys.path.insert(0, str(DINO_REPO))
    from util.slconfig import SLConfig
    from util.misc import NestedTensor
    from models.registry import MODULE_BUILD_FUNCS
    import models.dino  # noqa: F401  (registers the builder)
    return SLConfig, NestedTensor, MODULE_BUILD_FUNCS


class DINO(Detector):
    name = "DINO"

    def __init__(self, pretrained_coco=False, device="cuda", ckpt=WEIGHTS / "dino_r50_4scale_12ep.pth"):
        super().__init__(device)
        SLConfig, self.NestedTensor, REG = _dino_imports()
        args = SLConfig.fromfile(str(DINO_REPO / "config" / "DINO" / "DINO_4scale.py"))
        args.device = device
        if not pretrained_coco:
            args.num_classes, args.dn_labelbook_size = 1, 1      # one class: "piled garbage"
        else:
            self.single_class = False
        self.args = args
        self.model, self.criterion, _ = REG.get("dino")(args)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        if not pretrained_coco:                                    # drop the 91-class COCO heads
            own = self.model.state_dict()
            sd = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        missing = self.model.load_state_dict(sd, strict=False)
        self.reinitialised = len(missing.missing_keys)
        self.model.to(device)
        self.criterion.to(device)

    @staticmethod
    def is_backbone(n):
        return "backbone.0" in n

    def _samples(self, imgs, valid):
        x = (self.to_input(imgs) - MEAN.to(self.device)) / STD.to(self.device)
        return self.NestedTensor(x, ~valid_mask(valid, x.shape[-1], self.device))

    def loss(self, imgs, boxes, valid):
        targets = [{"labels": torch.zeros(len(b), dtype=torch.long, device=self.device),
                    "boxes": xyxy_to_cxcywh(b.to(self.device))} for b in boxes]
        out = self.model(self._samples(imgs, valid), targets)
        ld = self.criterion(out, targets)
        wd = self.criterion.weight_dict
        loss = sum(ld[k] * wd[k] for k in ld if k in wd)
        return loss, {k: float(v) for k, v in ld.items() if k in ("loss_ce", "loss_bbox", "loss_giou")}

    @torch.no_grad()
    def predict(self, imgs, valid, k=100):
        out = self.model(self._samples(imgs, valid))
        prob = out["pred_logits"].sigmoid()
        scores, labels = prob.max(-1)
        res = self._topk(scores, cxcywh_to_xyxy(out["pred_boxes"]), k)
        if not self.single_class:
            return [(b, s, labels[i][s.argsort(descending=True)]) for i, (b, s) in enumerate(res)]
        return res


# --------------------------------------------------------------------------------------------- Grounding DINO
class GroundingDINO(Detector):
    name = "Grounding DINO"

    def __init__(self, ckpt="IDEA-Research/grounding-dino-tiny", prompt="a pile of garbage.", device="cuda",
                 freeze_text=True, freeze_backbone=False):
        super().__init__(device)
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
        self.proc = AutoProcessor.from_pretrained(ckpt)
        self.model = GroundingDinoForObjectDetection.from_pretrained(ckpt).to(device)
        for n, p in self.model.named_parameters():
            if (freeze_text and "text_backbone" in n) or (freeze_backbone and n.startswith("model.backbone")):
                p.requires_grad_(False)
        # transformers 4.55 bug: the two-stage *encoder* classification loss is multiplied by the number of encoder
        # tokens (~8,400) instead of the number of queries, giving loss_ce_enc ~ 5e4 that swamps everything else.
        # The loss is therefore computed without the encoder term, but with the auxiliary (per-decoder-layer)
        # losses that the original Grounding DINO training uses.
        import copy
        orig = self.model.loss_function

        def _loss(*a, **kw):
            a = list(a)
            cfg = copy.copy(a[4])
            cfg.two_stage, cfg.auxiliary_loss = False, True
            a[4] = cfg
            return orig(*a, **kw)
        self.model.loss_function = _loss
        self.set_prompt(prompt)

    def set_prompt(self, prompt):
        prompt = prompt.strip().lower()
        if not prompt.endswith("."):
            prompt += "."
        self.prompt = prompt
        t = self.proc.tokenizer([prompt], return_tensors="pt")
        self.text = {k: v.to(self.device) for k, v in t.items()}
        ids = self.text["input_ids"][0]
        special = {self.proc.tokenizer.cls_token_id, self.proc.tokenizer.sep_token_id,
                   self.proc.tokenizer.convert_tokens_to_ids(".")}
        self.token_mask = torch.tensor([int(i) not in special for i in ids], device=self.device)

    @staticmethod
    def is_backbone(n):
        return n.startswith("model.backbone")

    def _forward(self, imgs, valid, labels=None):
        x = (self.to_input(imgs) - MEAN.to(self.device)) / STD.to(self.device)
        m = valid_mask(valid, x.shape[-1], self.device).long()
        B = x.shape[0]
        txt = {k: v.expand(B, -1) for k, v in self.text.items()}
        return self.model(pixel_values=x, pixel_mask=m, labels=labels, **txt)

    def loss(self, imgs, boxes, valid):
        labels = [{"class_labels": torch.zeros(len(b), dtype=torch.long, device=self.device),
                   "boxes": xyxy_to_cxcywh(b.to(self.device))} for b in boxes]
        out = self._forward(imgs, valid, labels)
        return out.loss, {k: float(v) for k, v in out.loss_dict.items() if k in ("loss_ce", "loss_bbox", "loss_giou")}

    @torch.no_grad()
    def predict(self, imgs, valid, k=100):
        out = self._forward(imgs, valid)
        L = self.token_mask.shape[0]
        prob = out.logits[..., :L].sigmoid()
        prob = prob.masked_fill(~self.token_mask[None, None, :], 0)
        scores = prob.max(-1).values                                 # best-matching prompt token per query
        return self._topk(scores, cxcywh_to_xyxy(out.pred_boxes), k)


def build(kind, **kw):
    return {"DETR": DETR, "RT-DETR": RTDETR, "DINO": DINO, "Grounding DINO": GroundingDINO}[kind](**kw)
