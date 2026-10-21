"""Backbones for clean/dirty classification: two CNNs, two Transformers and two Mamba (MambaVision) models.

All are ImageNet-1k classifiers of similar size (22-35 M parameters). Each gets a new 2-way linear head.
MambaVision is loaded from the Hugging Face hub; its selective-scan CUDA kernel is replaced by the pure-PyTorch scan in
third_party/mamba_ssm (identical maths, see that file).
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT / "third_party") not in sys.path:
    sys.path.insert(0, str(ROOT / "third_party"))

MODELS = {
    # name:            (family,        source,  checkpoint)
    "ResNet-50":       ("CNN",         "timm",  "resnet50.a1_in1k"),
    "ConvNeXt-T":      ("CNN",         "timm",  "convnext_tiny.fb_in1k"),
    "ViT-S/16":        ("Transformer", "timm",  "vit_small_patch16_224.augreg_in21k_ft_in1k"),
    "Swin-T":          ("Transformer", "timm",  "swin_tiny_patch4_window7_224.ms_in1k"),
    "MambaVision-T":   ("Mamba",       "hf",    "nvidia/MambaVision-T-1K"),
    "MambaVision-T2":  ("Mamba",       "hf",    "nvidia/MambaVision-T2-1K"),
}
FAMILY = {k: v[0] for k, v in MODELS.items()}


class MambaVisionClassifier(nn.Module):
    def __init__(self, ckpt, num_classes=2):
        super().__init__()
        import warnings
        from transformers import AutoModelForImageClassification
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hf = AutoModelForImageClassification.from_pretrained(ckpt, trust_remote_code=True)
        self.body = hf.model
        self.body.head = nn.Linear(self.body.head.in_features, num_classes)

    def forward(self, x):
        return self.body(x)

    def head_params(self):
        return list(self.body.head.parameters())


class TimmClassifier(nn.Module):
    def __init__(self, ckpt, num_classes=2, drop_path=0.1):
        super().__init__()
        import timm
        self.net = timm.create_model(ckpt, pretrained=True, num_classes=num_classes, drop_path_rate=drop_path)

    def forward(self, x):
        return self.net(x)

    def head_params(self):
        return list(self.net.get_classifier().parameters())


def build(name, num_classes=2):
    fam, src, ckpt = MODELS[name]
    return MambaVisionClassifier(ckpt, num_classes) if src == "hf" else TimmClassifier(ckpt, num_classes)


def n_params(m, trainable=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable))
