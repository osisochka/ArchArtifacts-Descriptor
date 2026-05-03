"""Model loading and inference for SIMILIS artifact description service."""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class ArtifactModel(nn.Module):
    def __init__(self, num_names: int, num_materials: int, dropout: float = 0.3):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.head_name     = nn.Linear(in_features, num_names)
        self.head_material = nn.Linear(in_features, num_materials)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.head_name(feat), self.head_material(feat), feat


def pad_to_square(img: Image.Image) -> Image.Image:
    img.thumbnail((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    padded = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
    offset = ((IMG_SIZE - img.size[0]) // 2, (IMG_SIZE - img.size[1]) // 2)
    padded.paste(img, offset)
    return padded


TRANSFORM = T.Compose([
    T.Lambda(pad_to_square),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_auto_description(name: str, material: str,
                            conf_name: float, conf_mat: float) -> str:
    name_str = name if conf_name >= 0.6 else f"{name} (?)"
    mat_str  = material if conf_mat >= 0.6 else f"{material} (?)"
    return f"{name_str}. Материал: {mat_str}."


class InferenceEngine:
    def __init__(self, checkpoint_path: str | Path,
                 name_le_path: str | Path,
                 mat_le_path: str | Path,
                 device: str = "cpu"):
        self.device = torch.device(device)

        with open(name_le_path, "rb") as f:
            self.name_le = pickle.load(f)
        with open(mat_le_path, "rb") as f:
            self.mat_le = pickle.load(f)

        num_names = len(self.name_le.classes_)
        num_mats  = len(self.mat_le.classes_)

        self.model = ArtifactModel(num_names, num_mats).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, img: Image.Image) -> dict:
        tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(self.device)
        logits_n, logits_m, _ = self.model(tensor)
        probs_n = F.softmax(logits_n, dim=1).cpu().numpy()[0]
        probs_m = F.softmax(logits_m, dim=1).cpu().numpy()[0]

        pred_name     = self.name_le.inverse_transform([probs_n.argmax()])[0]
        pred_material = self.mat_le.inverse_transform([probs_m.argmax()])[0]
        conf_name     = float(probs_n.max())
        conf_material = float(probs_m.max())

        top3_names = [
            {"label": self.name_le.inverse_transform([i])[0], "prob": round(float(p), 4)}
            for i, p in sorted(enumerate(probs_n), key=lambda x: -x[1])[:3]
        ]
        top3_mats = [
            {"label": self.mat_le.inverse_transform([i])[0], "prob": round(float(p), 4)}
            for i, p in sorted(enumerate(probs_m), key=lambda x: -x[1])[:3]
        ]

        return {
            "pred_name":       pred_name,
            "pred_material":   pred_material,
            "conf_name":       round(conf_name, 4),
            "conf_material":   round(conf_material, 4),
            "auto_description": build_auto_description(pred_name, pred_material,
                                                        conf_name, conf_material),
            "top3_names":      top3_names,
            "top3_materials":  top3_mats,
        }
