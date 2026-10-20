"""
Deep Learning Model Architecture Module
Crop Disease Detection App for Farmers
Author: Khushi Yadav
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights


class CropDiseaseMobileNetV2(nn.Module):
    """
    Fine-tuned MobileNetV2 architecture for plant disease leaf classification.

    Why MobileNetV2 for Indian Agriculture?
    1. Inverted Residuals with Linear Bottlenecks reduce memory footprints dramatically.
    2. Depthwise Separable Convolutions cut computational FLOPs by 8-9x compared to standard CNNs.
    3. Small parameter count (~2.8M) allows high-speed CPU inference on farmer mobile phones (<35ms).
    """

    def __init__(self, num_classes: int = 38, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained feature extractor
        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
            self.backbone = mobilenet_v2(weights=weights)
        else:
            self.backbone = mobilenet_v2(weights=None)

        in_features = self.backbone.classifier[1].in_features  # 1280

        # Custom hierarchical classifier head designed to mitigate overfitting on PlantVillage
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),  # Swish activation for smoother gradient flow
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_feature_extractor(self):
        """Freezes all layers except the newly initialized classification head."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_top_layers(self, num_blocks: int = 4):
        """
        Unfreezes the top inverted residual blocks for progressive fine-tuning
        with a lower learning rate.
        """
        total_layers = len(self.backbone.features)
        split_idx = max(0, total_layers - num_blocks)
        for i, layer in enumerate(self.backbone.features):
            if i >= split_idx:
                for param in layer.parameters():
                    param.requires_grad = True

    def unfreeze_all(self):
        """Unfreezes the entire network."""
        for param in self.parameters():
            param.requires_grad = True

    def get_parameter_summary(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
        }


class CropDiseaseEfficientNetB0(nn.Module):
    """
    Alternative architecture for comparative ablation study.
    Uses compound coefficient scaling for depth, width, and resolution.
    """

    def __init__(self, num_classes: int = 38, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
            self.backbone = efficientnet_b0(weights=weights)
        else:
            self.backbone = efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_trained_model(
    model_path: str,
    num_classes: int = 38,
    device: str = "cpu"
) -> CropDiseaseMobileNetV2:
    """Loads model weights for inference."""
    model = CropDiseaseMobileNetV2(num_classes=num_classes, pretrained=False)
    if Path(model_path).exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
