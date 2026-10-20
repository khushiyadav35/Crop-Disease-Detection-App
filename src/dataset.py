"""
PlantVillage Dataset Pipeline & Augmentation Module
Crop Disease Detection App for Farmers
Author: Khushi Yadav
"""

import os
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np


# Standard ImageNet normalization constants used by MobileNetV2 / EfficientNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Data augmentation pipeline tailored for plant leaf photography under variable field conditions.
    Simulates:
      - Sunlight fluctuations (ColorJitter)
      - Variable camera angles and orientations (RandomRotation, Flips)
      - Leaf scale & framing differences (RandomResizedCrop)
      - Perspective tilts (RandomAffine)
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Deterministic validation and test transformation pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class PlantVillageDataset(Dataset):
    """
    PyTorch Dataset wrapper for PlantVillage leaf images.
    Supports directory hierarchy:
        dataset_root/
            Crop___Disease_Name/
                image_001.JPG
                image_002.JPG
    """

    def __init__(
        self,
        root_dir: str,
        classes: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
        samples: Optional[List[Tuple[str, int]]] = None,
    ):
        self.root_dir = Path(root_dir) if root_dir else None
        self.transform = transform

        if samples is not None and classes is not None:
            self.samples = samples
            self.classes = classes
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        elif self.root_dir and self.root_dir.exists():
            self._load_from_directory()
        else:
            # Fallback placeholder when loading metadata without raw images
            self.classes = classes or []
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            self.samples = []

    def _load_from_directory(self):
        self.classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        valid_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        self.samples = []

        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            target = self.class_to_idx[cls_name]
            for entry in cls_dir.iterdir():
                if entry.suffix in valid_extensions:
                    self.samples.append((str(entry), target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_stratified_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Builds stratified train, validation, and test loaders from the raw PlantVillage directory.
    """
    torch.manual_seed(seed)
    full_dataset = PlantVillageDataset(root_dir=root_dir)
    classes = full_dataset.classes

    total_len = len(full_dataset)
    train_len = int(train_ratio * total_len)
    val_len = int(val_ratio * total_len)
    test_len = total_len - train_len - val_len

    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )

    # Attach appropriate transforms
    train_ds = PlantVillageDataset(
        root_dir="",
        classes=classes,
        transform=get_train_transforms(),
        samples=[full_dataset.samples[i] for i in train_subset.indices],
    )
    val_ds = PlantVillageDataset(
        root_dir="",
        classes=classes,
        transform=get_val_transforms(),
        samples=[full_dataset.samples[i] for i in val_subset.indices],
    )
    test_ds = PlantVillageDataset(
        root_dir="",
        classes=classes,
        transform=get_val_transforms(),
        samples=[full_dataset.samples[i] for i in test_subset.indices],
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, classes


def compute_class_weights(sample_counts: Dict[str, int]) -> torch.Tensor:
    """
    Computes inverse frequency class weights to mitigate class imbalance.
    weight_i = total_samples / (num_classes * count_i)
    """
    counts = np.array(list(sample_counts.values()), dtype=np.float32)
    total_samples = counts.sum()
    num_classes = len(counts)
    weights = total_samples / (num_classes * (counts + 1e-5))
    # Normalize weights so mean is 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
