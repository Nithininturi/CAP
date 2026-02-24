"""
Dataset loading and splitting utilities for CASIA v2 dataset.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from core.ela import compute_ela_from_pil, ela_to_tensor, extract_histogram_features

logger = logging.getLogger(__name__)


def load_casia_paths(dataset_root: str) -> Tuple[list, list]:
    """
    Load image paths and labels from CASIA v2 directory structure.

    Expected structure:
        dataset_root/
            Au/   -> authentic images
            Tp/   -> tampered images

    Returns:
        (paths, labels) where label 0 = authentic, 1 = tampered
    """
    authentic_dir = os.path.join(dataset_root, "Au")
    tampered_dir  = os.path.join(dataset_root, "Tp")

    extensions = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.tif", "*.tiff", "*.png")

    def collect(folder):
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
        return files

    authentic_paths = collect(authentic_dir)
    tampered_paths  = collect(tampered_dir)

    logger.info(f"Authentic images: {len(authentic_paths)}")
    logger.info(f"Tampered  images: {len(tampered_paths)}")

    paths  = authentic_paths + tampered_paths
    labels = [0] * len(authentic_paths) + [1] * len(tampered_paths)
    return paths, labels


def split_dataset(paths: list, labels: list, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train / val / test sets.
    """
    x_temp, x_test, y_temp, y_test = train_test_split(
        paths, labels, test_size=test_ratio, stratify=labels, random_state=seed
    )
    val_relative = val_ratio / (1.0 - test_ratio)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=seed
    )
    logger.info(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# ---------------------------------------------------------------------------
# PyTorch Dataset for CNN
# ---------------------------------------------------------------------------

class ELADatasetCNN(Dataset):
    """Returns ELA image tensors for CNN training."""

    AUGMENT_TRAIN = T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
    ])

    TRANSFORM_VAL = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
    ])

    def __init__(self, paths: list, labels: list, augment: bool = False, size: int = 224):
        self.paths   = paths
        self.labels  = labels
        self.augment = augment
        self.size    = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        label = self.labels[idx]

        try:
            pil_img = Image.open(path).convert("RGB")
            ela_arr = compute_ela_from_pil(pil_img)
            ela_resized = np.array(
                Image.fromarray(ela_arr).resize((self.size, self.size), Image.BILINEAR)
            )
        except Exception as e:
            logger.warning(f"Error loading {path}: {e} — using blank tensor")
            ela_resized = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        if self.augment:
            tensor = self.AUGMENT_TRAIN(ela_resized)
        else:
            tensor = self.TRANSFORM_VAL(ela_resized)

        return tensor, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Feature extraction for SVM (returns numpy arrays)
# ---------------------------------------------------------------------------

def extract_svm_features(paths: list, labels: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 256-bin ELA histogram features for SVM training.

    Returns:
        X: shape (N, 256)
        y: shape (N,)
    """
    X, y = [], []
    for i, (path, label) in enumerate(zip(paths, labels)):
        if i % 500 == 0:
            logger.info(f"  Extracting SVM features: {i}/{len(paths)}")
        try:
            pil_img = Image.open(path).convert("RGB")
            ela_arr = compute_ela_from_pil(pil_img)
            feat    = extract_histogram_features(ela_arr)
            X.append(feat)
            y.append(label)
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
