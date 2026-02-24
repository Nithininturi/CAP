"""
Error Level Analysis (ELA) Preprocessing Module
Computes ELA maps for image tampering detection.
"""

import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
import cv2
import logging

logger = logging.getLogger(__name__)


def compute_ela(image_path: str, quality: int = 90, amplify: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Error Level Analysis for an image.

    Args:
        image_path: Path to input image
        quality: JPEG recompression quality (default 90)
        amplify: Amplification factor for difference visualization

    Returns:
        Tuple of (original_array, ela_array) both as uint8 numpy arrays
    """
    try:
        original = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        raise

    # Recompress at given quality using in-memory buffer
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    # Compute absolute pixel difference
    diff = ImageChops.difference(original, recompressed)

    # Amplify for visualization
    diff_array = np.array(diff, dtype=np.float32)
    ela_amplified = np.clip(diff_array * amplify, 0, 255).astype(np.uint8)
    ela_image = Image.fromarray(ela_amplified)

    return np.array(original), ela_amplified


def compute_ela_from_pil(pil_image: Image.Image, quality: int = 90, amplify: float = 15.0) -> np.ndarray:
    """
    Compute ELA from a PIL Image object (for Django inference).

    Args:
        pil_image: PIL Image (RGB)
        quality: JPEG recompression quality
        amplify: Amplification factor

    Returns:
        ELA image as uint8 numpy array (H, W, 3)
    """
    pil_image = pil_image.convert("RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    diff = ImageChops.difference(pil_image, recompressed)
    diff_array = np.array(diff, dtype=np.float32)
    ela_amplified = np.clip(diff_array * amplify, 0, 255).astype(np.uint8)

    return ela_amplified


def ela_to_tensor(ela_array: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
    """
    Resize and normalize ELA array for CNN input.

    Args:
        ela_array: ELA image as uint8 numpy array
        size: Target (height, width)

    Returns:
        Normalized float32 array of shape (3, H, W)
    """
    resized = cv2.resize(ela_array, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    # HWC -> CHW
    tensor = np.transpose(normalized, (2, 0, 1))
    return tensor


def extract_histogram_features(ela_array: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Extract 256-bin histogram features from ELA image for SVM.

    Args:
        ela_array: ELA image as uint8 numpy array
        bins: Number of histogram bins

    Returns:
        Feature vector of shape (bins,) — normalized histogram
    """
    gray = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)
    hist, _ = np.histogram(gray.flatten(), bins=bins, range=(0, 256))
    # Normalize histogram
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist
