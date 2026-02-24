"""
Ensemble Prediction Module
Combines SVM + CNN predictions using weighted fusion.
"""

import logging
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

from core.ela import compute_ela_from_pil, extract_histogram_features
from core.cnn_model import TamperCNN

logger = logging.getLogger(__name__)

# Ensemble weights
CNN_WEIGHT = 0.8
SVM_WEIGHT = 0.2
THRESHOLD  = 0.5


class EnsemblePredictor:
    """
    Loads trained SVM and CNN models and runs weighted ensemble inference.
    """

    def __init__(
        self,
        svm_path: Union[str, Path],
        cnn_path: Union[str, Path],
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Ensemble running on device: {self.device}")

        # --- Load SVM ---
        with open(svm_path, "rb") as f:
            bundle = pickle.load(f)
        self.svm     = bundle["model"]
        self.scaler  = bundle["scaler"]
        logger.info("SVM model loaded.")

        # --- Load CNN ---
        self.cnn = TamperCNN(num_classes=2)
        state = torch.load(cnn_path, map_location=self.device)
        self.cnn.load_state_dict(state)
        self.cnn.to(self.device)
        self.cnn.eval()
        logger.info("CNN model loaded.")

    def _preprocess_for_cnn(self, ela_array: np.ndarray) -> torch.Tensor:
        """Resize, normalize, and convert ELA to CNN-ready tensor."""
        from PIL import Image
        pil_ela = Image.fromarray(ela_array).resize((224, 224), Image.BILINEAR)
        arr = np.array(pil_ela, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,224,224)
        return tensor.to(self.device)

    def predict(self, pil_image: Image.Image) -> dict:
        """
        Run full inference pipeline on a PIL image.

        Args:
            pil_image: Input PIL image (any mode)

        Returns:
            dict with keys:
                label        : "Authentic" | "Tampered"
                tamper_score : float in [0, 1]
                cnn_prob     : float
                svm_prob     : float
                ela_array    : np.ndarray (for visualization)
        """
        pil_image = pil_image.convert("RGB")

        # Step 1: ELA
        ela_array = compute_ela_from_pil(pil_image)

        # Step 2: SVM prediction
        hist_feat = extract_histogram_features(ela_array).reshape(1, -1)
        hist_scaled = self.scaler.transform(hist_feat)
        svm_proba = self.svm.predict_proba(hist_scaled)[0]  # [p_auth, p_tamper]
        Ps = float(svm_proba[1])

        # Step 3: CNN prediction
        cnn_input = self._preprocess_for_cnn(ela_array)
        with torch.no_grad():
            cnn_proba = self.cnn.predict_proba(cnn_input).cpu().numpy()[0]
        Pc = float(cnn_proba[1])

        # Step 4: Ensemble fusion
        final_score = CNN_WEIGHT * Pc + SVM_WEIGHT * Ps
        label = "Tampered" if final_score >= THRESHOLD else "Authentic"

        logger.debug(f"CNN_prob={Pc:.4f} | SVM_prob={Ps:.4f} | Score={final_score:.4f} → {label}")

        return {
            "label":        label,
            "tamper_score": round(final_score, 4),
            "cnn_prob":     round(Pc, 4),
            "svm_prob":     round(Ps, 4),
            "ela_array":    ela_array,
        }
