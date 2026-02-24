"""
Views for the ELA Tamper Detection Django app.
Handles image upload, inference, and result display.
"""

import io
import logging
import os
import uuid

import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.http import require_http_methods
from PIL import Image

from .models import AnalysisResult

logger = logging.getLogger(__name__)

# ─── Lazy model loader ────────────────────────────────────────────────────────
_predictor = None


def get_predictor():
    """Load the ensemble predictor once and cache it."""
    global _predictor
    if _predictor is None:
        import sys
        sys.path.insert(0, str(settings.PROJECT_ROOT))
        from core.ensemble import EnsemblePredictor

        svm_path = settings.SVM_MODEL_PATH
        cnn_path = settings.CNN_MODEL_PATH

        if not os.path.exists(svm_path):
            raise FileNotFoundError(f"SVM model not found: {svm_path}")
        if not os.path.exists(cnn_path):
            raise FileNotFoundError(f"CNN model not found: {cnn_path}")

        _predictor = EnsemblePredictor(svm_path=svm_path, cnn_path=cnn_path)
        logger.info("Ensemble predictor initialized.")
    return _predictor


def _save_ela_image(ela_array: np.ndarray) -> ContentFile:
    """Convert ELA numpy array to PNG ContentFile for Django storage."""
    pil_ela = Image.fromarray(ela_array.astype(np.uint8))
    buf = io.BytesIO()
    pil_ela.save(buf, format="PNG")
    buf.seek(0)
    return ContentFile(buf.read())


# ─── Views ───────────────────────────────────────────────────────────────────

@require_http_methods(["GET"])
def index(request):
    """Landing page with upload form."""
    recent = AnalysisResult.objects.all()[:10]
    return render(request, "detector/index.html", {"recent": recent})


@require_http_methods(["POST"])
def analyze(request):
    """
    Handle image upload, run inference, save result, redirect to result page.
    """
    if "image" not in request.FILES:
        return render(request, "detector/index.html", {"error": "No image file provided."})

    uploaded_file = request.FILES["image"]

    # Validate size
    if uploaded_file.size > settings.MAX_UPLOAD_SIZE:
        return render(request, "detector/index.html",
                      {"error": "File exceeds 10 MB limit."})

    # Validate type
    content_type = uploaded_file.content_type.lower()
    if content_type not in settings.ALLOWED_IMAGE_TYPES:
        return render(request, "detector/index.html",
                      {"error": f"Unsupported file type: {content_type}. "
                                 "Please upload PNG, JPG, JPEG, WebP, GIF, or HEIC."})

    try:
        # Read PIL image
        raw = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")

        # Run ensemble prediction
        predictor = get_predictor()
        result    = predictor.predict(pil_image)

        # Save record
        record = AnalysisResult(
            prediction       = result["label"],
            tamper_score     = result["tamper_score"],
            cnn_prob         = result["cnn_prob"],
            svm_prob         = result["svm_prob"],
            original_filename= uploaded_file.name,
        )

        # Save original image
        unique_name = f"{uuid.uuid4().hex}_{uploaded_file.name}"
        record.original_image.save(unique_name, ContentFile(raw), save=False)

        # Save ELA image
        ela_content = _save_ela_image(result["ela_array"])
        ela_name    = f"ela_{uuid.uuid4().hex}.png"
        record.ela_image.save(ela_name, ela_content, save=False)

        record.save()
        logger.info(f"Analysis saved: ID={record.id}, result={result['label']}, score={result['tamper_score']}")

        return redirect("result", pk=record.id)

    except FileNotFoundError as e:
        logger.error(f"Model file missing: {e}")
        return render(request, "detector/index.html",
                      {"error": "Models not found. Please train and place svm.pkl and cnn.pt in models_saved/."})
    except Exception as e:
        logger.exception(f"Inference error: {e}")
        return render(request, "detector/index.html",
                      {"error": f"Analysis failed: {str(e)}"})


@require_http_methods(["GET"])
def result(request, pk: int):
    """Display analysis result."""
    record = get_object_or_404(AnalysisResult, pk=pk)
    return render(request, "detector/result.html", {"record": record})


@require_http_methods(["GET"])
def history(request):
    """Show all past analyses."""
    records = AnalysisResult.objects.all()
    return render(request, "detector/history.html", {"records": records})
