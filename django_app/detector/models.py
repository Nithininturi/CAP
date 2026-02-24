"""
Database models for the detector Django app.
"""

from django.db import models


class AnalysisResult(models.Model):
    """Stores the result of each image tampering analysis."""

    AUTHENTIC = "Authentic"
    TAMPERED  = "Tampered"
    LABEL_CHOICES = [(AUTHENTIC, "Authentic"), (TAMPERED, "Tampered")]

    original_image = models.ImageField(upload_to="uploads/")
    ela_image      = models.ImageField(upload_to="ela_outputs/", blank=True, null=True)
    prediction     = models.CharField(max_length=10, choices=LABEL_CHOICES)
    tamper_score   = models.FloatField(help_text="Ensemble tamper probability (0–1)")
    cnn_prob       = models.FloatField(default=0.0)
    svm_prob       = models.FloatField(default=0.0)
    analyzed_at    = models.DateTimeField(auto_now_add=True)
    original_filename = models.CharField(max_length=255, blank=True)

    class Meta:
        ordering = ["-analyzed_at"]

    def __str__(self):
        return f"{self.original_filename} → {self.prediction} ({self.tamper_score:.2%})"

    @property
    def tamper_score_pct(self):
        return round(self.tamper_score * 100, 1)
