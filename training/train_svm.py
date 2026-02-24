"""
SVM Training Script
Trains an RBF-SVM on 256-bin ELA histogram features extracted from CASIA v2.

Usage:
    python training/train_svm.py --dataset /path/to/CASIA_v2 --output models_saved/svm.pkl
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# Allow importing from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.dataset import load_casia_paths, split_dataset, extract_svm_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("svm_training.log")]
)
logger = logging.getLogger(__name__)


def evaluate(model, scaler, X, y, split_name: str):
    X_scaled = scaler.transform(X)
    y_pred   = model.predict(X_scaled)
    y_proba  = model.predict_proba(X_scaled)[:, 1]

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)
    auc  = roc_auc_score(y, y_proba)

    logger.info(f"\n{'='*50}\n{split_name} Results")
    logger.info(f"  Accuracy : {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall   : {rec:.4f}")
    logger.info(f"  F1-Score : {f1:.4f}")
    logger.info(f"  ROC-AUC  : {auc:.4f}")
    logger.info(f"\n{classification_report(y, y_pred, target_names=['Authentic','Tampered'])}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}


def main():
    parser = argparse.ArgumentParser(description="Train SVM for image tampering detection")
    parser.add_argument("--dataset", required=True, help="Path to CASIA v2 root directory")
    parser.add_argument("--output",  default="models_saved/svm.pkl", help="Output path for saved model")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # --- Load dataset ---
    logger.info("Loading CASIA v2 dataset paths...")
    paths, labels = load_casia_paths(args.dataset)

    # --- Split ---
    (train_p, train_l), (val_p, val_l), (test_p, test_l) = split_dataset(paths, labels)

    # --- Extract features ---
    logger.info("Extracting SVM features (train)...")
    X_train, y_train = extract_svm_features(train_p, train_l)

    logger.info("Extracting SVM features (val)...")
    X_val, y_val = extract_svm_features(val_p, val_l)

    logger.info("Extracting SVM features (test)...")
    X_test, y_test = extract_svm_features(test_p, test_l)

    # --- Scale ---
    logger.info("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # --- Train SVM ---
    logger.info("Training SVM (RBF kernel, C=100)... This may take several minutes.")
    svm = SVC(kernel="rbf", C=100, gamma="scale", probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    logger.info("SVM training complete.")

    # --- Evaluate ---
    evaluate(svm, scaler, X_train, y_train, "Train")
    evaluate(svm, scaler, X_val,   y_val,   "Validation")
    evaluate(svm, scaler, X_test,  y_test,  "Test")

    # --- Save ---
    bundle = {"model": svm, "scaler": scaler}
    with open(args.output, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"SVM model saved to: {args.output}")


if __name__ == "__main__":
    main()
