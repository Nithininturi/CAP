"""
CNN Training Script
Trains the TamperCNN on ELA images from CASIA v2.

Usage:
    python training/train_cnn.py --dataset /path/to/CASIA_v2 --output models_saved/cnn.pt
"""

import argparse
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.cnn_model import TamperCNN
from core.dataset import load_casia_paths, split_dataset, ELADatasetCNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("cnn_training.log")]
)
logger = logging.getLogger(__name__)


def evaluate_cnn(model, loader, device, split_name: str):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = (probs >= 0.5).long()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    y, yhat, yprob = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    acc  = accuracy_score(y, yhat)
    prec = precision_score(y, yhat, zero_division=0)
    rec  = recall_score(y, yhat, zero_division=0)
    f1   = f1_score(y, yhat, zero_division=0)
    auc  = roc_auc_score(y, yprob)

    logger.info(f"\n{'='*50}\n{split_name} Results")
    logger.info(f"  Accuracy : {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall   : {rec:.4f}")
    logger.info(f"  F1-Score : {f1:.4f}")
    logger.info(f"  ROC-AUC  : {auc:.4f}")
    logger.info(f"\n{classification_report(y, yhat, target_names=['Authentic','Tampered'])}")
    return acc, f1, auc


def main():
    parser = argparse.ArgumentParser(description="Train CNN for image tampering detection")
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--output",     default="models_saved/cnn.pt")
    parser.add_argument("--epochs",     type=int,   default=25)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on: {device}")

    # --- Dataset ---
    logger.info("Loading CASIA v2 dataset...")
    paths, labels = load_casia_paths(args.dataset)
    (train_p, train_l), (val_p, val_l), (test_p, test_l) = split_dataset(paths, labels)

    train_ds = ELADatasetCNN(train_p, train_l, augment=True)
    val_ds   = ELADatasetCNN(val_p,   val_l,   augment=False)
    test_ds  = ELADatasetCNN(test_p,  test_l,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # --- Model ---
    model = TamperCNN(num_classes=2, dropout=0.5).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Loss (handle class imbalance) ---
    n_auth = train_l.count(0)
    n_tamp = train_l.count(1)
    weight = torch.tensor([1.0, n_auth / max(n_tamp, 1)], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- Training loop ---
    best_val_acc = 0.0
    best_path    = args.output

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)

        scheduler.step()

        train_loss = running_loss / total
        train_acc  = correct / total
        elapsed    = time.time() - t0

        # Validation accuracy
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)
        val_acc = val_correct / val_total

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            logger.info(f"  ✓ Best model saved (val_acc={best_val_acc:.4f})")

    # --- Final evaluation ---
    logger.info(f"\nLoading best model from {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device))

    evaluate_cnn(model, val_loader,  device, "Validation")
    evaluate_cnn(model, test_loader, device, "Test")

    logger.info(f"CNN training complete. Model saved to: {best_path}")


if __name__ == "__main__":
    main()
