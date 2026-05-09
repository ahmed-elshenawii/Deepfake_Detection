import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score
)


# ──────────────────────────────────────────────────────────────
#  Predict on full validation set
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ──────────────────────────────────────────────────────────────
#  Full Evaluation
# ──────────────────────────────────────────────────────────────
def evaluate(model, val_loader, config):
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    labels, preds, probs = predict(model, val_loader, config.DEVICE)
    class_names = config.CLASSES   # ['fake', 'real']

    # ── 1. Classification Report ──────────────────────────────
    print("\n" + "="*60)
    print("📋  Classification Report")
    print("="*60)
    print(classification_report(labels, preds, target_names=class_names))

    auc_score = roc_auc_score(labels, probs)
    print(f"🎯  AUC-ROC Score: {auc_score:.4f}")

    # ── 2. Confusion Matrix ───────────────────────────────────
    _plot_confusion_matrix(labels, preds, class_names, config.PLOTS_DIR)

    # ── 3. ROC Curve ─────────────────────────────────────────
    _plot_roc_curve(labels, probs, config.PLOTS_DIR)

    # ── 4. Confidence Distribution ───────────────────────────
    _plot_confidence_dist(labels, probs, config.PLOTS_DIR)

    return {"auc": auc_score, "labels": labels, "preds": preds, "probs": probs}


# ──────────────────────────────────────────────────────────────
#  Plot Helpers
# ──────────────────────────────────────────────────────────────
def _plot_confusion_matrix(labels, preds, class_names, plots_dir):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title)

    plt.tight_layout()
    path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    print(f"📊 Confusion matrix saved → {path}")
    plt.close()


def _plot_roc_curve(labels, probs, plots_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    path = os.path.join(plots_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    print(f"📈 ROC curve saved → {path}")
    plt.close()


def _plot_confidence_dist(labels, probs, plots_dir):
    plt.figure(figsize=(8, 5))
    plt.hist(probs[labels == 0], bins=50, alpha=0.6, color="red",   label="Fake")
    plt.hist(probs[labels == 1], bins=50, alpha=0.6, color="green", label="Real")
    plt.axvline(0.5, color="black", linestyle="--", label="Threshold 0.5")
    plt.xlabel("Predicted Probability (Real)")
    plt.ylabel("Count")
    plt.title("Confidence Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    path = os.path.join(plots_dir, "confidence_dist.png")
    plt.savefig(path, dpi=150)
    print(f"📊 Confidence distribution saved → {path}")
    plt.close()
