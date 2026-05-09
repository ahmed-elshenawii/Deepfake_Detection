"""
predict_single.py
-----------------
Run inference on a single image and print the result.

Usage:
    python predict_single.py --image path/to/face.jpg
    python predict_single.py --image path/to/face.jpg --threshold 0.6
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms

from config      import Config
from src.model   import build_model


# ──────────────────────────────────────────────────────────────
#  Load model from checkpoint
# ──────────────────────────────────────────────────────────────
def load_model(config):
    import os
    checkpoint = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    assert os.path.exists(checkpoint), (
        f"❌  Checkpoint not found at {checkpoint}\n"
        f"    Run main.py first to train the model."
    )
    model = build_model(config)
    ckpt = torch.load(checkpoint, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✅  Model loaded from {checkpoint}")
    return model


# ──────────────────────────────────────────────────────────────
#  Preprocess image
# ──────────────────────────────────────────────────────────────
def preprocess(image_path, config):
    tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])
    img = Image.open(image_path).convert("RGB")
    return img, tf(img).unsqueeze(0)   # raw PIL + tensor (1,3,H,W)


# ──────────────────────────────────────────────────────────────
#  Single prediction
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, tensor, config, threshold=0.5):
    tensor  = tensor.to(config.DEVICE)
    logits  = model(tensor)
    probs   = torch.softmax(logits, dim=1)[0]          # [fake_prob, real_prob]

    real_prob = probs[1].item()
    fake_prob = probs[0].item()
    label     = "REAL" if real_prob >= threshold else "FAKE"
    confidence = real_prob if label == "REAL" else fake_prob

    return {
        "label":      label,
        "confidence": confidence,
        "real_prob":  real_prob,
        "fake_prob":  fake_prob,
    }


# ──────────────────────────────────────────────────────────────
#  Pretty terminal output
# ──────────────────────────────────────────────────────────────
def print_result(result, image_path):
    bar_len   = 40
    real_fill = int(result["real_prob"] * bar_len)
    fake_fill = bar_len - real_fill

    print("\n" + "="*55)
    print(f"  🖼️   Image : {image_path}")
    print("="*55)

    color = "✅" if result["label"] == "REAL" else "🚨"
    print(f"  {color}  Prediction  : {result['label']}")
    print(f"  🎯  Confidence  : {result['confidence']:.2%}")
    print()
    print(f"  Real  [{('█' * real_fill).ljust(bar_len)}] {result['real_prob']:.2%}")
    print(f"  Fake  [{('█' * fake_fill).ljust(bar_len)}] {result['fake_prob']:.2%}")
    print("="*55 + "\n")


# ──────────────────────────────────────────────────────────────
#  Visual output (save image with result overlay)
# ──────────────────────────────────────────────────────────────
def save_visual(raw_img, result, image_path, config):
    import os
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.imshow(raw_img)
    ax.axis("off")

    # Color by prediction
    color = "#2ecc71" if result["label"] == "REAL" else "#e74c3c"
    label = result["label"]
    conf  = result["confidence"]

    # Title box
    ax.set_title(
        f"{label}  —  {conf:.2%} confidence",
        fontsize=16, fontweight="bold", color="white",
        bbox=dict(facecolor=color, edgecolor="none", pad=8)
    )

    # Probability bars at the bottom
    bar_ax = fig.add_axes([0.1, 0.02, 0.8, 0.06])
    bar_ax.barh([""], [result["real_prob"]], color="#2ecc71", label=f"Real {result['real_prob']:.2%}")
    bar_ax.barh([""], [result["fake_prob"]], left=[result["real_prob"]],
                color="#e74c3c", label=f"Fake {result['fake_prob']:.2%}")
    bar_ax.set_xlim(0, 1)
    bar_ax.axis("off")
    bar_ax.legend(loc="center", ncol=2, fontsize=9,
                  bbox_to_anchor=(0.5, -0.8), frameon=False)

    # Save
    base = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(config.PLOTS_DIR, f"prediction_{base}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🖼️   Visual result saved → {save_path}")


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Deepfake single-image predictor")
    parser.add_argument("--image",     required=True,        help="Path to the face image")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for REAL (default: 0.5)")
    parser.add_argument("--no-visual", action="store_true",
                        help="Skip saving the visual result image")
    args = parser.parse_args()

    config = Config()
    model  = load_model(config)

    raw_img, tensor = preprocess(args.image, config)
    result          = predict(model, tensor, config, threshold=args.threshold)

    print_result(result, args.image)

    if not args.no_visual:
        save_visual(raw_img, result, args.image, config)


if __name__ == "__main__":
    main()
