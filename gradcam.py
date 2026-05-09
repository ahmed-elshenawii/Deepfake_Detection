"""
Grad-CAM for ViT — shows which patches the model focuses on.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms


class AttentionRollout:
    def __init__(self, model, device):
        self.model      = model
        self.device     = device
        self.attentions = []
        self._register_hooks()

    def _register_hooks(self):
        encoder = self.model.vit.vit.encoder
        for layer in encoder.layer:
            layer.attention.attention.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # output[0] shape: (1, 197, 768)
        attn = output[0].detach().cpu()      # (1, 197, 768)
        attn = attn[0]                       # (197, 768)
        attn = torch.softmax(attn, dim=-1)   # (197, 768)
        attn = attn.mean(dim=-1)             # (197,)
        self.attentions.append(attn)

    @torch.no_grad()
    def __call__(self, image_tensor):
        self.attentions = []
        self.model.eval()
        image_tensor = image_tensor.to(self.device)
        _ = self.model(image_tensor)

        attn = torch.stack(self.attentions)  # (L, 197)
        attn = attn.mean(dim=0)              # (197,)
        mask = attn[1:]                      # (196,)
        n    = int(mask.shape[0] ** 0.5)
        mask = mask.reshape(n, n).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask


def visualise_attention(model, image_path, config, save_name="gradcam_result.png"):
    device  = config.DEVICE
    rollout = AttentionRollout(model, device)

    raw_img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])
    tensor = tf(raw_img).unsqueeze(0)

    mask = rollout(tensor)
    mask_resized = np.array(
        Image.fromarray((mask * 255).astype(np.uint8))
        .resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    ) / 255.0

    with torch.no_grad():
        logits = model(tensor.to(device))
        prob   = torch.softmax(logits, dim=1)[0, 1].item()
        pred   = "Real" if prob > 0.5 else "Fake"
        conf   = prob if prob > 0.5 else 1 - prob

    img_show = raw_img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    heatmap  = cm.jet(mask_resized)[..., :3]
    overlay  = 0.6 * np.array(img_show) / 255.0 + 0.4 * heatmap

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Prediction: {pred}  (confidence={conf:.2%})", fontsize=14)
    for ax, img, title in zip(
        axes,
        [np.array(img_show), mask_resized, np.clip(overlay, 0, 1)],
        ["Original", "Attention Map", "Overlay"]
    ):
        ax.imshow(img, cmap="jet" if img.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(config.PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=150)
    print(f"🔍 Attention visualisation saved → {save_path}")
    plt.close()
    return pred, conf


def visualise_batch(model, val_loader, config, n=8):
    import random
    dataset = val_loader.dataset
    indices = random.sample(range(len(dataset)), n)

    for i, idx in enumerate(indices):
        img_path, true_label = dataset.samples[idx]
        pred, conf = visualise_attention(
            model, img_path, config,
            save_name=f"gradcam_{i}_true{config.CLASSES[true_label]}.png"
        )
        print(f"  True: {config.CLASSES[true_label]}  |  Pred: {pred}  ({conf:.2%})")
