import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig


# ──────────────────────────────────────────────────────────────
#  Vision Transformer for Deepfake Detection
# ──────────────────────────────────────────────────────────────
class DeepfakeViT(nn.Module):
    """
    Pre-trained ViT-B/16  +  custom 2-class head
    Training strategy:
      Phase 1 → freeze backbone, train head only   (fast convergence)
      Phase 2 → unfreeze all, fine-tune end-to-end (best accuracy)
    """
    def __init__(self, model_name: str, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        # Load pre-trained ViT (replace its head with Identity)
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        # Replace classifier with a stronger head
        hidden_size = self.vit.config.hidden_size   # 768 for ViT-B
        self.vit.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits   # shape: (B, num_classes)

    # ── Freeze / Unfreeze helpers ──────────────────────────────
    def freeze_backbone(self):
        """Phase 1: only train the classifier head."""
        for name, param in self.vit.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("🔒 Backbone frozen — training head only.")

    def unfreeze_all(self):
        """Phase 2: fine-tune everything."""
        for param in self.vit.parameters():
            param.requires_grad = True
        print("🔓 All layers unfrozen — full fine-tuning.")

    def unfreeze_last_n_blocks(self, n: int = 4):
        """Unfreeze only the last n transformer blocks (lighter fine-tune)."""
        for param in self.vit.parameters():
            param.requires_grad = False
        # ViT blocks are in vit.vit.encoder.layer
        encoder_layers = self.vit.vit.encoder.layer
        for layer in encoder_layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        # Always unfreeze head
        for param in self.vit.classifier.parameters():
            param.requires_grad = True
        print(f"🔓 Last {n} blocks + head unfrozen.")

    def count_trainable_params(self):
        total    = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 Trainable: {trainable:,} / {total:,} params")
        return trainable


# ──────────────────────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────────────────────
def build_model(config):
    model = DeepfakeViT(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
    )
    model = model.to(config.DEVICE)
    return model
