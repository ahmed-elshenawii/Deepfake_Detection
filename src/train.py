import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, checkpoint_path="best_model.pt"):
        self.patience         = patience
        self.min_delta        = min_delta
        self.checkpoint_path  = checkpoint_path
        self.best_score       = None
        self.counter          = 0
        self.early_stop       = False

    def __call__(self, val_loss, model, optimizer, scheduler, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(model, optimizer, scheduler, epoch)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"   ⏳ EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(model, optimizer, scheduler, epoch)
            self.counter = 0

    def _save(self, model, optimizer, scheduler, epoch):
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_score":           self.best_score,
            "counter":              self.counter,
        }, self.checkpoint_path)
        print(f"   💾 Best model saved → {self.checkpoint_path}")


def mixup_data(x, y, alpha=0.2, device="cuda"):
    """MixUp augmentation — helps regularize the model."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ──────────────────────────────────────────────────────────────
#  One Epoch
# ──────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, phase="train", use_mixup=False):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train and use_mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels, device=device)
                logits = model(images)
                loss   = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                logits = model(images)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs       = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

            if batch_idx % 50 == 0:
                print(f"   [{batch_idx:>4}/{len(loader)}]  loss={loss.item():.4f}")

    avg_loss = total_loss / total
    acc      = correct / total
    f1       = f1_score(all_labels, np.array(all_probs) > 0.5, average="binary")
    auc      = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, f1, auc


# ──────────────────────────────────────────────────────────────
#  Main Training Function
# ──────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader, config):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR,      exist_ok=True)

    # ── تحقق من الـ permissions قبل ما يبدأ ──
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    device    = config.DEVICE

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  [],
               "train_f1":   [], "val_f1":   [],
               "train_auc":  [], "val_auc":  []}

    # ── Phase 1: Train head only (warmup) ─────────────────────
    print("\n" + "="*60)
    print("🚀  Phase 1 — Warmup: training classifier head only")
    print("="*60)
    model.freeze_backbone()
    model.count_trainable_params()

    optimizer  = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=config.LEARNING_RATE * 5, weight_decay=config.WEIGHT_DECAY)
    scheduler  = CosineAnnealingLR(optimizer, T_max=config.WARMUP_EPOCHS)

    early_stopping = EarlyStopping(patience=config.PATIENCE,
                                   checkpoint_path=checkpoint_path)

    for epoch in range(1, config.WARMUP_EPOCHS + 1):
        t0 = time.time()
        print(f"\n[Warmup Epoch {epoch}/{config.WARMUP_EPOCHS}]")
        tr_loss, tr_acc, tr_f1, tr_auc = run_epoch(model, train_loader, criterion,
                                                    optimizer, device, phase="train")
        vl_loss, vl_acc, vl_f1, vl_auc = run_epoch(model, val_loader, criterion,
                                                    None, device, phase="val")
        scheduler.step()
        _log(epoch, tr_loss, tr_acc, tr_f1, tr_auc,
                    vl_loss, vl_acc, vl_f1, vl_auc, time.time()-t0)
        _record(history, tr_loss, tr_acc, tr_f1, tr_auc,
                         vl_loss, vl_acc, vl_f1, vl_auc)

    # ── Phase 2: Fine-tune all layers ─────────────────────────
    print("\n" + "="*60)
    print("🔥  Phase 2 — Full fine-tuning")
    print("="*60)
    model.unfreeze_last_n_blocks(n=6)
    model.count_trainable_params()

    optimizer = AdamW(model.parameters(),
                      lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config.NUM_EPOCHS - config.WARMUP_EPOCHS)

    # ── Resume من آخر checkpoint لو موجود ──
    resume_path = os.path.join(config.CHECKPOINT_DIR, "resume_checkpoint.pt")
    start_epoch = config.WARMUP_EPOCHS + 1

    if os.path.exists(resume_path):
        print(f"\n🔄  Resume checkpoint found → {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch           = ckpt["epoch"] + 1
        early_stopping.best_score = ckpt["best_score"]
        early_stopping.counter    = ckpt["counter"]
        history                   = ckpt["history"]
        print(f"   ✅  Resuming from epoch {start_epoch}")
    else:
        print("\n🆕  Starting fresh training.")

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        t0 = time.time()
        print(f"\n[Epoch {epoch}/{config.NUM_EPOCHS}]")
        tr_loss, tr_acc, tr_f1, tr_auc = run_epoch(model, train_loader, criterion,
                                                    optimizer, device, "train",
                                                    use_mixup=True)
        vl_loss, vl_acc, vl_f1, vl_auc = run_epoch(model, val_loader, criterion,
                                                    None, device, "val")
        scheduler.step()
        _log(epoch, tr_loss, tr_acc, tr_f1, tr_auc,
                    vl_loss, vl_acc, vl_f1, vl_auc, time.time()-t0)
        _record(history, tr_loss, tr_acc, tr_f1, tr_auc,
                         vl_loss, vl_acc, vl_f1, vl_auc)

        early_stopping(vl_loss, model, optimizer, scheduler, epoch)

        # ── احفظ resume checkpoint بعد كل epoch ──
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_score":           early_stopping.best_score,
            "counter":              early_stopping.counter,
            "history":              history,
        }, resume_path)

        if early_stopping.early_stop:
            print("⛔  Early stopping triggered.")
            break

    # Load best weights back
    best_ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"\n✅  Training complete. Best model loaded from {checkpoint_path}")

    # امسح الـ resume checkpoint بعد ما التدريب يخلص
    if os.path.exists(resume_path):
        os.remove(resume_path)
        print("🗑️  Resume checkpoint deleted.")

    _plot_history(history, config.PLOTS_DIR)
    return model, history


# ──────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────
def _log(epoch, tl, ta, tf, tauc, vl, va, vf, vauc, elapsed):
    print(f"   Train → loss={tl:.4f}  acc={ta:.4f}  f1={tf:.4f}  auc={tauc:.4f}")
    print(f"   Val   → loss={vl:.4f}  acc={va:.4f}  f1={vf:.4f}  auc={vauc:.4f}")
    print(f"   ⏱  {elapsed:.1f}s")


def _record(h, tl, ta, tf, tauc, vl, va, vf, vauc):
    h["train_loss"].append(tl); h["val_loss"].append(vl)
    h["train_acc"].append(ta);  h["val_acc"].append(va)
    h["train_f1"].append(tf);   h["val_f1"].append(vf)
    h["train_auc"].append(tauc);h["val_auc"].append(vauc)


def _plot_history(history, plots_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    for ax, metric, title in zip(
        axes,
        [("train_loss","val_loss"), ("train_acc","val_acc"), ("train_auc","val_auc")],
        ["Loss", "Accuracy", "AUC-ROC"]
    ):
        ax.plot(epochs, history[metric[0]], label="Train")
        ax.plot(epochs, history[metric[1]], label="Val")
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(True)

    plt.tight_layout()
    path = os.path.join(plots_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    print(f"📈 Training curves saved → {path}")
    plt.close()
