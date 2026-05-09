import torch

class Config:
    # ─── Paths ───────────────────────────────────────────────
    DATA_DIR        = r"D:\deepfake_detection\data\real_vs_fake\real-vs-fake"
    TRAIN_DIR       = rf"{DATA_DIR}\train"
    VALID_DIR       = rf"{DATA_DIR}\valid"
    CHECKPOINT_DIR  = r"D:\deepfake_detection\outputs\checkpoints"
    PLOTS_DIR       = r"D:\deepfake_detection\outputs\plots"

    # ─── Model ───────────────────────────────────────────────
    MODEL_NAME      = "google/vit-base-patch16-224"   # Vision Transformer
    NUM_CLASSES     = 2                                # real / fake
    IMAGE_SIZE      = 224

    # ─── Training ────────────────────────────────────────────
    BATCH_SIZE      = 32
    NUM_EPOCHS      = 8
    LEARNING_RATE   = 2e-4
    WEIGHT_DECAY    = 1e-2
    WARMUP_EPOCHS   = 3
    PATIENCE        = 5          # Early stopping

    # ─── Augmentation ────────────────────────────────────────
    MEAN            = [0.5, 0.5, 0.5]
    STD             = [0.5, 0.5, 0.5]

    # ─── System ──────────────────────────────────────────────
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS     = 4
    SEED            = 42

    # ─── Labels ──────────────────────────────────────────────
    CLASSES         = ["fake", "real"]   # alphabetical = 0,1
    CLASS_NAMES     = {0: "Fake", 1: "Real"}
