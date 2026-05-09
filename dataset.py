import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ──────────────────────────────────────────────────────────────
#  Transforms
# ──────────────────────────────────────────────────────────────
def get_transforms(image_size, mean, std, mode="train"):
    """
    mode = 'train' → heavy augmentation
    mode = 'val'   → only resize + normalize
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            # Simulate compression artifacts (deepfakes often differ in compression)
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# ──────────────────────────────────────────────────────────────
#  Dataset Class
# ──────────────────────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    """
    Expects folder structure:
        root/
          real/  *.jpg
          fake/  *.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir  = root_dir
        self.transform = transform
        self.samples   = []   # list of (image_path, label)
        self.classes   = sorted(os.listdir(root_dir))   # ['fake', 'real']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            label = self.class_to_idx[cls]
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def class_distribution(self):
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        return {self.classes[k]: v for k, v in sorted(counts.items())}


# ──────────────────────────────────────────────────────────────
#  DataLoader Factory
# ──────────────────────────────────────────────────────────────
def get_dataloaders(config):
    train_tf = get_transforms(config.IMAGE_SIZE, config.MEAN, config.STD, mode="train")
    val_tf   = get_transforms(config.IMAGE_SIZE, config.MEAN, config.STD, mode="val")

    train_ds = DeepfakeDataset(config.TRAIN_DIR, transform=train_tf)
    val_ds   = DeepfakeDataset(config.VALID_DIR, transform=val_tf)

    print(f"✅ Train samples : {len(train_ds):,}  | dist: {train_ds.class_distribution()}")
    print(f"✅ Val   samples : {len(val_ds):,}  | dist: {val_ds.class_distribution()}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    return train_loader, val_loader
