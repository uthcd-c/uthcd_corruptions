import argparse
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import regnet_x_400mf, RegNet_X_400MF_Weights

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 1):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True  # allow fastest convolution algo

# ----------------------------
# Early Stopping
# ----------------------------
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, current: float):
        if self.best_score is None:
            self.best_score = current
            return
        improvement = (current < self.best_score - self.min_delta) if self.mode == "min" else (current > self.best_score + self.min_delta)
        if improvement:
            self.best_score = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

# ----------------------------
# Metrics structure
# ----------------------------
@dataclass
class Metrics:
    best_epoch: int
    best_val_loss: float
    best_val_acc: float
    train_history: Dict[str, list]
    val_history: Dict[str, list]
    test_loss: float
    test_acc: float
    total_epochs_run: int
    elapsed_sec: float

# ----------------------------
# Dataset utilities
# ----------------------------
def build_transforms():
    # Required: resize to 224, invert, ToTensor, Normalize
    # Normalize uses ImageNet mean/std to match pretrained weights
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=7),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),       
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return tfm

def make_dataloaders(data_root: str, batch_size: int, num_workers: int = 4, dry_run: bool = False):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    tfm = build_transforms()
    train_ds = ImageFolder(train_dir, transform=tfm)
    val_ds = ImageFolder(val_dir, transform=tfm)
    test_ds = ImageFolder(test_dir, transform=tfm)

    if dry_run:
        # Overfit a single class: take class index 0 if available
        target_class = 0
        train_indices = [i for i, (_, y) in enumerate(train_ds.samples) if y == target_class]
        val_indices = [i for i, (_, y) in enumerate(val_ds.samples) if y == target_class]
        # Keep a small subset to speed up
        train_indices = train_indices[: min(64, len(train_indices))]
        val_indices = val_indices[: min(64, len(val_indices))]
        train_ds = Subset(train_ds, train_indices)
        val_ds = Subset(val_ds, val_indices)
        # Test on same class
        test_indices = [i for i, (_, y) in enumerate(test_ds.samples) if y == target_class]
        test_indices = test_indices[: min(64, len(test_indices))]
        test_ds = Subset(test_ds, test_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Determine num_classes from dataset
    if isinstance(train_ds, Subset):
        # Subset keeps original dataset in .dataset
        num_classes = len(train_ds.dataset.classes)
        class_to_idx = train_ds.dataset.class_to_idx
    else:
        num_classes = len(train_ds.classes)
        class_to_idx = train_ds.class_to_idx

    return train_loader, val_loader, test_loader, num_classes, class_to_idx

# ----------------------------
# Model setup
# ----------------------------
def build_model(num_classes: int, dropout_p: float = 0.4):
    # Load pretrained ImageNet-1K weights
    weights = RegNet_X_400MF_Weights.DEFAULT
    model = regnet_x_400mf(weights=weights)
    # Replace classifier to match num_classes and add stronger dropout (regularization)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(        
        nn.Linear(in_features, num_classes, bias=True),
    )
    return model

# ----------------------------
# Train / Eval loops
# ----------------------------
def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct, total

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        c, t = accuracy_from_logits(outputs, targets)
        correct += c
        total += t

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        c, t = accuracy_from_logits(outputs, targets)
        correct += c
        total += t
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

# ----------------------------
# Checkpointing
# ----------------------------
def save_checkpoint(state: Dict[str, Any], ckpt_path: Path):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_path)

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune regnetx_400 on custom dataset")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Root dataset dir containing train/val/test")
    parser.add_argument("--out_dir", type=str, default="outputs/regnetx_400", help="Directory to save checkpoints and metrics")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs)")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Early stopping minimal improvement")
    parser.add_argument("--dropout", type=float, default=0.4, help="Classifier dropout probability")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--dry_run", action="store_true", help="Overfit a single class for sanity-check")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    train_loader, val_loader, test_loader, num_classes, class_to_idx = make_dataloaders(
        data_root=args.data_dir, batch_size=args.batch_size, num_workers=4, dry_run=args.dry_run
    )

    model = build_model(num_classes=num_classes, dropout_p=args.dropout)
    model = model.to(device)

    # Mixed precision to fit larger batches within 23 GB GPU
    use_amp = False
    scaler = None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_val_loss = float("inf")
    best_val_acc = 0.0

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"
    metrics_path = out_dir / "metrics.json"

    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", scaler.state_dict()))
        start_epoch = state["epoch"] + 1
        best_val_loss = state.get("best_val_loss", best_val_loss)
        best_val_acc = state.get("best_val_acc", best_val_acc)

    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta, mode="min")

    train_hist = {"loss": [], "acc": []}
    val_hist = {"loss": [], "acc": []}

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_hist["loss"].append(train_loss)
        train_hist["acc"].append(train_acc)
        val_hist["loss"].append(val_loss)
        val_hist["acc"].append(val_acc)

        # Save last checkpoint
        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": None,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "class_to_idx": class_to_idx,
            "num_classes": num_classes,
            "args": vars(args),
        }, last_ckpt)

        # Save best checkpoint on val loss
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_checkpoint({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": None, #
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "class_to_idx": class_to_idx,
                "num_classes": num_classes,
                "args": vars(args),
            }, best_ckpt)

        # Early stopping checks
        early_stopper(val_loss)
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
        if early_stopper.should_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best checkpoint for final test
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    elapsed = time.time() - start_time
    best_epoch = state["epoch"] if best_ckpt.exists() else (len(val_hist["loss"]) - 1)

    metrics = Metrics(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        train_history=train_hist,
        val_history=val_hist,
        test_loss=test_loss,
        test_acc=test_acc,
        total_epochs_run=len(train_hist["loss"]),
        elapsed_sec=elapsed
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(asdict(metrics), f, indent=2)

    print(f"Saved best checkpoint to {best_ckpt}")
    print(f"Saved last checkpoint to {last_ckpt}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
