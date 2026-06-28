# train.py
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from dataset import StanfordDogsWithGaroDataset, build_class_mapping, summarize_dataset


SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
HEAD_EPOCHS = 4
FINETUNE_EPOCHS = 4
VAL_RATIO = 0.15
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path("output_dog_project")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "resnet18_stanforddogs120_plus_garo121.pth"
SUMMARY_PATH = OUTPUT_DIR / "dataset_summary.json"
HISTORY_PATH = OUTPUT_DIR / "train_history.json"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_train_transform():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])


def get_eval_transform():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])


def create_model(num_classes):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_stage(model, train_loader, val_loader, epochs, lr, stage_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )

    history = []
    best_acc = -1.0
    best_state = deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        row = {
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
        history.append(row)

        print(
            f"[{stage_name}] Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history, best_acc


def main():
    set_seed(SEED)

    dog_classes, class_to_idx, idx_to_class, garo_idx = build_class_mapping("train", garo_enabled=True)
    num_classes = len(dog_classes) + 1

    full_ds = StanfordDogsWithGaroDataset(
        folders=["train", "train_garo"],
        transform=get_train_transform(),
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        garo_idx=garo_idx,
        allowed_classes=set(range(num_classes)),
    )

    if len(full_ds) == 0:
        raise RuntimeError("Kein Trainingsdatensatz gefunden.")

    val_len = max(1, int(len(full_ds) * VAL_RATIO))
    train_len = len(full_ds) - val_len

    generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = random_split(full_ds, [train_len, val_len], generator=generator)

    val_subset.dataset.transform = get_eval_transform()

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = create_model(num_classes)

    freeze_backbone(model)
    model, hist_head, best_head = train_stage(
        model, train_loader, val_loader, HEAD_EPOCHS, LR_HEAD, "head"
    )

    unfreeze_all(model)
    model, hist_ft, best_ft = train_stage(
        model, train_loader, val_loader, FINETUNE_EPOCHS, LR_FINETUNE, "finetune"
    )

    torch.save(model.state_dict(), MODEL_PATH)

    dataset_info = summarize_dataset("train", "test", "train_garo", "test_garo")
    dataset_info["best_acc"] = float(best_ft)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist_head + hist_ft, f, indent=2)

    print("\nTraining abgeschlossen.")
    print(f"Modell gespeichert: {MODEL_PATH}")
    print(f"Dataset-Summary:    {SUMMARY_PATH}")
    print(f"Train-History:      {HISTORY_PATH}")


if __name__ == "__main__":
    main()