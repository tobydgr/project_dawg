# evaluate_cmp.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

from dataset import StanfordDogsWithGaroDataset, build_class_mapping


IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path("output_dog_project_compare")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "efficientnet_b0_stanforddogs120_plus_garo121.pth"


def get_eval_transform():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])


def create_model(num_classes):
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


@torch.no_grad()
def predict(model, loader, idx_to_class):
    model.eval()

    y_true, y_pred, rows = [], [], []

    for batch in loader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].cpu().numpy()
        paths = batch["path"]
        true_names = batch["class_name"]

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, dim=1)

        preds = preds.cpu().numpy()
        confs = confs.cpu().numpy()

        for path, yt, yp, conf, true_name in zip(paths, labels, preds, confs, true_names):
            pred_name = idx_to_class[int(yp)]
            rows.append({
                "file": Path(path).name,
                "true_label_idx": int(yt),
                "true_label_name": true_name,
                "pred_label_idx": int(yp),
                "pred_label_name": pred_name,
                "confidence": float(conf),
                "correct": int(yt) == int(yp)
            })

        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

    return np.array(y_true), np.array(y_pred), pd.DataFrame(rows)


def plot_confusion_matrix(cm, class_names, out_file):
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title("Confusion Matrix - 121 Klassen")
    plt.xlabel("Vorhergesagte Klasse")
    plt.ylabel("Wahre Klasse")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {MODEL_PATH}")

    dog_classes, class_to_idx, idx_to_class, garo_idx = build_class_mapping("train", garo_enabled=True)
    num_classes = len(dog_classes) + 1

    test_ds = StanfordDogsWithGaroDataset(
        folders=["test", "test_garo"],
        transform=get_eval_transform(),
        allowed_classes=set(range(num_classes)),
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        garo_idx=garo_idx,
    )

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = create_model(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    y_true, y_pred, result_df = predict(model, test_loader, idx_to_class)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    labels = list(range(num_classes))
    target_names = [idx_to_class[i] for i in labels]

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    result_df.to_csv(OUTPUT_DIR / "prediction_overview.csv", index=False)

    with open(OUTPUT_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "num_test_images": int(len(result_df)),
        "model_name": "efficientnet_b0",
    }

    with open(OUTPUT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(cm, target_names, OUTPUT_DIR / "confusion_matrix_121.png")

    garo_df = result_df[result_df["true_label_idx"] == garo_idx].copy()
    garo_correct = int(garo_df["correct"].sum())
    garo_total = int(len(garo_df))
    garo_acc = (garo_correct / garo_total) if garo_total > 0 else 0.0

    garo_confusions = (
        garo_df[garo_df["correct"] == False]["pred_label_name"]
        .value_counts()
        .to_dict()
    )

    with open(OUTPUT_DIR / "garo_analysis.json", "w", encoding="utf-8") as f:
        json.dump({
            "garo_test_images": garo_total,
            "garo_correct": garo_correct,
            "garo_accuracy_within_own_testset": garo_acc,
            "garo_most_common_confusions": garo_confusions,
            "model_name": "efficientnet_b0",
        }, f, indent=2)

    print("\n=== Ergebnis (EfficientNet-B0) ===")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Macro-F1:     {macro_f1:.4f}")
    print(f"Weighted-F1:  {weighted_f1:.4f}")
    print(f"Garo-Testbilder: {garo_total}")
    print(f"Garo korrekt:    {garo_correct}")
    print(f"Garo-Erkennung:  {garo_acc:.4f}")
    print(f"\nDateiübersicht gespeichert in: {OUTPUT_DIR / 'prediction_overview.csv'}")
    print(f"Confusion Matrix gespeichert in: {OUTPUT_DIR / 'confusion_matrix_121.png'}")


if __name__ == "__main__":
    main()