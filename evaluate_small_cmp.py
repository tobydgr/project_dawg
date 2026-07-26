# evaluate_small_cmp.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

from dataset import StanfordDogsWithGaroDataset, build_class_mapping


IMG_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0
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
def predict_small(model, loader, idx_to_class):
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
            correct = int(yt) == int(yp)

            rows.append({
                "file": Path(path).name,
                "true_label_idx": int(yt),
                "true_label_name": true_name,
                "pred_label_idx": int(yp),
                "pred_label_name": pred_name,
                "confidence": float(conf),
                "correct": correct
            })

            y_true.append(int(yt))
            y_pred.append(int(yp))

    return y_true, y_pred, pd.DataFrame(rows)


def save_small_confusion_matrix(y_true, y_pred, idx_to_class, out_png):
    present_labels = sorted(list(set(y_true) | set(y_pred)))
    present_names = [idx_to_class[i] for i in present_labels]
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)

    plt.figure(figsize=(max(6, len(present_labels) * 1.2), max(5, len(present_labels) * 1.0)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=present_names, yticklabels=present_names, cbar=True
    )
    plt.title("Confusion Matrix - nur vorkommende Klassen")
    plt.xlabel("Vorhergesagte Klasse")
    plt.ylabel("Wahre Klasse")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return present_labels, present_names, cm.tolist()


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

    y_true, y_pred, result_df = predict_small(model, test_loader, idx_to_class)

    if len(result_df) == 0:
        raise RuntimeError("Keine Testbilder gefunden.")

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    result_df = result_df.sort_values("file").reset_index(drop=True)
    result_df.to_csv(OUTPUT_DIR / "prediction_overview_small.csv", index=False)

    present_labels, present_names, cm = save_small_confusion_matrix(
        y_true, y_pred, idx_to_class, OUTPUT_DIR / "confusion_matrix_small.png"
    )

    summary = {
        "num_test_images": int(len(result_df)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "present_class_indices": present_labels,
        "present_class_names": present_names,
        "note": "Kleines Testset: Metriken nur vorsichtig interpretieren.",
        "model_name": "efficientnet_b0",
    }

    with open(OUTPUT_DIR / "metrics_small.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "confusion_matrix_small.json", "w", encoding="utf-8") as f:
        json.dump({
            "labels": present_labels,
            "label_names": present_names,
            "matrix": cm,
            "model_name": "efficientnet_b0",
        }, f, indent=2)

    print("\n=== Kleine Testauswertung (EfficientNet-B0) ===")
    print(f"Anzahl Testbilder: {len(result_df)}")
    print(f"Accuracy:         {acc:.4f}")
    print(f"Macro-F1:         {macro_f1:.4f}")
    print(f"Weighted-F1:      {weighted_f1:.4f}")
    print("\nEinzelausgaben pro Bild:")

    for _, row in result_df.iterrows():
        status = "KORREKT" if row["correct"] else "FALSCH"
        print(
            f"- {row['file']}: wahr={row['true_label_name']} | "
            f"vorhergesagt={row['pred_label_name']} | "
            f"confidence={row['confidence']:.4f} | {status}"
        )


if __name__ == "__main__":
    main()