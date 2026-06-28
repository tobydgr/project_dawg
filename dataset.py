# dataset.py
from pathlib import Path
from typing import List, Optional, Dict, Any

from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def discover_dog_classes(folder: Path) -> List[str]:
    if not folder.exists():
        return []
    return sorted([p.name for p in folder.iterdir() if p.is_dir()])


def build_class_mapping(train_folder: str = "train", garo_enabled: bool = True):
    train_path = Path(train_folder)
    dog_classes = discover_dog_classes(train_path)

    class_to_idx = {name: i for i, name in enumerate(dog_classes)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    garo_idx = None
    if garo_enabled:
        garo_idx = len(dog_classes)
        class_to_idx["garo"] = garo_idx
        idx_to_class[garo_idx] = "garo"

    return dog_classes, class_to_idx, idx_to_class, garo_idx


class StanfordDogsWithGaroDataset(Dataset):
    def __init__(
        self,
        folders: List[str],
        transform=None,
        allowed_classes: Optional[set] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        idx_to_class: Optional[Dict[int, str]] = None,
        garo_idx: Optional[int] = None,
    ):
        self.transform = transform
        self.allowed_classes = allowed_classes

        if class_to_idx is None or idx_to_class is None or garo_idx is None:
            _, class_to_idx, idx_to_class, garo_idx = build_class_mapping("train", garo_enabled=True)

        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.garo_idx = garo_idx

        self.samples = []
        self.class_counts = {}

        for folder_name in folders:
            folder = Path(folder_name)
            if not folder.exists():
                continue

            if folder.name in {"train_garo", "test_garo"}:
                for img_path in sorted(folder.rglob("*")):
                    if not is_image_file(img_path):
                        continue
                    label = self.garo_idx
                    if self.allowed_classes is not None and label not in self.allowed_classes:
                        continue
                    self.samples.append((img_path, label, "garo"))
                    self.class_counts["garo"] = self.class_counts.get("garo", 0) + 1
            else:
                for class_dir in sorted(folder.iterdir()):
                    if not class_dir.is_dir():
                        continue
                    class_name = class_dir.name
                    if class_name not in self.class_to_idx:
                        continue
                    label = self.class_to_idx[class_name]
                    if self.allowed_classes is not None and label not in self.allowed_classes:
                        continue

                    for img_path in sorted(class_dir.rglob("*")):
                        if not is_image_file(img_path):
                            continue
                        self.samples.append((img_path, label, class_name))
                        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1

        self.samples = sorted(self.samples, key=lambda x: str(x[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, label, class_name = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "class_name": class_name,
            "path": str(img_path),
        }


def summarize_dataset(train_folder="train", test_folder="test", train_garo="train_garo", test_garo="test_garo"):
    dog_classes, class_to_idx, idx_to_class, garo_idx = build_class_mapping(train_folder, garo_enabled=True)

    train_ds = StanfordDogsWithGaroDataset(
        folders=[train_folder, train_garo],
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        garo_idx=garo_idx,
    )
    test_ds = StanfordDogsWithGaroDataset(
        folders=[test_folder, test_garo],
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        garo_idx=garo_idx,
    )

    return {
        "num_dog_classes": len(dog_classes),
        "num_total_classes": len(dog_classes) + 1,
        "garo_idx": garo_idx,
        "train_counts": train_ds.class_counts,
        "test_counts": test_ds.class_counts,
    }