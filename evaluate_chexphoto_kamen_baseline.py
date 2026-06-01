from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import densenet121


LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


class CheXphotoKamenDataset(Dataset):
    def __init__(self, file_index_csv: Path, chexpert_valid_csv: Path, image_root: Path) -> None:
        idx = pd.read_csv(file_index_csv)
        idx["key"] = idx["file_name"].str.extract(r"(patient\d+/study\d+/view\d+_[^/]+\.jpg)$", expand=False)

        chex = pd.read_csv(chexpert_valid_csv)
        chex["key"] = chex["Path"].str.extract(r"(patient\d+/study\d+/view\d+_[^/]+\.jpg)$", expand=False)

        merged = idx.merge(chex[["key", *LABELS]], on="key", how="left")
        if merged[LABELS].isna().any().any():
            raise ValueError("Some CheXphoto rows could not be matched to CheXpert labels")

        self.df = merged.reset_index(drop=True)
        self.image_root = image_root
        self.transform = T.Compose(
            [
                T.Resize(320),
                T.CenterCrop(320),
                lambda x: torch.from_numpy(np.array(x, copy=True)).float().div(255).unsqueeze(0),
                T.Normalize(mean=[0.5330], std=[0.0349]),
                lambda x: x.expand(3, -1, -1),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(self.image_root / row["file_name"]).convert("L")
        image = self.transform(image)
        target = torch.tensor(row[LABELS].astype(np.float32).to_numpy(), dtype=torch.float32)
        return image, target


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = targets.cpu().numpy()
    aucs = {}
    for i, name in enumerate(LABELS):
        aucs[name] = float(roc_auc_score(labels[:, i], probs[:, i]))
    macro_auc = float(np.mean(list(aucs.values())))

    thresholds = np.linspace(0.0, 1.0, 101)
    best_thresholds = []
    tuned_preds = np.zeros_like(labels)
    for i in range(labels.shape[1]):
        best_t = 0.5
        best_f1 = -1.0
        for t in thresholds:
            pred = (probs[:, i] >= t).astype(np.int32)
            score = float(f1_score(labels[:, i], pred, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_t = float(t)
                tuned_preds[:, i] = pred
        best_thresholds.append(best_t)
    tuned_macro_f1 = float(f1_score(labels, tuned_preds, average="macro", zero_division=0))

    return {
        "macro_auc": macro_auc,
        "tuned_macro_f1": tuned_macro_f1,
        "best_thresholds": best_thresholds,
        "per_class_auc": aucs,
    }


def main() -> None:
    root = Path("/home/zhang/workplace/mech_term_paper")
    checkpoint_path = root / "_references/chexpert_kamen/runs/full_densenet121_pretrained_10e/checkpoint_latest.pt"
    dataset = CheXphotoKamenDataset(
        file_index_csv=Path("/home/zhang/dataset/chestx/valid.csv"),
        chexpert_valid_csv=Path("/home/zhang/dataset/chexpert-small/CheXpert-v1.0-small/valid.csv"),
        image_root=Path("/home/zhang/dataset/chestx/valid"),
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = densenet121(pretrained=False).to(device)
    model.classifier = nn.Linear(model.classifier.in_features, out_features=len(LABELS)).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    outputs = []
    targets = []
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            logits = model(images)
            outputs.append(logits.cpu())
            targets.append(y)

    logits = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    metrics = compute_metrics(logits, targets)
    record = {
        "name": "chexphoto_valid_kamen_densenet121_pretrained",
        "checkpoint": str(checkpoint_path),
        "rows": len(dataset),
        "valid": metrics,
    }

    out_dir = root / "chexphoto_ood_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chexphoto_valid_kamen_densenet121_pretrained_metrics.json"
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
