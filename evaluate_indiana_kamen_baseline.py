from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import densenet121


ALL_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
OOD_LABELS = ["Atelectasis", "Cardiomegaly", "Pleural Effusion"]


def _positive_without_negation(text: str, positive_pattern: str, negative_pattern: str) -> bool:
    positive = re.search(positive_pattern, text) is not None
    negative = re.search(negative_pattern, text) is not None
    return positive and not negative


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    text = df["report"].fillna("").str.lower()
    out = df.copy()
    out["Atelectasis"] = text.apply(
        lambda t: _positive_without_negation(
            t,
            r"\batelecta(?:sis|tic)\b",
            r"\bno\b[^.]{0,40}\batelecta(?:sis|tic)\b|\bwithout\b[^.]{0,40}\batelecta(?:sis|tic)\b",
        )
    )
    out["Cardiomegaly"] = text.apply(
        lambda t: _positive_without_negation(
            t,
            r"\bcardiomegaly\b|\benlarged heart\b|\bcardiac enlargement\b",
            r"\bno\b[^.]{0,40}\bcardiomegaly\b|\bheart size is normal\b|\bcardiomediastinal silhouette is normal\b",
        )
    )
    out["Pleural Effusion"] = text.apply(
        lambda t: _positive_without_negation(
            t,
            r"\bpleural effusion\b|\bpleural effusions\b|\beffusion\b|\beffusions\b",
            r"\bno\b[^.]{0,50}\bpleural effusion\b|\bno\b[^.]{0,30}\beffusion[s]?\b|\bwithout\b[^.]{0,50}\bpleural effusion\b",
        )
    )
    return out


class IndianaMirrorKamenDataset(Dataset):
    def __init__(self, parquet_path: Path) -> None:
        df = pd.read_parquet(parquet_path)
        self.df = build_labels(df).reset_index(drop=True)
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
        image_obj = row["image"]
        image = Image.open(io.BytesIO(image_obj["bytes"])).convert("L")
        image = self.transform(image)
        target = torch.zeros(len(ALL_LABELS), dtype=torch.float32)
        for i, name in enumerate(ALL_LABELS):
            if name in row.index:
                target[i] = float(row[name])
        return image, target


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = targets.cpu().numpy()
    selected = [ALL_LABELS.index(x) for x in OOD_LABELS]
    aucs = {}
    best_thresholds = {}
    tuned_f1 = {}
    for idx, name in zip(selected, OOD_LABELS):
        aucs[name] = float(roc_auc_score(labels[:, idx], probs[:, idx]))
        best_t = 0.5
        best_f = -1.0
        for t in np.linspace(0.0, 1.0, 101):
            pred = (probs[:, idx] >= t).astype(int)
            score = float(f1_score(labels[:, idx], pred, zero_division=0))
            if score > best_f:
                best_f = score
                best_t = float(t)
        best_thresholds[name] = best_t
        tuned_f1[name] = best_f
    return {
        "macro_auc_3class": float(np.mean(list(aucs.values()))),
        "macro_tuned_f1_3class": float(np.mean(list(tuned_f1.values()))),
        "per_class_auc": aucs,
        "per_class_tuned_f1": tuned_f1,
        "best_thresholds": best_thresholds,
    }


def main() -> None:
    root = Path("/home/zhang/workplace/mech_term_paper")
    checkpoint_path = root / "_references/chexpert_kamen/runs/full_densenet121_pretrained_10e/checkpoint_latest.pt"
    parquet_path = Path("/home/zhang/dataset/indiana_mirror/test.parquet")
    dataset = IndianaMirrorKamenDataset(parquet_path)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = densenet121(pretrained=False).to(device)
    model.classifier = nn.Linear(model.classifier.in_features, out_features=len(ALL_LABELS)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    outputs = []
    targets = []
    with torch.no_grad():
        for images, y in loader:
            outputs.append(model(images.to(device)).cpu())
            targets.append(y)

    logits = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    metrics = compute_metrics(logits, targets)
    positive_counts = {label: int(dataset.df[label].sum()) for label in OOD_LABELS}

    record = {
        "name": "indiana_mirror_test_structured3_kamen_densenet121_pretrained",
        "checkpoint": str(checkpoint_path),
        "rows": len(dataset),
        "positive_counts": positive_counts,
        "valid": metrics,
    }
    out_dir = root / "indiana_mirror_ood_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "indiana_mirror_test_structured3_kamen_densenet121_pretrained_metrics.json"
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
