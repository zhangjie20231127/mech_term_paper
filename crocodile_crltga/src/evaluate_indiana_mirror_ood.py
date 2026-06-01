from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.network import CrocodileCrltgaNet
from utils.config import load_config
from utils.io import ensure_dir
from utils.random import seed_everything


ALL_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
OOD_LABELS = ["Atelectasis", "Cardiomegaly", "Pleural Effusion"]
OOD_INDEX = {name: ALL_LABELS.index(name) for name in OOD_LABELS}


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


class IndianaMirrorDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_size: int) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        image_obj = row["image"]
        if isinstance(image_obj, dict) and "bytes" in image_obj:
            image = Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
            raw_path = image_obj.get("path", f"row_{index}.png")
        else:
            raise ValueError("Unexpected image object format in Indiana mirror parquet")

        image = self.transform(image)
        targets = torch.zeros(len(ALL_LABELS), dtype=torch.float32)
        for label in OOD_LABELS:
            targets[OOD_INDEX[label]] = float(row[label])

        return {
            "image": image,
            "disease_targets": targets,
            "raw_path": raw_path,
            "path": raw_path,
        }


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = targets.cpu().numpy()
    aucs = {}
    tuned_f1 = {}
    best_thresholds = {}
    for label in OOD_LABELS:
        idx = OOD_INDEX[label]
        aucs[label] = float(roc_auc_score(labels[:, idx], probs[:, idx]))
        best_t = 0.5
        best_f = -1.0
        for t in [x / 100 for x in range(101)]:
            pred = (probs[:, idx] >= t).astype(int)
            score = float(f1_score(labels[:, idx], pred, zero_division=0))
            if score > best_f:
                best_f = score
                best_t = t
        tuned_f1[label] = best_f
        best_thresholds[label] = best_t

    return {
        "macro_auc_3class": float(sum(aucs.values()) / len(aucs)),
        "macro_tuned_f1_3class": float(sum(tuned_f1.values()) / len(tuned_f1)),
        "per_class_auc": aucs,
        "per_class_tuned_f1": tuned_f1,
        "best_thresholds": best_thresholds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="indiana_mirror_ood")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])
    device = torch.device("cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu")

    df = pd.read_parquet(args.parquet)
    df = build_labels(df)
    dataset = IndianaMirrorDataset(df, int(config["image_size"]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = CrocodileCrltgaNet(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    outputs = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))["disease_logits"]
            outputs.append(logits.cpu())
            targets.append(batch["disease_targets"])

    logits = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    metrics = compute_metrics(logits, targets)
    positives = {label: int(df[label].sum()) for label in OOD_LABELS}

    record = {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "rows": len(df),
        "positive_counts": positives,
        "valid": metrics,
    }
    out_dir = ensure_dir(Path(args.output_dir))
    (out_dir / f"{args.name}_metrics.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
