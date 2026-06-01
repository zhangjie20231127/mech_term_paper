from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd
import requests
import time
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.network import CrocodileCrltgaNet
from utils.config import load_config
from utils.io import ensure_dir
from utils.random import seed_everything


ALL_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
OPENI_LABELS = ["Atelectasis", "Cardiomegaly", "Pleural Effusion"]
OPENI_INDEX = {name: ALL_LABELS.index(name) for name in OPENI_LABELS}


def build_openi_dataframe(report_csv: Path, projection_csv: Path) -> pd.DataFrame:
    reports = pd.read_csv(report_csv).fillna("")
    projections = pd.read_csv(projection_csv)

    frontal = projections[projections["projection"].str.lower() == "frontal"].copy().reset_index(drop=False)
    frontal = frontal.rename(columns={"index": "row_idx"})
    merged = frontal.merge(reports[["uid", "MeSH", "Problems"]], on="uid", how="left").fillna("")
    struct = (merged["MeSH"].astype(str) + ";" + merged["Problems"].astype(str)).str.lower()

    merged["Atelectasis"] = struct.str.contains(r"atelect", regex=True)
    merged["Cardiomegaly"] = struct.str.contains(r"cardiomeg", regex=True)
    merged["Pleural Effusion"] = struct.str.contains(r"pleural effusion|(^|[; /])effusion([; /]|$)", regex=True)

    keep_cols = ["row_idx", "uid", "filename", *OPENI_LABELS]
    return merged[keep_cols].reset_index(drop=True)


def fetch_openi_urls(row_indices: list[int], page: int = 20, retries: int = 4) -> dict[int, str]:
    urls: dict[int, str] = {}
    base = "https://datasets-server.huggingface.co/rows"
    row_indices = sorted(row_indices)
    for start in range(0, len(row_indices), page):
        batch = row_indices[start : start + page]
        offset = batch[0]
        length = batch[-1] - batch[0] + 1
        params = {
            "dataset": "sasi2004/chest-xrays-indiana-university",
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                resp = requests.get(base, params=params, timeout=90)
                resp.raise_for_status()
                data = resp.json()
                wanted = set(batch)
                for item in data["rows"]:
                    if item["row_idx"] in wanted:
                        urls[item["row_idx"]] = item["row"]["image"]["src"]
                break
            except Exception as exc:  # pragma: no cover - transient network path
                last_error = exc
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Failed to fetch OpenI row batch starting at {offset}") from last_error
    return urls


class OpenIStreamDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, cache_root: Path, image_size: int) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
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
        cache_path = self.cache_root / row["filename"]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            response = requests.get(row["url"], timeout=120)
            response.raise_for_status()
            cache_path.write_bytes(response.content)

        image = Image.open(cache_path).convert("RGB")
        image = self.transform(image)

        full_targets = torch.zeros(len(ALL_LABELS), dtype=torch.float32)
        for name in OPENI_LABELS:
            full_targets[OPENI_INDEX[name]] = float(row[name])

        return {
            "image": image,
            "disease_targets": full_targets,
            "path": str(cache_path),
            "raw_path": row["filename"],
            "uid": int(row["uid"]),
        }


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = targets.cpu().numpy()

    aucs = {}
    f1s = {}
    best_thresholds = {}
    selected_indices = [OPENI_INDEX[name] for name in OPENI_LABELS]
    for idx, name in zip(selected_indices, OPENI_LABELS):
        aucs[name] = float(roc_auc_score(labels[:, idx], probs[:, idx]))
        thresholds = [x / 100 for x in range(101)]
        best_t = 0.5
        best_f = -1.0
        for t in thresholds:
            pred = (probs[:, idx] >= t).astype(int)
            score = float(f1_score(labels[:, idx], pred, zero_division=0))
            if score > best_f:
                best_f = score
                best_t = t
        f1s[name] = best_f
        best_thresholds[name] = best_t

    return {
        "macro_auc_3class": float(sum(aucs.values()) / len(aucs)),
        "macro_tuned_f1_3class": float(sum(f1s.values()) / len(f1s)),
        "per_class_auc": aucs,
        "per_class_tuned_f1": f1s,
        "best_thresholds": best_thresholds,
    }


def evaluate_model(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    seed_everything(config["seed"])
    device = torch.device("cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu")

    dataframe = build_openi_dataframe(Path(args.report_csv), Path(args.projection_csv))
    urls = fetch_openi_urls(dataframe["row_idx"].tolist())
    dataframe["url"] = dataframe["row_idx"].map(urls)
    if dataframe["url"].isna().any():
        raise ValueError("Some OpenI frontal rows are missing signed URLs")

    dataset = OpenIStreamDataset(dataframe, Path(args.cache_root), int(config["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

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
    return {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "rows": len(dataframe),
        "valid": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report-csv", required=True)
    parser.add_argument("--projection-csv", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="openi_ood")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    record = evaluate_model(args)
    out_dir = ensure_dir(Path(args.output_dir))
    (out_dir / f"{args.name}_metrics.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
