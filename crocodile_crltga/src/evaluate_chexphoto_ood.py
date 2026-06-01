from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.network import CrocodileCrltgaNet
from train import run_epoch, setup_logger
from utils.config import load_config
from utils.io import ensure_dir
from utils.random import seed_everything


LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


class CheXphotoOODDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_root: Path, image_size: int) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]
        image_path = self.image_root / row["file_name"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "disease_targets": torch.tensor(row[LABELS].astype(float).to_numpy(), dtype=torch.float32),
            "domain_target": torch.tensor(0, dtype=torch.long),
            "path": str(image_path),
            "raw_path": row["file_name"],
            "patient_id": row["key"].split("/")[0],
        }


def build_dataframe(chexphoto_csv: Path, chexpert_valid_csv: Path) -> pd.DataFrame:
    chexphoto = pd.read_csv(chexphoto_csv)
    chexphoto["key"] = chexphoto["file_name"].str.extract(r"(patient\d+/study\d+/view\d+_[^/]+\.jpg)$", expand=False)

    chexpert = pd.read_csv(chexpert_valid_csv)
    chexpert["key"] = chexpert["Path"].str.extract(r"(patient\d+/study\d+/view\d+_[^/]+\.jpg)$", expand=False)

    merged = chexphoto.merge(chexpert[["key", *LABELS]], on="key", how="left")
    if merged[LABELS].isna().any().any():
        missing = int(merged[LABELS].isna().any(axis=1).sum())
        raise ValueError(f"{missing} CheXphoto rows could not be matched back to CheXpert labels")
    return merged


def evaluate_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    chexphoto_csv: Path,
    chexphoto_images_root: Path,
    chexpert_valid_csv: Path,
    output_dir: Path,
    name: str,
) -> dict:
    config = load_config(str(config_path))
    seed_everything(config["seed"])
    device = torch.device("cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu")

    logger = setup_logger(ensure_dir(output_dir))
    logger.info("starting CheXphoto OOD eval name=%s checkpoint=%s", name, checkpoint_path)

    dataframe = build_dataframe(chexphoto_csv, chexpert_valid_csv)
    dataset = CheXphotoOODDataset(dataframe, chexphoto_images_root, int(config["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1 if int(config.get("num_workers", 4)) > 0 else None,
    )

    model = CrocodileCrltgaNet(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    # Pos weights are only used in training losses; OOD eval calls run_epoch in valid mode.
    metrics = run_epoch(model, loader, None, config, device, logger, 0, disease_pos_weight=None)
    record = {
        "name": name,
        "checkpoint": str(checkpoint_path),
        "rows": len(dataframe),
        "valid": metrics,
    }
    with open(output_dir / f"{name}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--chexphoto-csv", required=True)
    parser.add_argument("--chexphoto-images-root", required=True)
    parser.add_argument("--chexpert-valid-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="chexphoto_ood")
    parser.add_argument("--path-contains", default="")
    args = parser.parse_args()

    dataframe = build_dataframe(Path(args.chexphoto_csv), Path(args.chexpert_valid_csv))
    if args.path_contains:
        dataframe = dataframe[dataframe["file_name"].str.contains(args.path_contains, regex=False)].reset_index(drop=True)
        if len(dataframe) == 0:
            raise ValueError(f"No CheXphoto rows matched path filter: {args.path_contains}")

    config = load_config(str(Path(args.config)))
    seed_everything(config["seed"])
    device = torch.device("cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu")

    logger = setup_logger(ensure_dir(Path(args.output_dir)))
    logger.info("starting CheXphoto OOD eval name=%s checkpoint=%s rows=%d", args.name, args.checkpoint, len(dataframe))

    dataset = CheXphotoOODDataset(dataframe, Path(args.chexphoto_images_root), int(config["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1 if int(config.get("num_workers", 4)) > 0 else None,
    )

    model = CrocodileCrltgaNet(config).to(device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model"])
    metrics = run_epoch(model, loader, None, config, device, logger, 0, disease_pos_weight=None)
    record = {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "rows": len(dataframe),
        "path_contains": args.path_contains,
        "valid": metrics,
    }
    with open(Path(args.output_dir) / f"{args.name}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
