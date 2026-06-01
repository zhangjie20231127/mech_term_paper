from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class CheXpertSample:
    image_path: str
    patient_id: str
    disease_targets: list[float]
    domain_target: int
    raw_path: str


def _parse_patient_id(path_str: str) -> str:
    parts = path_str.split("/")
    for part in parts:
        if part.startswith("patient"):
            return part
    return "unknown"


def _parse_domain(row: dict, domain_mode: str) -> int | None:
    if domain_mode == "frontal_lateral":
        value = (row.get("Frontal/Lateral") or "").strip()
        if value == "Frontal":
            return 0
        if value == "Lateral":
            return 1
        return None
    raise ValueError(f"Unsupported domain_mode: {domain_mode}")


def _parse_label(value: str, uncertain_policy: str) -> float:
    if value is None or value == "":
        return 0.0
    numeric = float(value)
    if numeric == -1.0:
        if uncertain_policy == "zero":
            return 0.0
        if uncertain_policy == "one":
            return 1.0
        raise ValueError(f"Unsupported uncertain_policy: {uncertain_policy}")
    return 1.0 if numeric > 0 else 0.0


def load_chexpert_samples(
    csv_path: str,
    dataset_root: str,
    label_names: list[str],
    domain_mode: str,
    uncertain_policy: str,
) -> list[CheXpertSample]:
    samples: list[CheXpertSample] = []
    root = Path(dataset_root)

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            domain_target = _parse_domain(row, domain_mode)
            if domain_target is None:
                continue
            raw_path = row["Path"]
            rel_parts = raw_path.split("/", 1)
            rel_path = rel_parts[1] if len(rel_parts) == 2 else raw_path
            image_path = root / rel_path
            if not image_path.exists():
                continue
            disease_targets = [_parse_label(row.get(label_name), uncertain_policy) for label_name in label_names]
            samples.append(
                CheXpertSample(
                    image_path=str(image_path),
                    patient_id=_parse_patient_id(raw_path),
                    disease_targets=disease_targets,
                    domain_target=domain_target,
                    raw_path=raw_path,
                )
            )
    return samples


def select_subset(
    samples: list[CheXpertSample],
    subset_size: int,
    seed: int,
) -> list[CheXpertSample]:
    if subset_size <= 0 or subset_size >= len(samples):
        return list(samples)

    grouped: dict[int, list[CheXpertSample]] = {0: [], 1: []}
    for sample in samples:
        grouped[sample.domain_target].append(sample)

    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)

    per_domain = max(1, subset_size // 2)
    subset = grouped[0][:per_domain] + grouped[1][:per_domain]
    remaining = subset_size - len(subset)
    if remaining > 0:
        leftovers = grouped[0][per_domain:] + grouped[1][per_domain:]
        rng.shuffle(leftovers)
        subset.extend(leftovers[:remaining])

    rng.shuffle(subset)
    return subset


class CheXpertDataset(Dataset):
    def __init__(self, samples: list[CheXpertSample], transform: Callable | None = None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "disease_targets": torch.tensor(sample.disease_targets, dtype=torch.float32),
            "domain_target": torch.tensor(sample.domain_target, dtype=torch.long),
            "patient_id": sample.patient_id,
            "path": sample.image_path,
            "raw_path": sample.raw_path,
        }
