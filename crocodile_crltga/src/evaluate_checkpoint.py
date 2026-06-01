from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models.network import CrocodileCrltgaNet
from src.train import build_loaders, compute_pos_weight, run_epoch, setup_logger
from src.utils.config import load_config
from src.utils.io import ensure_dir
from src.utils.random import seed_everything


def averaged_state_dict(checkpoint_paths: list[str], weights: list[float], device: torch.device) -> tuple[dict, dict]:
    checkpoints = [torch.load(path, map_location=device) for path in checkpoint_paths]
    total_weight = sum(weights)
    base_state = checkpoints[0]["model"]
    avg_state = {}
    for name, tensor in base_state.items():
        if torch.is_floating_point(tensor):
            avg = torch.zeros_like(tensor, dtype=torch.float32)
            for checkpoint, weight in zip(checkpoints, weights):
                avg += checkpoint["model"][name].float() * weight
            avg /= total_weight
            avg_state[name] = avg.to(dtype=tensor.dtype)
        else:
            avg_state[name] = tensor.clone()
    return avg_state, checkpoints[0].get("config", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--weight", action="append", type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="eval")
    args = parser.parse_args()
    weights = args.weight or [1.0] * len(args.checkpoint)
    if len(weights) != len(args.checkpoint):
        raise ValueError("--weight must be provided once per --checkpoint")

    config = load_config(args.config)
    seed_everything(config["seed"])
    device = torch.device("cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu")

    output_dir = ensure_dir(Path(args.output_dir))
    logger = setup_logger(output_dir)
    logger.info("starting checkpoint evaluation name=%s checkpoints=%s weights=%s", args.name, args.checkpoint, weights)

    train_loader, valid_loader, train_subset, val_subset = build_loaders(config)
    del train_loader, val_subset

    model = CrocodileCrltgaNet(config).to(device)
    state, _ = averaged_state_dict(args.checkpoint, weights, device)
    model.load_state_dict(state)

    pos_weight_mode = config.get("pos_weight_mode", "none")
    pos_weight_cap = float(config.get("pos_weight_cap", 3.0))
    disease_pos_weight = compute_pos_weight(train_subset, config["label_names"], pos_weight_mode, pos_weight_cap)
    if disease_pos_weight is not None:
        disease_pos_weight = disease_pos_weight.to(device)

    metrics = run_epoch(model, valid_loader, None, config, device, logger, 0, disease_pos_weight)
    record = {"name": args.name, "checkpoints": args.checkpoint, "weights": weights, "valid": metrics}
    with open(output_dir / f"{args.name}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
