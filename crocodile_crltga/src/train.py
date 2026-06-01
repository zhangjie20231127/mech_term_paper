from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.chexpert import CheXpertDataset, load_chexpert_samples, select_subset
from src.models.disentangler import batch_triplet_loss, dag_penalty, uniform_multiclass_penalty, uniform_multilabel_penalty
from src.models.network import CrocodileCrltgaNet
from src.utils.config import load_config
from src.utils.io import append_jsonl, ensure_dir, write_json
from src.utils.metrics import (
    binary_accuracy,
    multilabel_macro_auc,
    multilabel_macro_f1,
    multilabel_mean_accuracy,
    multilabel_per_class_diagnostics,
    multilabel_tuned_thresholds,
)
from src.utils.random import seed_everything


def _trainable_parameters(module: nn.Module):
    return [param for param in module.parameters() if param.requires_grad]


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("crocodile_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(output_dir / "train_status.txt", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def build_transforms(image_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def build_loaders(config: dict):
    train_samples = load_chexpert_samples(
        csv_path=config["train_csv"],
        dataset_root=config["dataset_root"],
        label_names=config["label_names"],
        domain_mode=config["domain_mode"],
        uncertain_policy=config["uncertain_policy"],
    )
    valid_samples = load_chexpert_samples(
        csv_path=config["valid_csv"],
        dataset_root=config["dataset_root"],
        label_names=config["label_names"],
        domain_mode=config["domain_mode"],
        uncertain_policy=config["uncertain_policy"],
    )

    train_subset = select_subset(train_samples, config["train_subset_size"], config["seed"])
    val_subset = select_subset(valid_samples, config["val_subset_size"], config["seed"] + 1)
    train_transform, eval_transform = build_transforms(config["image_size"])

    train_dataset = CheXpertDataset(train_subset, transform=train_transform)
    valid_dataset = CheXpertDataset(val_subset, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1 if config["num_workers"] > 0 else None,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1 if config["num_workers"] > 0 else None,
    )
    return train_loader, valid_loader, train_subset, val_subset


def compute_pos_weight(samples, label_names: list[str], mode: str, cap: float) -> torch.Tensor | None:
    if mode == "none":
        return None

    targets = torch.tensor([sample.disease_targets for sample in samples], dtype=torch.float32)
    positives = targets.sum(dim=0)
    negatives = targets.size(0) - positives
    raw = negatives / positives.clamp_min(1.0)

    if mode == "sqrt_neg_pos":
        weights = torch.sqrt(raw)
    elif mode == "neg_pos":
        weights = raw
    else:
        raise ValueError(f"Unsupported pos_weight_mode: {mode}")

    return weights.clamp(max=cap)


def asymmetric_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_neg: float = 2.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
) -> torch.Tensor:
    probs_pos = torch.sigmoid(logits)
    probs_neg = 1.0 - probs_pos
    if clip > 0.0:
        probs_neg = (probs_neg + clip).clamp(max=1.0)

    log_pos = torch.log(probs_pos.clamp(min=1e-8))
    log_neg = torch.log(probs_neg.clamp(min=1e-8))
    loss = targets * log_pos + (1.0 - targets) * log_neg

    pt = probs_pos * targets + probs_neg * (1.0 - targets)
    gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
    loss *= torch.pow(1.0 - pt, gamma)
    return -loss.mean()


def compute_loss(config: dict, outputs: dict, batch: dict, disease_pos_weight: torch.Tensor | None = None):
    disease_targets = batch["disease_targets"].float()
    domain_targets = batch["domain_target"]

    z_x = torch.nan_to_num(outputs["z_x"], nan=0.0, posinf=20.0, neginf=-20.0).float()
    z_c = torch.nan_to_num(outputs["z_c"], nan=0.0, posinf=20.0, neginf=-20.0).float()
    z_c_cap = torch.nan_to_num(outputs["z_c_cap"], nan=0.0, posinf=20.0, neginf=-20.0).float()
    z_x_domain = torch.nan_to_num(outputs["z_x_domain"], nan=0.0, posinf=20.0, neginf=-20.0).float()
    z_c_domain = torch.nan_to_num(outputs["z_c_domain"], nan=0.0, posinf=20.0, neginf=-20.0).float()
    z_c_cap_domain = torch.nan_to_num(outputs["z_c_cap_domain"], nan=0.0, posinf=20.0, neginf=-20.0).float()
    loss_type = str(config.get("loss_type", "bce")).lower()
    if loss_type == "bce":
        disease_main = F.binary_cross_entropy_with_logits(z_x, disease_targets, pos_weight=disease_pos_weight)
    elif loss_type == "asl":
        disease_main = asymmetric_loss_with_logits(
            z_x,
            disease_targets,
            gamma_neg=float(config.get("asl_gamma_neg", 2.0)),
            gamma_pos=float(config.get("asl_gamma_pos", 0.0)),
            clip=float(config.get("asl_clip", 0.05)),
        )
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    disease_sp = uniform_multilabel_penalty(z_c)
    disease_bd = F.binary_cross_entropy_with_logits(z_c_cap, disease_targets, pos_weight=disease_pos_weight)
    domain_main = F.cross_entropy(z_x_domain, domain_targets)
    domain_sp = uniform_multiclass_penalty(z_c_domain)
    domain_bd = F.cross_entropy(z_c_cap_domain, domain_targets)
    lambda_d_main = float(config.get("lambda_d_main", 1.0))
    triplet_weight = float(config.get("lambda_triplet", 0.0))
    dag_weight = float(config.get("lambda_dag", 0.0))
    triplet = outputs["z_x"].new_tensor(0.0)
    if triplet_weight != 0.0:
        triplet = batch_triplet_loss(outputs["disease_embeddings"], disease_targets, config["triplet_margin"])
    dag = outputs["z_x"].new_tensor(0.0)
    if dag_weight != 0.0:
        dag = dag_penalty(outputs["disease_adjacency"])

    total = (
        disease_main
        + config["lambda_y_sp"] * disease_sp
        + config["lambda_y_bd"] * disease_bd
        + lambda_d_main * domain_main
        + config["lambda_d_sp"] * domain_sp
        + config["lambda_d_bd"] * domain_bd
        + config["lambda_triplet"] * triplet
        + config["lambda_dag"] * dag
    )
    components = {
        "disease_main": disease_main.detach().item(),
        "disease_sp": disease_sp.detach().item(),
        "disease_bd": disease_bd.detach().item(),
        "domain_main": domain_main.detach().item(),
        "domain_sp": domain_sp.detach().item(),
        "domain_bd": domain_bd.detach().item(),
        "triplet": triplet.detach().item(),
        "dag": dag.detach().item(),
        "total": total.detach().item(),
    }
    return total, components


def build_optimizer(model: nn.Module, config: dict) -> Adam:
    base_lr = float(config["lr"])
    weight_decay = float(config["weight_decay"])
    disease_feature_lr = config.get("disease_feature_block_lr")

    if disease_feature_lr is None:
        return Adam((param for param in model.parameters() if param.requires_grad), lr=base_lr, weight_decay=weight_decay)

    disease_feature_lr = float(disease_feature_lr)
    disease_feature_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("disease_feature_block."):
            disease_feature_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "main"})
    if disease_feature_params:
        param_groups.append({"params": disease_feature_params, "lr": disease_feature_lr, "name": "disease_feature_block"})
    return Adam(param_groups, weight_decay=weight_decay)


def build_scheduler(config: dict, optimizer: Adam):
    scheduler_name = str(config.get("lr_scheduler", "none")).lower()
    if scheduler_name == "none":
        return None

    if scheduler_name == "cosine":
        total_epochs = max(1, int(config.get("scheduler_t_max", config["epochs"])))
        min_factor = float(config.get("scheduler_min_lr_factor", 0.1))

        def cosine_factor(epoch_index: int) -> float:
            progress = min(epoch_index, total_epochs) / total_epochs
            cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
            return min_factor + (1.0 - min_factor) * cosine

        return LambdaLR(optimizer, lr_lambda=cosine_factor)

    if scheduler_name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(config.get("scheduler_factor", 0.5)),
            patience=int(config.get("scheduler_patience", 3)),
            min_lr=float(config.get("scheduler_min_lr", 1e-6)),
        )

    raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")


def get_learning_rates(optimizer: Adam) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def summarize_optimizer_groups(optimizer: Adam) -> list[dict]:
    summary = []
    for index, group in enumerate(optimizer.param_groups):
        param_count = sum(param.numel() for param in group["params"])
        trainable_count = sum(param.numel() for param in group["params"] if param.requires_grad)
        summary.append(
            {
                "index": index,
                "name": group.get("name", f"group_{index}"),
                "lr": float(group["lr"]),
                "params": int(param_count),
                "trainable_params": int(trainable_count),
            }
        )
    return summary


def step_scheduler(scheduler, optimizer: Adam, valid_metrics: dict) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(valid_metrics.get("macro_auc", valid_metrics.get("macro_f1", 0.0)))
    else:
        scheduler.step()


def align_epoch_scheduler(scheduler, start_epoch: int) -> None:
    if scheduler is None or isinstance(scheduler, ReduceLROnPlateau) or start_epoch <= 0:
        return
    if isinstance(scheduler, LambdaLR):
        learning_rates = []
        for group, base_lr, lr_lambda in zip(scheduler.optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas):
            lr = float(base_lr) * float(lr_lambda(start_epoch))
            group["lr"] = lr
            learning_rates.append(lr)
        scheduler.last_epoch = start_epoch
        scheduler._last_lr = learning_rates


def first_nonfinite_tensor(tensors: dict) -> tuple[str, torch.Tensor] | None:
    for name, value in tensors.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            return name, value
    return None


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = dict(batch)
    moved["image"] = batch["image"].to(device, non_blocking=True)
    moved["disease_targets"] = batch["disease_targets"].to(device, non_blocking=True)
    moved["domain_target"] = batch["domain_target"].to(device, non_blocking=True)
    return moved


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam | None,
    config: dict,
    device: torch.device,
    logger: logging.Logger,
    epoch_index: int,
    disease_pos_weight: torch.Tensor | None = None,
):
    is_train = optimizer is not None
    model.train(is_train)
    phase_start = time.time()
    accumulation_steps = max(1, int(config.get("gradient_accumulation_steps", 1)))
    log_interval = max(1, int(config.get("log_interval", 50)))
    phase_name = "train" if is_train else "valid"

    all_disease_logits = []
    all_disease_targets = []
    all_domain_logits = []
    all_domain_targets = []
    component_sums = {}
    grad_clip_norm = float(config.get("gradient_clip_norm", 0.0))
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(batch["image"])
                bad_output = first_nonfinite_tensor(outputs)
                if bad_output is not None:
                    name, tensor = bad_output
                    raise FloatingPointError(
                        f"Non-finite tensor detected in forward output `{name}` at batch {batch_index + 1} "
                        f"with shape {tuple(tensor.shape)}."
                    )
            loss, components = compute_loss(config, outputs, batch, disease_pos_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss detected at batch {batch_index + 1}: {components}"
                )

        if is_train:
            scaler.scale(loss / accumulation_steps).backward()
            if (batch_index + 1) % accumulation_steps == 0 or (batch_index + 1) == len(loader):
                scaler.unscale_(optimizer)
                bad_grad_name = None
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        bad_grad_name = name
                        break
                if bad_grad_name is not None:
                    raise FloatingPointError(
                        f"Non-finite gradient detected at batch {batch_index + 1} for parameter `{bad_grad_name}`."
                    )
                if grad_clip_norm > 0:
                    clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        all_disease_logits.append(outputs["disease_logits"].detach().cpu())
        all_disease_targets.append(batch["disease_targets"].detach().cpu())
        all_domain_logits.append(outputs["domain_logits"].detach().cpu())
        all_domain_targets.append(batch["domain_target"].detach().cpu())
        for key, value in components.items():
            component_sums[key] = component_sums.get(key, 0.0) + value

        if (batch_index + 1) % log_interval == 0 or (batch_index + 1) == len(loader):
            elapsed = time.time() - phase_start
            avg_total = component_sums.get("total", 0.0) / (batch_index + 1)
            logger.info(
                "epoch=%d phase=%s batch=%d/%d avg_total=%.6f elapsed_sec=%.1f",
                epoch_index,
                phase_name,
                batch_index + 1,
                len(loader),
                avg_total,
                elapsed,
            )

    count = max(1, len(loader))
    disease_logits = torch.cat(all_disease_logits, dim=0)
    disease_targets = torch.cat(all_disease_targets, dim=0)
    domain_logits = torch.cat(all_domain_logits, dim=0)
    domain_targets = torch.cat(all_domain_targets, dim=0)

    metrics = {key: value / count for key, value in component_sums.items()}
    metrics["macro_f1"] = multilabel_macro_f1(disease_logits, disease_targets)
    metrics["mean_accuracy"] = multilabel_mean_accuracy(disease_logits, disease_targets)
    metrics["macro_auc"] = multilabel_macro_auc(disease_logits, disease_targets)
    metrics["domain_accuracy"] = binary_accuracy(domain_logits, domain_targets)
    if not is_train:
        metrics.update(multilabel_tuned_thresholds(disease_logits, disease_targets))
        metrics.update(multilabel_per_class_diagnostics(disease_logits, disease_targets, config["label_names"]))
    metrics["phase_time_sec"] = time.time() - phase_start
    return metrics


def save_grad_cam_sample(model: nn.Module, sample: dict, device: torch.device, output_path: Path, image_size: int) -> None:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients.append(grad_output[0].detach())

    handle_forward = model.backbone.stem[-1][-1].conv3.register_forward_hook(forward_hook)
    handle_backward = model.backbone.stem[-1][-1].conv3.register_full_backward_hook(backward_hook)

    image = sample["image"].unsqueeze(0).to(device)
    outputs = model(image)
    class_index = int(torch.argmax(torch.sigmoid(outputs["disease_logits"])[0]).item())
    model.zero_grad(set_to_none=True)
    outputs["disease_logits"][0, class_index].backward()

    if not activations or not gradients:
        handle_forward.remove()
        handle_backward.remove()
        return

    grad = gradients[-1][0]
    act = activations[-1][0]
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = F.relu((weights * act).sum(dim=0))
    cam = cam / cam.max().clamp_min(1e-6)
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)[0, 0].cpu()

    raw = sample["image"].cpu()
    raw = raw * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    raw = raw.clamp(0, 1).permute(1, 2, 0).numpy()
    heat = plt.get_cmap("jet")(cam.numpy())[:, :, :3]
    overlay = (0.55 * raw + 0.45 * heat).clip(0, 1)

    plt.figure(figsize=(5, 5))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    handle_forward.remove()
    handle_backward.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="", help="Path to checkpoint to resume from.")
    parser.add_argument(
        "--new-run-from-resume",
        action="store_true",
        help="Load weights from --resume but write metrics/checkpoints to a new output directory.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])

    device = torch.device("cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    if args.resume and not args.new_run_from_resume:
        output_dir = ensure_dir(Path(args.resume).resolve().parent)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = ensure_dir(Path(config["output_root"]) / f"{config['experiment_name']}_{timestamp}")
    logger = setup_logger(output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    if not args.resume or args.new_run_from_resume:
        with open(output_dir / "config_snapshot.json", "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)
    logger.info("starting training output_dir=%s device=%s resume=%s", output_dir, device, args.resume or "none")
    logger.info("config=%s", json.dumps(config, ensure_ascii=False, sort_keys=True))

    train_loader, valid_loader, train_subset, val_subset = build_loaders(config)
    split_summary = {
        "train_size": len(train_subset),
        "val_size": len(val_subset),
        "train_domains": {0: sum(s.domain_target == 0 for s in train_subset), 1: sum(s.domain_target == 1 for s in train_subset)},
        "val_domains": {0: sum(s.domain_target == 0 for s in val_subset), 1: sum(s.domain_target == 1 for s in val_subset)},
        "train_patients": len({s.patient_id for s in train_subset}),
        "val_patients": len({s.patient_id for s in val_subset}),
    }
    if not args.resume or args.new_run_from_resume:
        with open(output_dir / "split_summary.json", "w", encoding="utf-8") as handle:
            json.dump(split_summary, handle, indent=2, ensure_ascii=False)
    logger.info("split_summary=%s", json.dumps(split_summary, ensure_ascii=False, sort_keys=True))

    model = CrocodileCrltgaNet(config).to(device)
    pos_weight_mode = config.get("pos_weight_mode", "none")
    pos_weight_cap = float(config.get("pos_weight_cap", 3.0))
    disease_pos_weight = compute_pos_weight(train_subset, config["label_names"], pos_weight_mode, pos_weight_cap)
    if disease_pos_weight is not None:
        disease_pos_weight = disease_pos_weight.to(device)
        logger.info("disease_pos_weight=%s", disease_pos_weight.detach().cpu().tolist())
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(config, optimizer)
    logger.info("optimizer_groups=%s", json.dumps(summarize_optimizer_groups(optimizer), ensure_ascii=False, sort_keys=True))
    logger.info("lr_scheduler=%s initial_lrs=%s", config.get("lr_scheduler", "none"), get_learning_rates(optimizer))

    best_macro_f1 = float("-inf")
    best_macro_auc = float("-inf")
    best_tuned_macro_f1 = float("-inf")
    start_epoch = 0
    previous_cumulative_time = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer_loaded = False
        if not args.new_run_from_resume:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                optimizer_loaded = True
            except ValueError as exc:
                logger.warning(
                    "optimizer_state_not_loaded reason=%s; continuing with fresh optimizer state. "
                    "This is expected when changing trainable parameter groups.",
                    exc,
                )
        else:
            logger.info("new_run_from_resume=true; checkpoint optimizer state ignored.")
        if optimizer_loaded and scheduler is not None and checkpoint.get("scheduler") is not None:
            try:
                scheduler.load_state_dict(checkpoint["scheduler"])
            except ValueError as exc:
                logger.warning("scheduler_state_not_loaded reason=%s; continuing with fresh scheduler state.", exc)
        start_epoch = 0 if args.new_run_from_resume else int(checkpoint.get("epoch", 0))
        if scheduler is not None and checkpoint.get("scheduler") is None and not args.new_run_from_resume:
            align_epoch_scheduler(scheduler, start_epoch)
            logger.info("scheduler_aligned_to_start_epoch=%d lrs=%s", start_epoch, get_learning_rates(optimizer))
        if metrics_path.exists() and not args.new_run_from_resume:
            records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if records:
                best_macro_f1 = max(record["valid"]["macro_f1"] for record in records)
                best_macro_auc = max(record["valid"].get("macro_auc", float("-inf")) for record in records)
                best_tuned_macro_f1 = max(record["valid"].get("tuned_macro_f1", float("-inf")) for record in records)
                previous_cumulative_time = float(records[-1].get("cumulative_time_sec", 0.0))
        logger.info(
            "resumed_from=%s start_epoch=%d best_macro_f1=%.6f best_macro_auc=%.6f best_tuned_macro_f1=%.6f previous_cumulative_time_sec=%.1f",
            args.resume,
            start_epoch,
            best_macro_f1,
            best_macro_auc,
            best_tuned_macro_f1,
            previous_cumulative_time,
        )
        logger.info("resume_lrs=%s", get_learning_rates(optimizer))

    train_start = time.time()
    try:
        for epoch in range(start_epoch, config["epochs"]):
            epoch_id = epoch + 1
            logger.info("epoch=%d/%d started", epoch_id, config["epochs"])
            epoch_start = time.time()
            train_metrics = run_epoch(model, train_loader, optimizer, config, device, logger, epoch_id, disease_pos_weight)
            valid_metrics = run_epoch(model, valid_loader, None, config, device, logger, epoch_id, disease_pos_weight)
            epoch_time = time.time() - epoch_start
            cumulative_time = previous_cumulative_time + (time.time() - train_start)
            avg_epoch_time = cumulative_time / epoch_id
            remaining_epochs = config["epochs"] - epoch_id
            eta_seconds = remaining_epochs * avg_epoch_time
            epoch_learning_rates = get_learning_rates(optimizer)
            step_scheduler(scheduler, optimizer, valid_metrics)
            next_learning_rates = get_learning_rates(optimizer)
            record = {
                "epoch": epoch_id,
                "train": train_metrics,
                "valid": valid_metrics,
                "learning_rates": epoch_learning_rates,
                "next_learning_rates": next_learning_rates,
                "epoch_time_sec": epoch_time,
                "cumulative_time_sec": cumulative_time,
                "eta_remaining_sec": eta_seconds,
            }
            append_jsonl(metrics_path, record)
            if scheduler is not None:
                logger.info("epoch=%d scheduler_step lrs=%s", epoch_id, next_learning_rates)
            logger.info("epoch_record=%s", json.dumps(record, ensure_ascii=False, sort_keys=True))

            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "config": config,
                "epoch": epoch_id,
            }
            torch.save(state, output_dir / "last.pt")
            if valid_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = valid_metrics["macro_f1"]
                torch.save(state, output_dir / "best.pt")
                logger.info("epoch=%d new_best_macro_f1=%.6f", epoch_id, best_macro_f1)
            if valid_metrics.get("macro_auc", float("-inf")) > best_macro_auc:
                best_macro_auc = valid_metrics["macro_auc"]
                torch.save(state, output_dir / "best_auc.pt")
                write_json(
                    output_dir / "best_auc_thresholds.json",
                    {
                        "epoch": epoch_id,
                        "macro_auc": best_macro_auc,
                        "tuned_macro_f1": valid_metrics.get("tuned_macro_f1"),
                        "tuned_mean_accuracy": valid_metrics.get("tuned_mean_accuracy"),
                        "best_thresholds": valid_metrics.get("best_thresholds"),
                        "per_class_auc": valid_metrics.get("per_class_auc"),
                        "per_class_f1_at_0_5": valid_metrics.get("per_class_f1_at_0_5"),
                        "per_class_best_threshold": valid_metrics.get("per_class_best_threshold"),
                        "per_class_tuned_f1": valid_metrics.get("per_class_tuned_f1"),
                        "per_class_predicted_positives_at_0_5": valid_metrics.get("per_class_predicted_positives_at_0_5"),
                        "per_class_predicted_positives_at_tuned": valid_metrics.get("per_class_predicted_positives_at_tuned"),
                    },
                )
                logger.info("epoch=%d new_best_macro_auc=%.6f", epoch_id, best_macro_auc)
            if valid_metrics.get("tuned_macro_f1", float("-inf")) > best_tuned_macro_f1:
                best_tuned_macro_f1 = valid_metrics["tuned_macro_f1"]
                torch.save(state, output_dir / "best_tuned_f1.pt")
                write_json(
                    output_dir / "best_tuned_f1_thresholds.json",
                    {
                        "epoch": epoch_id,
                        "macro_auc": valid_metrics.get("macro_auc"),
                        "tuned_macro_f1": best_tuned_macro_f1,
                        "tuned_mean_accuracy": valid_metrics.get("tuned_mean_accuracy"),
                        "best_thresholds": valid_metrics.get("best_thresholds"),
                        "per_class_auc": valid_metrics.get("per_class_auc"),
                        "per_class_f1_at_0_5": valid_metrics.get("per_class_f1_at_0_5"),
                        "per_class_best_threshold": valid_metrics.get("per_class_best_threshold"),
                        "per_class_tuned_f1": valid_metrics.get("per_class_tuned_f1"),
                        "per_class_predicted_positives_at_0_5": valid_metrics.get("per_class_predicted_positives_at_0_5"),
                        "per_class_predicted_positives_at_tuned": valid_metrics.get("per_class_predicted_positives_at_tuned"),
                    },
                )
                logger.info("epoch=%d new_best_tuned_macro_f1=%.6f", epoch_id, best_tuned_macro_f1)

        sample_batch = next(iter(valid_loader))
        sample = {key: value[0] if torch.is_tensor(value) else value[0] for key, value in sample_batch.items()}
        save_grad_cam_sample(model, sample, device, output_dir / "gradcam_sample.png", config["image_size"])
    except Exception:
        logger.exception("training_failed")
        raise

    logger.info("artifacts_saved_to=%s", output_dir)


if __name__ == "__main__":
    main()
