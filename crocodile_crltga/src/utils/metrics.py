from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def multilabel_macro_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1.0 - targets)).sum(dim=0)
    fn = ((1.0 - preds) * targets).sum(dim=0)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return f1.mean().item()


def multilabel_mean_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= threshold).float()
    return (preds == targets.float()).float().mean().item()


def binary_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def multilabel_macro_auc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels = targets.detach().cpu().numpy()
    aucs: list[float] = []
    for index in range(labels.shape[1]):
        y_true = labels[:, index]
        if np.unique(y_true).size < 2:
            continue
        aucs.append(float(roc_auc_score(y_true, probs[:, index])))
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def multilabel_tuned_thresholds(logits: torch.Tensor, targets: torch.Tensor, num_thresholds: int = 101) -> dict:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels = targets.detach().cpu().numpy().astype(np.int32)
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    best_thresholds = []
    tuned_preds = np.zeros_like(labels)

    for index in range(labels.shape[1]):
        y_true = labels[:, index]
        if np.unique(y_true).size < 2:
            best_thresholds.append(0.5)
            tuned_preds[:, index] = (probs[:, index] >= 0.5).astype(np.int32)
            continue

        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in thresholds:
            preds = (probs[:, index] >= threshold).astype(np.int32)
            score = float(f1_score(y_true, preds, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        best_thresholds.append(best_threshold)
        tuned_preds[:, index] = (probs[:, index] >= best_threshold).astype(np.int32)

    macro_f1 = float(f1_score(labels, tuned_preds, average="macro", zero_division=0))
    mean_accuracy = float((tuned_preds == labels).mean())
    return {
        "tuned_macro_f1": macro_f1,
        "tuned_mean_accuracy": mean_accuracy,
        "best_thresholds": best_thresholds,
    }


def multilabel_per_class_diagnostics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_names: list[str],
    threshold: float = 0.5,
    num_thresholds: int = 101,
) -> dict:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels = targets.detach().cpu().numpy().astype(np.int32)
    default_preds = (probs >= threshold).astype(np.int32)
    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    aucs: dict[str, float | None] = {}
    f1_at_default: dict[str, float] = {}
    best_thresholds: dict[str, float] = {}
    tuned_f1: dict[str, float] = {}
    predicted_positives_at_default: dict[str, int] = {}
    predicted_positives_at_tuned: dict[str, int] = {}

    for index, label_name in enumerate(label_names):
        y_true = labels[:, index]
        y_prob = probs[:, index]
        y_pred_default = default_preds[:, index]

        if np.unique(y_true).size < 2:
            aucs[label_name] = None
        else:
            aucs[label_name] = float(roc_auc_score(y_true, y_prob))

        f1_at_default[label_name] = float(f1_score(y_true, y_pred_default, zero_division=0))
        predicted_positives_at_default[label_name] = int(y_pred_default.sum())

        best_threshold = 0.5
        best_f1 = -1.0
        best_pred = y_pred_default
        for candidate in thresholds:
            y_pred = (y_prob >= candidate).astype(np.int32)
            score = float(f1_score(y_true, y_pred, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_threshold = float(candidate)
                best_pred = y_pred

        best_thresholds[label_name] = best_threshold
        tuned_f1[label_name] = best_f1
        predicted_positives_at_tuned[label_name] = int(best_pred.sum())

    return {
        "per_class_auc": aucs,
        "per_class_f1_at_0_5": f1_at_default,
        "per_class_best_threshold": best_thresholds,
        "per_class_tuned_f1": tuned_f1,
        "per_class_predicted_positives_at_0_5": predicted_positives_at_default,
        "per_class_predicted_positives_at_tuned": predicted_positives_at_tuned,
    }
