from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score


def multi_class_eval(labels: np.ndarray, pred_prob: np.ndarray):
    pred = pred_prob.argmax(axis=1)
    class_labels = np.arange(pred_prob.shape[1])
    acc = accuracy_score(labels, pred) * 100.0
    f1 = (
        f1_score(
            labels,
            pred,
            labels=class_labels,
            average="macro",
            zero_division=0,
        )
        * 100.0
    )
    precision = (
        precision_score(
            labels,
            pred,
            labels=class_labels,
            average="macro",
            zero_division=0,
        )
        * 100.0
    )
    recall = (
        recall_score(
            labels,
            pred,
            labels=class_labels,
            average="macro",
            zero_division=0,
        )
        * 100.0
    )
    observed_labels = np.unique(np.concatenate([labels, pred]))
    if observed_labels.size < 2:
        kappa = 0.0
    else:
        kappa = cohen_kappa_score(labels, pred, labels=class_labels) * 100.0
        if not np.isfinite(kappa):
            kappa = 0.0
    return {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "kappa": kappa,
    }


def longtail_group_eval(
    labels: np.ndarray,
    pred_prob: np.ndarray,
    class_train_counts: np.ndarray,
    many_threshold: int = 100,
    medium_threshold: int = 20,
    grouping: str = "fixed",
):
    counts = np.asarray(class_train_counts, dtype=np.float32).reshape(-1)
    num_classes = int(counts.shape[0])
    if num_classes == 0:
        return {}

    pred = pred_prob.argmax(axis=1)
    per_class_f1 = (
        f1_score(
            labels,
            pred,
            labels=np.arange(num_classes),
            average=None,
            zero_division=0,
        )
        * 100.0
    )

    grouping = str(grouping).lower().strip()
    if grouping == "tercile":
        order = np.lexsort((np.arange(num_classes), -counts))
        first_cut = int(np.ceil(num_classes / 3.0))
        second_cut = int(np.ceil(2.0 * num_classes / 3.0))
        many = order[:first_cut]
        medium = order[first_cut:second_cut]
        few = order[second_cut:]
    elif grouping == "fixed":
        many = np.where(counts >= float(many_threshold))[0]
        medium = np.where((counts < float(many_threshold)) & (counts >= float(medium_threshold)))[0]
        few = np.where(counts < float(medium_threshold))[0]
    else:
        raise ValueError(f"Unsupported long-tail grouping: {grouping}")

    out = {}
    for name, idx in [("many", many), ("medium", medium), ("few", few)]:
        if len(idx) == 0:
            out[f"{name}_f1"] = 0.0
            out[f"{name}_class_num"] = 0
        else:
            out[f"{name}_f1"] = float(per_class_f1[idx].mean())
            out[f"{name}_class_num"] = int(len(idx))

    tail = np.concatenate([medium, few], axis=0)
    out["tail_f1"] = float(per_class_f1[tail].mean()) if len(tail) > 0 else 0.0
    out["tail_class_num"] = int(len(tail))
    out["many_threshold"] = int(many_threshold)
    out["medium_threshold"] = int(medium_threshold)
    out["longtail_grouping"] = grouping
    return out


def _imbalance_loss(logits: torch.Tensor, labels: torch.Tensor, class_counts: torch.Tensor, tau: float = 1.0):
    prior = class_counts / class_counts.sum().clamp_min(1e-8)
    adjust = tau * torch.log(prior.clamp_min(1e-8))
    return F.cross_entropy(logits + adjust.unsqueeze(0), labels.long())


def _build_true_assignment(labels: torch.Tensor, num_relations: int):
    return F.one_hot(labels.long(), num_classes=num_relations).float()


def train_one_epoch_stride(
    model,
    loader,
    optimizer,
    device: torch.device,
    class_counts: torch.Tensor,
    lambda_align: float = 0.1,
    lambda_consistency: float = 0.1,
    lambda_imbalance: float = 0.2,
    lambda_energy: float = 0.2,
    lambda_stride_mi: float = 0.1,
    energy_margin: float = 0.1,
    logit_adjust_tau: float = 1.0,
    decode_steps: int = 3,
) -> Dict[str, float]:
    model.train()
    all_prob = []
    all_label = []
    loss_meter = 0.0
    sample_num = 0
    margin = torch.tensor(float(energy_margin), device=device)

    for batch in loader:
        mol1, kg1, mol2, kg2, y = batch
        mol1 = mol1.to(device)
        kg1 = kg1.to(device)
        mol2 = mol2.to(device)
        kg2 = kg2.to(device)
        y = y.long().to(device)

        outputs = model(
            mol1,
            kg1,
            mol2,
            kg2,
            label_type="multi_class",
            decode_steps=decode_steps,
            isolate_backward=True,
        )
        structured_logits = outputs["structured_logits"]

        pred_loss = F.cross_entropy(structured_logits, y)
        imbalance_loss = _imbalance_loss(
            structured_logits,
            y,
            class_counts=class_counts,
            tau=logit_adjust_tau,
        )
        y_true_assignment = _build_true_assignment(y, model.num_relations)
        true_energy = model.energy_from_assignment(
            y=y_true_assignment,
            logits=outputs["logits"],
            mol_view_logits=outputs["mol_view_logits"],
            kg_view_logits=outputs["kg_view_logits"],
        )["total"].mean()
        pred_energy = outputs["energy"]["total"].mean()
        energy_loss = F.relu(margin + true_energy - pred_energy)

        align_loss = outputs["align_loss"]
        consistency_loss = outputs["consistency_loss"]
        stride_mi_loss = outputs["mi_mol_loss"] + outputs["mi_kg_loss"]

        total_loss = (
            pred_loss
            + lambda_align * align_loss
            + lambda_consistency * consistency_loss
            + lambda_imbalance * imbalance_loss
            + lambda_energy * energy_loss
            + lambda_stride_mi * stride_mi_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        bsz = int(y.shape[0])
        sample_num += bsz
        loss_meter += float(total_loss.item()) * bsz
        all_prob.append(torch.softmax(structured_logits, dim=-1).detach().cpu().numpy())
        all_label.append(y.detach().cpu().numpy())

    all_prob = np.concatenate(all_prob, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    metrics = multi_class_eval(all_label, all_prob)
    metrics["loss"] = loss_meter / max(1, sample_num)
    return metrics


@torch.no_grad()
def evaluate_stride(
    model,
    loader,
    device: torch.device,
    class_counts: torch.Tensor,
    lambda_align: float = 0.1,
    lambda_consistency: float = 0.1,
    lambda_imbalance: float = 0.2,
    lambda_energy: float = 0.2,
    lambda_stride_mi: float = 0.1,
    energy_margin: float = 0.1,
    logit_adjust_tau: float = 1.0,
    decode_steps: int = 3,
    class_train_counts: np.ndarray = None,
    many_threshold: int = 100,
    medium_threshold: int = 20,
    longtail_grouping: str = "fixed",
) -> Dict[str, float]:
    model.eval()
    all_prob = []
    all_label = []
    loss_meter = 0.0
    sample_num = 0
    margin = torch.tensor(float(energy_margin), device=device)

    for batch in loader:
        mol1, kg1, mol2, kg2, y = batch
        mol1 = mol1.to(device)
        kg1 = kg1.to(device)
        mol2 = mol2.to(device)
        kg2 = kg2.to(device)
        y = y.long().to(device)

        outputs = model(
            mol1,
            kg1,
            mol2,
            kg2,
            label_type="multi_class",
            decode_steps=decode_steps,
            isolate_backward=True,
        )
        structured_logits = outputs["structured_logits"]

        pred_loss = F.cross_entropy(structured_logits, y)
        imbalance_loss = _imbalance_loss(
            structured_logits,
            y,
            class_counts=class_counts,
            tau=logit_adjust_tau,
        )
        y_true_assignment = _build_true_assignment(y, model.num_relations)
        true_energy = model.energy_from_assignment(
            y=y_true_assignment,
            logits=outputs["logits"],
            mol_view_logits=outputs["mol_view_logits"],
            kg_view_logits=outputs["kg_view_logits"],
        )["total"].mean()
        pred_energy = outputs["energy"]["total"].mean()
        energy_loss = F.relu(margin + true_energy - pred_energy)

        align_loss = outputs["align_loss"]
        consistency_loss = outputs["consistency_loss"]
        stride_mi_loss = outputs["mi_mol_loss"] + outputs["mi_kg_loss"]

        total_loss = (
            pred_loss
            + lambda_align * align_loss
            + lambda_consistency * consistency_loss
            + lambda_imbalance * imbalance_loss
            + lambda_energy * energy_loss
            + lambda_stride_mi * stride_mi_loss
        )

        bsz = int(y.shape[0])
        sample_num += bsz
        loss_meter += float(total_loss.item()) * bsz
        all_prob.append(torch.softmax(structured_logits, dim=-1).cpu().numpy())
        all_label.append(y.cpu().numpy())

    all_prob = np.concatenate(all_prob, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    metrics = multi_class_eval(all_label, all_prob)
    if class_train_counts is not None:
        metrics.update(
            longtail_group_eval(
                labels=all_label,
                pred_prob=all_prob,
                class_train_counts=class_train_counts,
                many_threshold=many_threshold,
                medium_threshold=medium_threshold,
                grouping=longtail_grouping,
            )
        )
    metrics["loss"] = loss_meter / max(1, sample_num)
    return metrics
