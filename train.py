import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import (
    STRIDEDataset,
    build_relation_statistics,
    build_stride_feature_caches,
    load_deng_drug_space,
    load_longtail_split,
    load_official_s1_folds,
    load_relation_family_override,
    load_source_relation_ids,
    stride_collate,
)
from engine import evaluate_stride, train_one_epoch_stride
from model import STRIDE


PRESETS = {
    "primekg+deng": {
        "data_root": "data/primekg+deng",
        "kg_dir": "data/primekg+deng/kg",
        "family_csv": "data/primekg+deng/ddi/relation_family.csv",
        "ddi_dir": "ddi",
        "kg_name": "kg",
        "longtail_root": "data/controlled_longtail/deng",
    },
    "primekg+ryu": {
        "data_root": "data/primekg+ryu",
        "kg_dir": "data/primekg+ryu/kg",
        "family_csv": "data/primekg+ryu/ddi/relation_family.csv",
        "ddi_dir": "ddi",
        "kg_name": "kg",
        "longtail_root": "data/controlled_longtail/ryu",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="STRIDE main and LT-100 experiments")
    parser.add_argument("--data_bundle", choices=sorted(PRESETS), default="primekg+deng")
    parser.add_argument("--protocol", choices=["standard", "longtail"], default="standard")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--kg_dir", default=None)
    parser.add_argument("--family_csv", default=None)
    parser.add_argument("--longtail_root", default=None)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--fold", type=int, choices=range(6), default=0, metavar="{0,1,2,3,4,5}", help="Official fold to run. Use 0-4 for one fold or 5 for all folds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--expert_hidden", type=int, default=256)
    parser.add_argument("--alignment_hidden", type=int, default=256)
    parser.add_argument("--alignment_chunk", type=int, default=8)
    parser.add_argument("--decode_steps", type=int, default=1)
    parser.add_argument("--kg_hops", type=int, default=2)
    parser.add_argument("--kg_frontier", type=int, default=32)
    parser.add_argument("--mutex_temperature", type=float, default=0.05)
    parser.add_argument("--mutex_quantile", type=float, default=0.25)
    parser.add_argument("--lambda_pairwise", type=float, default=0.0)
    parser.add_argument("--lambda_smooth", type=float, default=0.0)
    parser.add_argument("--lambda_view", type=float, default=0.1)
    parser.add_argument("--lambda_mutex", type=float, default=0.0)
    parser.add_argument("--mutex_warmup_epochs", type=int, default=20)
    parser.add_argument("--loss_align_weight", type=float, default=0.05)
    parser.add_argument("--loss_consistency_weight", type=float, default=0.05)
    parser.add_argument("--loss_imbalance_weight", type=float, default=0.2)
    parser.add_argument("--loss_energy_weight", type=float, default=0.20)
    parser.add_argument("--loss_mi_weight", type=float, default=0.10)
    parser.add_argument("--energy_margin", type=float, default=0.10)
    parser.add_argument("--logit_adjust_tau", type=float, default=1.0)
    parser.add_argument("--many_threshold", type=int, default=100)
    parser.add_argument("--medium_threshold", type=int, default=20)
    parser.add_argument("--longtail_grouping", choices=["fixed", "tercile"], default="fixed")
    parser.add_argument("--disable_alignment", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_args(args):
    nonnegative = [
        "mutex_temperature",
        "lambda_pairwise",
        "lambda_smooth",
        "lambda_view",
        "lambda_mutex",
        "loss_align_weight",
        "loss_consistency_weight",
        "loss_imbalance_weight",
        "loss_energy_weight",
        "loss_mi_weight",
        "energy_margin",
        "logit_adjust_tau",
    ]
    for name in nonnegative:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be nonnegative")
    if not 0.0 <= args.mutex_quantile <= 1.0:
        raise ValueError("--mutex_quantile must be in [0, 1]")
    if args.mutex_warmup_epochs < 0:
        raise ValueError("--mutex_warmup_epochs must be nonnegative")
    if args.medium_threshold < 0 or args.many_threshold < args.medium_threshold:
        raise ValueError(
            "--many_threshold must be at least --medium_threshold, "
            "and both must be nonnegative"
        )
    positive = [
        "epochs",
        "patience",
        "batch_size",
        "hidden_dim",
        "heads",
        "depth",
        "expert_hidden",
        "alignment_hidden",
        "alignment_chunk",
        "decode_steps",
        "kg_frontier",
        "learning_rate",
    ]
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.protocol == "longtail" and args.fold != 5:
        raise ValueError("--fold must be 5 for the longtail protocol")
    if args.kg_hops < 0:
        raise ValueError("--kg_hops must be nonnegative")
    if args.workers < 0:
        raise ValueError("--workers must be nonnegative")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay must be nonnegative")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.hidden_dim % args.heads != 0:
        raise ValueError("--hidden_dim must be divisible by --heads")


def resolve(value):
    return str(Path(value).expanduser().resolve())


def select_official_folds(folds, fold: int):
    if fold == 5:
        return list(enumerate(folds))
    if not 0 <= fold < len(folds):
        raise ValueError(
            f"Requested fold {fold}, but only {len(folds)} folds are available"
        )
    return [(fold, folds[fold])]


def compact_relation_labels(folds):
    labels = []
    for fold in folds:
        for split in fold:
            split = np.asarray(split, dtype=np.int64).reshape(-1, 3)
            if len(split) == 0:
                raise ValueError("Every train/validation/test split must be non-empty")
            labels.append(split[:, 2])
    relation_ids = np.unique(np.concatenate(labels)).astype(np.int64)
    if relation_ids.min() < 0:
        raise ValueError("Relation labels must be non-negative")

    lookup = np.full((int(relation_ids.max()) + 1,), -1, dtype=np.int64)
    lookup[relation_ids] = np.arange(len(relation_ids), dtype=np.int64)
    compact_folds = []
    for fold in folds:
        compact_splits = []
        for split in fold:
            compact = np.asarray(split, dtype=np.int64).reshape(-1, 3).copy()
            compact[:, 2] = lookup[compact[:, 2]]
            compact_splits.append(compact)
        compact_folds.append(tuple(compact_splits))
    return compact_folds, relation_ids


def summarize(results):
    keys = [
        "acc",
        "f1",
        "precision",
        "recall",
        "loss",
    ]
    if "few_f1" in results[0]:
        keys.extend(["few_f1", "medium_f1", "many_f1", "tail_f1"])
    return {
        key: {
            "mean": float(np.mean([result[key] for result in results])),
            "std": float(np.std([result[key] for result in results])),
        }
        for key in keys
    }


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preset = PRESETS[args.data_bundle]
    data_root = resolve(args.data_root or preset["data_root"])
    kg_dir = resolve(args.kg_dir or preset["kg_dir"])
    longtail_root = resolve(args.longtail_root or preset["longtail_root"])
    if args.family_csv:
        family_csv = resolve(args.family_csv)
    elif args.protocol == "longtail":
        family_csv = resolve(
            os.path.join(
                longtail_root,
                f"LT100_seed{args.seed}",
                "relation_family.csv",
            )
        )
    else:
        family_csv = resolve(preset["family_csv"])
    output_dir = (
        Path(args.output_dir).resolve()
        / args.protocol
        / f"fold_{args.fold}"
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles, drug_to_local, drug_to_kg = load_deng_drug_space(
        data_root=data_root,
        drkg_dataset_dir=preset["kg_name"],
        deng_dataset_dir=preset["ddi_dir"],
    )
    if args.protocol == "standard":
        all_folds = load_official_s1_folds(
            data_root=data_root,
            dbid_to_local=drug_to_local,
            fold_num=5,
            deng_dataset_dir=preset["ddi_dir"],
        )
        selected_folds = select_official_folds(all_folds, args.fold)
    else:
        longtail_folds = load_longtail_split(
            longtail_root=longtail_root,
            dbid_to_local=drug_to_local,
            variant="LT100",
            seed=args.seed,
            test_mode="full",
        )
        selected_folds = [(0, longtail_folds[0])]

    compact_folds, relation_ids = compact_relation_labels(
        [splits for _, splits in selected_folds]
    )
    selected_folds = [
        (fold_id, compact_splits)
        for (fold_id, _), compact_splits in zip(selected_folds, compact_folds)
    ]
    num_classes = len(relation_ids)
    if args.num_classes not in {0, num_classes}:
        raise ValueError(
            f"--num_classes={args.num_classes} conflicts with the "
            f"{num_classes} relation labels present in the selected splits"
        )
    relation_to_family, _ = load_relation_family_override(
        family_csv,
        num_classes,
        relation_ids=relation_ids,
    )
    source_relation_ids = load_source_relation_ids(family_csv, relation_ids)
    molecule_cache, kg_cache, feature_stats = build_stride_feature_caches(
        smiles_list=smiles,
        drug_to_kg_id=drug_to_kg,
        drkg_data_dir=kg_dir,
        khop=args.kg_hops,
        fixed_num=args.kg_frontier,
    )

    test_results = []
    for fold, (train_data, valid_data, test_data) in selected_folds:
        fold_seed = args.seed + fold
        set_seed(fold_seed)
        raw_counts = np.bincount(
            train_data[:, 2],
            minlength=num_classes,
        ).astype(np.float32)
        zero_shot_internal = np.flatnonzero(raw_counts == 0).astype(np.int64)
        zero_shot_source = source_relation_ids[zero_shot_internal]
        if len(zero_shot_internal) > 0:
            print(
                f"[warning] fold={fold} relations absent from training: "
                f"internal={zero_shot_internal.tolist()} "
                f"source={zero_shot_source.tolist()}"
            )
        relation_stats = build_relation_statistics(
            sample=train_data,
            num_relations=num_classes,
            drug_num=len(smiles),
            mutex_temperature=args.mutex_temperature,
            tau_quantile=args.mutex_quantile,
            relation_to_family_override=relation_to_family,
            device=device,
        )
        train_generator = torch.Generator()
        train_generator.manual_seed(fold_seed)
        loaders = []
        for samples, shuffle in [
            (train_data, True),
            (valid_data, False),
            (test_data, False),
        ]:
            loaders.append(
                DataLoader(
                    STRIDEDataset(samples, molecule_cache, kg_cache),
                    batch_size=args.batch_size,
                    shuffle=shuffle,
                    num_workers=args.workers,
                    collate_fn=stride_collate,
                    generator=train_generator if shuffle else None,
                )
            )

        model = STRIDE(
            num_features_drug=feature_stats["num_features_drug"],
            num_nodes=feature_stats["num_nodes"],
            num_relations_mol=feature_stats["num_rel_mol"],
            num_relations_graph=feature_stats["num_rel_graph"],
            max_degree_graph=feature_stats["max_degree_graph"],
            max_degree_node=feature_stats["max_degree_node"],
            class_num=num_classes,
            relation_to_family=relation_stats["relation_to_family"],
            relation_graph=relation_stats["relation_graph"],
            mutex_prior=relation_stats["mutex_prior"],
            max_layer=args.depth,
            hidden_dim=args.hidden_dim,
            num_heads=args.heads,
            expert_hidden=args.expert_hidden,
            align_hidden=args.alignment_hidden,
            decode_steps=args.decode_steps,
            align_chunk_size=args.alignment_chunk,
            dropout=args.dropout,
            lambda_pairwise=args.lambda_pairwise,
            lambda_smooth=args.lambda_smooth,
            lambda_consistency=args.lambda_view,
            lambda_mutex=args.lambda_mutex,
            enable_alignment=not args.disable_alignment,
        ).to(device)
        optimizer = Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        best_f1 = -1.0
        best_epoch = -1
        best_state = None
        stale = 0
        base_mutex = model.lambda_mutex
        for epoch in range(args.epochs):
            if args.mutex_warmup_epochs > 0:
                progress = min(
                    1.0,
                    float(epoch + 1) / float(args.mutex_warmup_epochs),
                )
                model.lambda_mutex = base_mutex * 0.5 * (
                    1.0 - np.cos(np.pi * progress)
                )
            else:
                model.lambda_mutex = base_mutex
            train_metrics = train_one_epoch_stride(
                model,
                loaders[0],
                optimizer,
                device,
                relation_stats["class_counts"],
                lambda_align=args.loss_align_weight,
                lambda_consistency=args.loss_consistency_weight,
                lambda_imbalance=args.loss_imbalance_weight,
                lambda_energy=args.loss_energy_weight,
                lambda_stride_mi=args.loss_mi_weight,
                energy_margin=args.energy_margin,
                logit_adjust_tau=args.logit_adjust_tau,
                decode_steps=args.decode_steps,
            )
            training_mutex = model.lambda_mutex
            model.lambda_mutex = base_mutex
            try:
                valid_metrics = evaluate_stride(
                    model,
                    loaders[1],
                    device,
                    relation_stats["class_counts"],
                    lambda_align=args.loss_align_weight,
                    lambda_consistency=args.loss_consistency_weight,
                    lambda_imbalance=args.loss_imbalance_weight,
                    lambda_energy=args.loss_energy_weight,
                    lambda_stride_mi=args.loss_mi_weight,
                    energy_margin=args.energy_margin,
                    logit_adjust_tau=args.logit_adjust_tau,
                    decode_steps=args.decode_steps,
                )
            finally:
                model.lambda_mutex = training_mutex
            print(
                f"fold={fold} epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"valid_f1={valid_metrics['f1']:.2f}"
            )
            if valid_metrics["f1"] > best_f1:
                best_f1 = valid_metrics["f1"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
                model.lambda_mutex = base_mutex
                try:
                    best_test_metrics = evaluate_stride(
                        model,
                        loaders[2],
                        device,
                        relation_stats["class_counts"],
                        lambda_align=args.loss_align_weight,
                        lambda_consistency=args.loss_consistency_weight,
                        lambda_imbalance=args.loss_imbalance_weight,
                        lambda_energy=args.loss_energy_weight,
                        lambda_stride_mi=args.loss_mi_weight,
                        energy_margin=args.energy_margin,
                        logit_adjust_tau=args.logit_adjust_tau,
                        decode_steps=args.decode_steps,
                        class_train_counts=(
                            raw_counts if args.protocol == "longtail" else None
                        ),
                        many_threshold=args.many_threshold,
                        medium_threshold=args.medium_threshold,
                        longtail_grouping=args.longtail_grouping,
                    )
                finally:
                    model.lambda_mutex = training_mutex
                print(
                    f"fold={fold} epoch={epoch:03d} best_test "
                    f"ACC={best_test_metrics['acc']:.2f} "
                    f"F1={best_test_metrics['f1']:.2f} "
                    f"REC={best_test_metrics['recall']:.2f} "
                    f"PRE={best_test_metrics['precision']:.2f}"
                )
            else:
                stale += 1
                if stale >= args.patience:
                    break

        model.load_state_dict(best_state)
        model.lambda_mutex = base_mutex
        test_metrics = evaluate_stride(
            model,
            loaders[2],
            device,
            relation_stats["class_counts"],
            lambda_align=args.loss_align_weight,
            lambda_consistency=args.loss_consistency_weight,
            lambda_imbalance=args.loss_imbalance_weight,
            lambda_energy=args.loss_energy_weight,
            lambda_stride_mi=args.loss_mi_weight,
            energy_margin=args.energy_margin,
            logit_adjust_tau=args.logit_adjust_tau,
            decode_steps=args.decode_steps,
            class_train_counts=raw_counts if args.protocol == "longtail" else None,
            many_threshold=args.many_threshold,
            medium_threshold=args.medium_threshold,
            longtail_grouping=args.longtail_grouping,
        )
        test_metrics["best_epoch"] = best_epoch
        test_metrics["fold"] = fold
        test_metrics["zero_shot_relation_ids"] = zero_shot_internal.tolist()
        test_metrics["zero_shot_source_relation_ids"] = zero_shot_source.tolist()
        test_results.append(test_metrics)
        torch.save(
            {
                "model_state_dict": best_state,
                "args": vars(args),
                "best_validation_f1": best_f1,
                "relation_ids": relation_ids.tolist(),
                "source_relation_ids": source_relation_ids.tolist(),
                "family_csv": family_csv,
                "fold_seed": fold_seed,
                "zero_shot_relation_ids": zero_shot_internal.tolist(),
                "zero_shot_source_relation_ids": zero_shot_source.tolist(),
            },
            output_dir / f"fold_{fold}.pt",
        )
        print(f"fold={fold} test={test_metrics}")

    summary = summarize(test_results)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": args.protocol,
                "relation_ids": relation_ids.tolist(),
                "source_relation_ids": source_relation_ids.tolist(),
                "family_csv": family_csv,
                "num_classes": num_classes,
                "feature_stats": feature_stats,
                "args": vars(args),
                "folds": test_results,
                "summary": summary,
            },
            handle,
            indent=2,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
