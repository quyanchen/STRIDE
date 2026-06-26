import argparse
import csv
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


Row = Tuple[str, int, str]
DBID_PATTERN = re.compile(r"^DB\d+$")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sanitize_text(value: object) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for _ in handle)


def read_drug_list(path: Path) -> List[Tuple[str, str]]:
    drugs: List[Tuple[str, str]] = []
    seen = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            dbid, smiles = row[0].strip(), row[1].strip()
            if dbid.lower() in {"dbid", "drugbank_id", "drug_id", "drug"}:
                continue
            if not dbid or dbid in seen:
                continue
            seen.add(dbid)
            drugs.append((dbid, smiles))
    return drugs


def read_pairs(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            d1 = str(row.get("d1", "")).strip()
            d2 = str(row.get("d2", "")).strip()
            label_raw = row.get("type", row.get("label", ""))
            if not d1 or not d2 or str(label_raw).strip() == "":
                continue
            rows.append((d1, int(label_raw), d2))
    if not rows:
        raise ValueError(f"No valid DDI rows loaded from {path}")
    return rows


def write_pairs(path: Path, rows: Iterable[Row]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["d1", "type", "d2"])
        writer.writeheader()
        for d1, relation, d2 in rows:
            writer.writerow({"d1": d1, "type": int(relation), "d2": d2})


def read_relation_family(path: Path) -> Dict[int, Tuple[int, str]]:
    mapping: Dict[int, Tuple[int, str]] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_relation = row.get("relation_id", "")
            if str(raw_relation).strip() == "":
                continue
            relation = int(raw_relation)
            family_name = str(
                row.get("family_name", row.get("mechanism_family", "unknown_mixed"))
            ).strip() or "unknown_mixed"
            try:
                family_id = int(row.get("family_id", 0))
            except (TypeError, ValueError):
                family_id = 0
            mapping[relation] = (family_id, family_name)
    return mapping


def write_default_relation_family(path: Path, labels: Iterable[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["relation_id", "family_id", "family_name"]
        )
        writer.writeheader()
        for relation in sorted(set(int(label) for label in labels)):
            writer.writerow(
                {
                    "relation_id": relation,
                    "family_id": 0,
                    "family_name": "unknown_mixed",
                }
            )


def write_remapped_relation_family(
    path: Path,
    old_to_new: Dict[int, int],
    source_family: Dict[int, Tuple[int, str]],
) -> None:
    family_name_to_id: Dict[str, int] = {}
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["relation_id", "family_id", "family_name", "source_relation_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for old_relation, new_relation in sorted(
            old_to_new.items(), key=lambda item: item[1]
        ):
            family_name = source_family.get(old_relation, (0, "unknown_mixed"))[1]
            if family_name not in family_name_to_id:
                family_name_to_id[family_name] = len(family_name_to_id)
            writer.writerow(
                {
                    "relation_id": new_relation,
                    "family_id": family_name_to_id[family_name],
                    "family_name": family_name,
                    "source_relation_id": old_relation,
                }
            )


def write_label_mapping(
    path: Path, old_to_new: Dict[int, int], original_counts: Counter
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_relation_id", "relation_id", "original_count"],
        )
        writer.writeheader()
        for old_relation, new_relation in sorted(
            old_to_new.items(), key=lambda item: item[1]
        ):
            writer.writerow(
                {
                    "source_relation_id": old_relation,
                    "relation_id": new_relation,
                    "original_count": int(original_counts[old_relation]),
                }
            )


def prepare_primekg(
    primekg_dir: Path,
    output_dir: Path,
    keep_drug_drug: bool = False,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    nodes_csv = primekg_dir / "nodes.csv"
    kg_csv = primekg_dir / "kg.csv"
    if not nodes_csv.exists() or not kg_csv.exists():
        raise FileNotFoundError(
            f"PrimeKG requires nodes.csv and kg.csv under {primekg_dir}"
        )

    ensure_dir(output_dir)
    nodes_tsv = output_dir / "nodes.tsv"
    edges_tsv = output_dir / "edges.tsv"
    node_type_to_id: Dict[str, int] = {}
    node_type_counts = Counter()
    dbid_to_node: Dict[str, int] = {}

    with nodes_csv.open("r", encoding="utf-8-sig", newline="") as source, nodes_tsv.open(
        "w", encoding="utf-8"
    ) as target:
        for row in csv.DictReader(source):
            node_index = int(row["node_index"])
            node_type = sanitize_text(row.get("node_type", "unknown")) or "unknown"
            node_source = sanitize_text(row.get("node_source", "unknown")) or "unknown"
            node_id = sanitize_text(row.get("node_id", ""))
            node_name = sanitize_text(row.get("node_name", ""))
            if node_type not in node_type_to_id:
                node_type_to_id[node_type] = len(node_type_to_id)
            type_id = node_type_to_id[node_type]
            node_type_counts[node_type] += 1
            target.write(
                f"{node_index}\t"
                f"{node_type}::{node_source}::{node_id}::{node_name}\t{type_id}\n"
            )
            if node_source == "DrugBank" and DBID_PATTERN.match(node_id):
                dbid_to_node.setdefault(node_id, node_index)

    relation_to_id: Dict[str, int] = {}
    relation_counts = Counter()
    total_edges = 0
    dropped_drug_drug = 0
    with kg_csv.open("r", encoding="utf-8-sig", newline="") as source, edges_tsv.open(
        "w", encoding="utf-8"
    ) as target:
        for row in csv.DictReader(source):
            total_edges += 1
            source_type = sanitize_text(row.get("x_type", ""))
            target_type = sanitize_text(row.get("y_type", ""))
            if (
                not keep_drug_drug
                and source_type == "drug"
                and target_type == "drug"
            ):
                dropped_drug_drug += 1
                continue
            relation_name = sanitize_text(row.get("relation", "unknown")) or "unknown"
            if relation_name not in relation_to_id:
                relation_to_id[relation_name] = len(relation_to_id)
            relation_id = relation_to_id[relation_name]
            head, tail = int(row["x_index"]), int(row["y_index"])
            target.write(f"{head}\t{relation_id}\t{tail}\n")
            relation_counts[relation_name] += 1

            for prefix, node_index in (("x", head), ("y", tail)):
                source_name = sanitize_text(row.get(f"{prefix}_source", ""))
                source_id = sanitize_text(row.get(f"{prefix}_id", ""))
                if source_name == "DrugBank" and DBID_PATTERN.match(source_id):
                    dbid_to_node.setdefault(source_id, node_index)

    with (output_dir / "relation_map.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["relation_id", "relation_name", "edge_count"])
        for name, relation_id in sorted(
            relation_to_id.items(), key=lambda item: item[1]
        ):
            writer.writerow([relation_id, name, relation_counts[name]])

    with (output_dir / "node_type_map.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["node_type_id", "node_type_name", "node_count"])
        for name, type_id in sorted(node_type_to_id.items(), key=lambda item: item[1]):
            writer.writerow([type_id, name, node_type_counts[name]])

    stats = {
        "nodes": count_lines(nodes_tsv),
        "edges": count_lines(edges_tsv),
        "drugbank_nodes": len(dbid_to_node),
        "kg_rows_total": total_edges,
        "kg_rows_dropped_drug_drug": dropped_drug_drug,
    }
    return dbid_to_node, stats


def prepare_ddi_bundle(
    dataset: str,
    source_dir: Path,
    output_dir: Path,
    dbid_to_kg: Dict[str, int],
) -> Dict[str, object]:
    dataset = dataset.lower()
    drug_file = source_dir / (
        "drug_listxiao.csv" if dataset == "deng" else "drug_smiles.csv"
    )
    all_pairs_file = source_dir / "newddixiao-1.csv"
    split_names = {
        "deng": {
            "train": "ddi_training1xiao.csv",
            "val": "ddi_validation1xiao.csv",
            "test": "ddi_test1xiao.csv",
        },
        "ryu": {
            "train": "ddi_training1.csv",
            "val": "ddi_validation1.csv",
            "test": "ddi_test1.csv",
        },
    }[dataset]
    if not drug_file.exists() or not all_pairs_file.exists():
        raise FileNotFoundError(
            f"{dataset} requires {drug_file.name} and newddixiao-1.csv under {source_dir}"
        )

    ensure_dir(output_dir)
    drugs = read_drug_list(drug_file)
    local_by_dbid = {dbid: index for index, (dbid, _) in enumerate(drugs)}
    with (output_dir / "drug_listxiao.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        csv.writer(handle).writerows(drugs)
    with (output_dir / "smiles.tsv").open("w", encoding="utf-8") as handle:
        for index, (_, smiles) in enumerate(drugs):
            handle.write(f"{index}\t{smiles}\n")

    all_pairs = read_pairs(all_pairs_file)
    write_pairs(output_dir / "newddixiao-1.csv", all_pairs)
    with (output_dir / "ddi.tsv").open("w", encoding="utf-8") as handle:
        for d1, relation, d2 in all_pairs:
            if d1 in local_by_dbid and d2 in local_by_dbid:
                handle.write(
                    f"{local_by_dbid[d1]}\t{local_by_dbid[d2]}\t{relation}\n"
                )

    fold_sizes: Dict[str, Dict[str, int]] = {}
    output_names = {
        "train": ("ddi_training1xiao.csv", "ddi_training1.csv"),
        "val": ("ddi_validation1xiao.csv", "ddi_validation1.csv"),
        "test": ("ddi_test1xiao.csv", "ddi_test1.csv"),
    }
    for fold in range(5):
        fold_dir = output_dir / str(fold)
        ensure_dir(fold_dir)
        fold_sizes[str(fold)] = {}
        for split, source_name in split_names.items():
            rows = read_pairs(source_dir / str(fold) / source_name)
            for output_name in output_names[split]:
                write_pairs(fold_dir / output_name, rows)
            fold_sizes[str(fold)][split] = len(rows)

    labels = [relation for _, relation, _ in all_pairs]
    source_family = source_dir / "relation_family.csv"
    if source_family.exists():
        shutil.copy2(source_family, output_dir / "relation_family.csv")
    else:
        write_default_relation_family(output_dir / "relation_family.csv", labels)

    mapped = sum(1 for dbid, _ in drugs if dbid in dbid_to_kg)
    with (output_dir / "drug_to_kg.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["local_id", "drugbank_id", "kg_node_id", "mapped"])
        for local_id, (dbid, _) in enumerate(drugs):
            kg_id = dbid_to_kg.get(dbid, -1)
            writer.writerow([local_id, dbid, kg_id, int(kg_id >= 0)])

    return {
        "num_drugs": len(drugs),
        "num_pairs_all": len(all_pairs),
        "num_labels": len(set(labels)),
        "fold_sizes": fold_sizes,
        "drug_mapping": {
            "total": len(drugs),
            "mapped": mapped,
            "missing": len(drugs) - mapped,
            "ratio": round(100.0 * mapped / max(1, len(drugs)), 4),
        },
    }


def prepare_bundle(args: argparse.Namespace, root: Path) -> Path:
    output_root = resolve(root, args.output_root)
    primekg_dir = resolve(root, args.primekg_dir)
    ddi_dir = resolve(root, args.ddi_dir)
    bundle_dir = output_root / f"primekg+{args.dataset}"
    kg_dir, bundle_ddi_dir = bundle_dir / "kg", bundle_dir / "ddi"
    ensure_dir(bundle_dir)

    dbid_to_kg, kg_stats = prepare_primekg(
        primekg_dir,
        kg_dir,
        keep_drug_drug=args.keep_primekg_drug_drug,
    )
    ddi_stats = prepare_ddi_bundle(
        args.dataset, ddi_dir, bundle_ddi_dir, dbid_to_kg
    )
    summary = {
        "kg": "primekg",
        "ddi": args.dataset,
        "kg_source": str(primekg_dir),
        "ddi_source": str(ddi_dir),
        "drug_drug_edges_removed": not args.keep_primekg_drug_drug,
        "kg_stats": kg_stats,
        "ddi_stats": ddi_stats,
    }
    with (bundle_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return bundle_dir


def count_labels(rows: Iterable[Row]) -> Counter:
    return Counter(relation for _, relation, _ in rows)


def allocate_holdout_counts(
    total: int,
    val_ratio: float,
    test_ratio: float,
    min_val: int,
    min_test: int,
    min_train: int,
) -> Tuple[int, int]:
    if total < min_val + min_test + min_train:
        raise ValueError(f"Class with {total} rows cannot satisfy split minima")
    n_val = max(min_val, int(round(total * val_ratio)))
    n_test = max(min_test, int(round(total * test_ratio)))
    overflow = n_val + n_test + min_train - total
    if overflow > 0:
        reduction = min(overflow, max(0, n_test - min_test))
        n_test -= reduction
        overflow -= reduction
    if overflow > 0:
        reduction = min(overflow, max(0, n_val - min_val))
        n_val -= reduction
        overflow -= reduction
    if overflow > 0:
        raise ValueError(f"Unable to allocate holdouts for class with {total} rows")
    return n_val, n_test


def split_by_class(
    rows: List[Row],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    min_val: int,
    min_test: int,
    min_train: int,
) -> Tuple[Dict[int, List[Row]], List[Row], List[Row]]:
    rng = random.Random(seed)
    grouped: Dict[int, List[Row]] = defaultdict(list)
    for row in rows:
        grouped[row[1]].append(row)
    train_pool: Dict[int, List[Row]] = {}
    validation: List[Row] = []
    test: List[Row] = []
    for relation in sorted(grouped):
        class_rows = list(grouped[relation])
        rng.shuffle(class_rows)
        n_val, n_test = allocate_holdout_counts(
            len(class_rows), val_ratio, test_ratio, min_val, min_test, min_train
        )
        validation.extend(class_rows[:n_val])
        test.extend(class_rows[n_val : n_val + n_test])
        train_pool[relation] = class_rows[n_val + n_test :]
    rng.shuffle(validation)
    rng.shuffle(test)
    return train_pool, validation, test


def build_longtail_train(
    train_pool: Dict[int, List[Row]],
    imbalance_factor: float,
    min_tail_train: int,
    max_head_train: int,
    seed: int,
) -> Tuple[List[Row], Dict[int, int], List[int]]:
    rng = random.Random(seed)
    ranked_relations = sorted(
        train_pool, key=lambda relation: (-len(train_pool[relation]), relation)
    )
    if not ranked_relations:
        raise ValueError("No classes available for long-tail construction")
    if max_head_train > 0:
        head_count = min(max_head_train, len(train_pool[ranked_relations[0]]))
    else:
        head_count = min(
            int(round(imbalance_factor * min_tail_train)),
            len(train_pool[ranked_relations[0]]),
        )
    head_count = max(1, head_count)
    selected: List[Row] = []
    counts: Dict[int, int] = {}
    class_num = len(ranked_relations)
    for rank, relation in enumerate(ranked_relations):
        available = list(train_pool[relation])
        rng.shuffle(available)
        target = (
            head_count
            if class_num == 1
            else head_count
            * math.pow(imbalance_factor, -float(rank) / float(class_num - 1))
        )
        keep = min(len(available), max(min_tail_train, int(round(target))))
        counts[relation] = keep
        selected.extend(available[:keep])
    rng.shuffle(selected)
    return selected, counts, ranked_relations


def shot_group(count: int, many_threshold: int, medium_threshold: int) -> str:
    if count >= many_threshold:
        return "many"
    if count >= medium_threshold:
        return "medium"
    return "few"


def write_relation_frequency(
    path: Path,
    ranked_relations: List[int],
    original_counts: Counter,
    pool_counts: Counter,
    train_counts: Dict[int, int],
    validation_counts: Counter,
    test_counts: Counter,
    many_threshold: int,
    medium_threshold: int,
) -> None:
    fields = [
        "rank_by_train_pool",
        "relation_id",
        "original_count",
        "train_pool_count",
        "longtail_train_count",
        "val_count",
        "test_count",
        "shot_group",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, relation in enumerate(ranked_relations):
            train_count = int(train_counts[relation])
            writer.writerow(
                {
                    "rank_by_train_pool": rank,
                    "relation_id": relation,
                    "original_count": int(original_counts[relation]),
                    "train_pool_count": int(pool_counts[relation]),
                    "longtail_train_count": train_count,
                    "val_count": int(validation_counts[relation]),
                    "test_count": int(test_counts[relation]),
                    "shot_group": shot_group(
                        train_count, many_threshold, medium_threshold
                    ),
                }
            )


def prepare_longtail(args: argparse.Namespace, root: Path) -> Path:
    pair_path = resolve(root, args.pairs)
    family_path = resolve(root, args.relation_family)
    output_root = resolve(root, args.longtail_output_root) / args.dataset
    ensure_dir(output_root)

    rows = read_pairs(pair_path)
    original_counts = count_labels(rows)
    kept = sorted(
        relation
        for relation, count in original_counts.items()
        if count >= args.min_class_count
    )
    dropped = sorted(set(original_counts) - set(kept))
    old_to_new = {old: new for new, old in enumerate(kept)}
    remapped = [
        (d1, old_to_new[relation], d2)
        for d1, relation, d2 in rows
        if relation in old_to_new
    ]
    remapped_counts = count_labels(remapped)
    train_pool, validation, test = split_by_class(
        remapped,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_val=args.min_val_per_class,
        min_test=args.min_test_per_class,
        min_train=args.min_train_per_class,
    )
    longtail_train, train_counts, ranked = build_longtail_train(
        train_pool,
        imbalance_factor=args.imbalance_factor,
        min_tail_train=args.min_tail_train,
        max_head_train=args.max_head_train,
        seed=args.seed + int(round(args.imbalance_factor * 100)),
    )
    variant = (
        str(int(args.imbalance_factor))
        if float(args.imbalance_factor).is_integer()
        else str(args.imbalance_factor)
    )
    output_dir = output_root / f"LT{variant}_seed{args.seed}"
    ensure_dir(output_dir)
    write_pairs(output_dir / "ddi_training1.csv", longtail_train)
    write_pairs(output_dir / "ddi_validation1.csv", validation)
    write_pairs(output_dir / "ddi_test1.csv", test)
    write_label_mapping(output_dir / "label_mapping.csv", old_to_new, original_counts)
    write_remapped_relation_family(
        output_dir / "relation_family.csv",
        old_to_new,
        read_relation_family(family_path),
    )
    pool_counts = Counter({relation: len(items) for relation, items in train_pool.items()})
    write_relation_frequency(
        output_dir / "relation_frequency.csv",
        ranked,
        remapped_counts,
        pool_counts,
        train_counts,
        count_labels(validation),
        count_labels(test),
        args.many_threshold,
        args.medium_threshold,
    )
    values = list(train_counts.values())
    group_counts = Counter(
        shot_group(count, args.many_threshold, args.medium_threshold)
        for count in values
    )
    summary = {
        "dataset": args.dataset,
        "source_pair_path": str(pair_path),
        "imbalance_factor": args.imbalance_factor,
        "seed": args.seed,
        "class_num": len(kept),
        "dropped_source_relation_ids": dropped,
        "train_rows": len(longtail_train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "max_train_class_count": max(values),
        "min_train_class_count": min(values),
        "actual_train_imbalance": max(values) / max(1, min(values)),
        "min_tail_train": args.min_tail_train,
        "max_head_train": args.max_head_train,
        "many_threshold": args.many_threshold,
        "medium_threshold": args.medium_threshold,
        "shot_group_class_counts": dict(group_counts),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return output_dir


def add_bundle_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=["deng", "ryu"], required=True)
    parser.add_argument("--primekg_dir", default="data/PrimeKG")
    parser.add_argument("--ddi_dir", required=True)
    parser.add_argument("--output_root", default="data")
    parser.add_argument("--keep_primekg_drug_drug", action="store_true")


def add_longtail_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--longtail_output_root", default="data/controlled_longtail")
    parser.add_argument("--imbalance_factor", type=float, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_class_count", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--min_val_per_class", type=int, default=1)
    parser.add_argument("--min_test_per_class", type=int, default=1)
    parser.add_argument("--min_train_per_class", type=int, default=3)
    parser.add_argument("--min_tail_train", type=int, default=3)
    parser.add_argument("--max_head_train", type=int, default=0)
    parser.add_argument("--many_threshold", type=int, default=100)
    parser.add_argument("--medium_threshold", type=int, default=20)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare STRIDE PrimeKG bundles and controlled long-tail splits."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bundle = subparsers.add_parser("bundle", help="Build a filtered PrimeKG+DDI bundle.")
    add_bundle_arguments(bundle)

    longtail = subparsers.add_parser(
        "longtail", help="Build an exponential controlled long-tail split."
    )
    longtail.add_argument("--dataset", choices=["deng", "ryu"], required=True)
    longtail.add_argument("--pairs", default=None)
    longtail.add_argument("--relation_family", default=None)
    add_longtail_options(longtail)

    combined = subparsers.add_parser("all", help="Run bundle then longtail preparation.")
    add_bundle_arguments(combined)
    add_longtail_options(combined)
    return parser


def apply_longtail_defaults(args: argparse.Namespace) -> None:
    if args.pairs is None:
        args.pairs = f"data/primekg+{args.dataset}/ddi/newddixiao-1.csv"
    if args.relation_family is None:
        args.relation_family = f"data/primekg+{args.dataset}/ddi/relation_family.csv"


def main() -> None:
    root = Path(__file__).resolve().parent
    args = build_parser().parse_args()
    if args.command == "bundle":
        prepare_bundle(args, root)
    elif args.command == "longtail":
        apply_longtail_defaults(args)
        prepare_longtail(args, root)
    else:
        bundle_dir = prepare_bundle(args, root)
        args.pairs = str(bundle_dir / "ddi" / "newddixiao-1.csv")
        args.relation_family = str(bundle_dir / "ddi" / "relation_family.csv")
        prepare_longtail(args, root)


if __name__ == "__main__":
    main()
