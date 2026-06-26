import csv
import hashlib
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import degree

from preprocess import calculate_shortest_path, k_hop_subgraph, single_smile_to_graph


FEATURE_CACHE_VERSION = 6


def _read_dbid_smiles(deng_drug_csv: str) -> Tuple[List[str], Dict[str, int]]:
    smiles = []
    dbid_to_local = {}
    with open(deng_drug_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            dbid = row[0].strip()
            smi = row[1].strip()
            if dbid.lower() in {"dbid", "drugbank_id", "drug_id", "drug"}:
                continue
            if not dbid or dbid in dbid_to_local:
                continue
            dbid_to_local[dbid] = len(smiles)
            smiles.append(smi)
    return smiles, dbid_to_local


_DBID_REGEX = re.compile(r"\bDB\d{5}\b")
_PRIMEKG_DRUGBANK_REGEX = re.compile(r"(?:^|::)DrugBank::(DB\d{5})(?:$|::)")


def _extract_dbid_from_node_name(node_name: str, node_type: Optional[int]) -> Optional[str]:
    if not node_name:
        return None

    # DRKG-style compound node, e.g., "Compound::DB00001".
    if node_type == 0 and node_name.startswith("Compound::"):
        return node_name.split("::", 1)[1]

    # PrimeKG converted node style from prepare_stride_inputs.py:
    # "<type>::DrugBank::DBxxxxx::<name>".
    m = _PRIMEKG_DRUGBANK_REGEX.search(node_name)
    if m is not None:
        return m.group(1)

    # Fallback for future node naming styles that still carry DrugBank IDs.
    m = _DBID_REGEX.search(node_name)
    if m is not None:
        return m.group(0)
    return None


def _read_compound_to_kg(kg_nodes_tsv: str) -> Dict[str, int]:
    mapping = {}
    with open(kg_nodes_tsv, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            node_id = int(parts[0])
            node_name = parts[1]
            node_type = None
            if len(parts) >= 3:
                try:
                    node_type = int(parts[2])
                except Exception:
                    node_type = None

            dbid = _extract_dbid_from_node_name(node_name=node_name, node_type=node_type)
            if dbid is not None and dbid not in mapping:
                mapping[dbid] = node_id
    return mapping


def load_deng_drug_space(
    data_root: str,
    drkg_dataset_dir: str = "DRKG+Deng",
    deng_dataset_dir: str = "Deng's dataset",
) -> Tuple[List[str], Dict[str, int], np.ndarray]:
    deng_drug_csv = os.path.join(data_root, deng_dataset_dir, "drug_listxiao.csv")
    kg_nodes_tsv = os.path.join(data_root, drkg_dataset_dir, "nodes.tsv")
    if not os.path.exists(deng_drug_csv):
        raise FileNotFoundError(f"Deng drug list not found: {deng_drug_csv}")
    if not os.path.exists(kg_nodes_tsv):
        raise FileNotFoundError(f"KG nodes file not found: {kg_nodes_tsv}")

    smiles, dbid_to_local = _read_dbid_smiles(deng_drug_csv)
    dbid_to_kg = _read_compound_to_kg(kg_nodes_tsv)

    drug_to_kg_id = np.full((len(smiles),), -1, dtype=np.int64)
    mapped = 0
    for dbid, local_id in dbid_to_local.items():
        kg_id = dbid_to_kg.get(dbid, None)
        if kg_id is not None:
            drug_to_kg_id[local_id] = int(kg_id)
            mapped += 1

    print(
        f"[load_deng_drug_space] num_drugs={len(smiles)} mapped_to_kg={mapped} "
        f"missing_in_kg={len(smiles) - mapped}"
    )
    return smiles, dbid_to_local, drug_to_kg_id


def _resolve_pair_drug_id(raw_id: str, dbid_to_local: Dict[str, int]) -> Optional[int]:
    drug_id = str(raw_id).strip()
    if drug_id in dbid_to_local:
        return int(dbid_to_local[drug_id])
    if drug_id.isdigit():
        local_id = int(drug_id)
        if 0 <= local_id < len(dbid_to_local):
            return local_id
    return None


def _read_pair_csv(csv_path: str, dbid_to_local: Dict[str, int]) -> np.ndarray:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d1 = str(row.get("d1", "")).strip()
            d2 = str(row.get("d2", "")).strip()
            rel = row.get("type", row.get("label", ""))
            d1_local = _resolve_pair_drug_id(d1, dbid_to_local)
            d2_local = _resolve_pair_drug_id(d2, dbid_to_local)
            if d1_local is None or d2_local is None:
                raise ValueError(
                    f"Unresolved drug ID at {csv_path}:{reader.line_num}: "
                    f"d1={d1!r}, d2={d2!r}"
                )
            if str(rel).strip() == "":
                raise ValueError(
                    f"Missing relation label at {csv_path}:{reader.line_num}"
                )
            try:
                relation_id = int(rel)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid relation label at {csv_path}:{reader.line_num}: {rel!r}"
                ) from exc
            rows.append([d1_local, d2_local, relation_id])
    if len(rows) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def load_official_s1_folds(
    data_root: str,
    dbid_to_local: Dict[str, int],
    fold_num: int = 5,
    deng_dataset_dir: str = "Deng's dataset",
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    folds = []
    for fold in range(fold_num):
        fold_dir = os.path.join(data_root, deng_dataset_dir, str(fold))
        train_csv = os.path.join(fold_dir, "ddi_training1xiao.csv")
        val_csv = os.path.join(fold_dir, "ddi_validation1xiao.csv")
        test_csv = os.path.join(fold_dir, "ddi_test1xiao.csv")
        if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
            raise FileNotFoundError(f"Official Deng fold files missing under: {fold_dir}")

        train_arr = _read_pair_csv(train_csv, dbid_to_local)
        val_arr = _read_pair_csv(val_csv, dbid_to_local)
        test_arr = _read_pair_csv(test_csv, dbid_to_local)
        folds.append((train_arr, val_arr, test_arr))
    return folds


def load_longtail_split(
    longtail_root: str,
    dbid_to_local: Dict[str, int],
    variant: str = "LT100",
    seed: int = 42,
    test_mode: str = "full",
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    variant = str(variant).strip().upper().replace("-", "")
    if variant != "LT100":
        raise ValueError(f"Unsupported long-tail variant: {variant}. This release keeps only LT100.")
    split_dir = os.path.join(longtail_root, f"{variant}_seed{int(seed)}")
    test_mode = str(test_mode).strip().lower()
    test_name = "ddi_test_balanced.csv" if test_mode == "balanced" else "ddi_test1.csv"

    train_csv = os.path.join(split_dir, "ddi_training1.csv")
    val_csv = os.path.join(split_dir, "ddi_validation1.csv")
    test_csv = os.path.join(split_dir, test_name)
    if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
        raise FileNotFoundError(f"Long-tail split files missing under: {split_dir}")

    train_arr = _read_pair_csv(train_csv, dbid_to_local)
    val_arr = _read_pair_csv(val_csv, dbid_to_local)
    test_arr = _read_pair_csv(test_csv, dbid_to_local)
    print(
        f"[load_longtail_split] root={longtail_root} variant={variant} seed={seed} test_mode={test_mode} "
        f"train={len(train_arr)} val={len(val_arr)} test={len(test_arr)}"
    )
    if len(train_arr) == 0 or len(val_arr) == 0 or len(test_arr) == 0:
        raise ValueError(f"Long-tail split has an empty mapped split under {split_dir}.")
    return [(train_arr, val_arr, test_arr)]


def load_relation_family_override(
    csv_path: str,
    num_relations: int,
    relation_ids: Optional[np.ndarray] = None,
):
    if csv_path is None or len(str(csv_path).strip()) == 0:
        return None, None
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"relation_family_csv not found: {csv_path}")

    if relation_ids is None:
        relation_ids = np.arange(num_relations, dtype=np.int64)
    relation_ids = np.asarray(relation_ids, dtype=np.int64).reshape(-1)
    if relation_ids.shape[0] != num_relations:
        raise ValueError(
            f"relation_ids mismatch: expected {num_relations}, got {relation_ids.shape[0]}"
        )
    if relation_ids.size > 0 and relation_ids.min() < 0:
        raise ValueError("relation_ids must be non-negative")

    family_by_relation = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row is None:
                continue
            rid_raw = row.get("relation_id", None)
            if rid_raw is None or str(rid_raw).strip() == "":
                continue
            rid = int(rid_raw)
            if rid < 0:
                continue

            fam_id_raw = row.get("family_id", None)
            fam_name = row.get("family_name", row.get("mechanism_family", "unknown_mixed"))
            fam_name = str(fam_name).strip() if fam_name is not None else "unknown_mixed"
            if fam_name == "":
                fam_name = "unknown_mixed"

            if fam_id_raw is not None and str(fam_id_raw).strip() != "":
                family_key = ("id", int(fam_id_raw))
            else:
                family_key = ("name", fam_name)
            family_by_relation[rid] = (family_key, fam_name)

    family_key_to_compact = {}
    id_to_name = {}
    relation_to_family = np.empty((num_relations,), dtype=np.int64)
    for compact_rid, original_rid in enumerate(relation_ids.tolist()):
        family_key, family_name = family_by_relation.get(
            int(original_rid),
            (("name", "unknown_mixed"), "unknown_mixed"),
        )
        if family_key not in family_key_to_compact:
            compact_family = len(family_key_to_compact)
            family_key_to_compact[family_key] = compact_family
            id_to_name[compact_family] = family_name
        relation_to_family[compact_rid] = family_key_to_compact[family_key]

    return relation_to_family, id_to_name


def load_source_relation_ids(
    csv_path: str,
    relation_ids: np.ndarray,
) -> np.ndarray:
    relation_ids = np.asarray(relation_ids, dtype=np.int64).reshape(-1)
    source_by_relation = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid_raw = row.get("relation_id", None)
            if rid_raw is None or str(rid_raw).strip() == "":
                continue
            rid = int(rid_raw)
            source_raw = row.get("source_relation_id", rid)
            source_by_relation[rid] = int(source_raw)
    return np.asarray(
        [source_by_relation.get(int(rid), int(rid)) for rid in relation_ids],
        dtype=np.int64,
    )


def build_relation_statistics(
    sample: np.ndarray,
    num_relations: int,
    drug_num: int,
    relation_group_size: int = 8,
    mutex_temperature: float = 0.05,
    tau_quantile: float = 0.2,
    relation_to_family_override: Optional[np.ndarray] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    num_relations = int(num_relations)
    if num_relations <= 0:
        raise ValueError("num_relations must be positive")

    if drug_num <= 0:
        raise ValueError("drug_num must be positive")

    class_counts = np.ones(num_relations, dtype=np.float32)
    relation_drug = np.zeros((num_relations, 2 * drug_num), dtype=np.float32)
    for row in sample:
        d1 = int(row[0])
        d2 = int(row[1])
        rel = int(row[2])
        if not 0 <= rel < num_relations:
            raise ValueError(f"Relation label {rel} is outside [0, {num_relations})")
        if not 0 <= d1 < drug_num or not 0 <= d2 < drug_num:
            raise ValueError(f"Drug pair ({d1}, {d2}) is outside [0, {drug_num})")
        class_counts[rel] += 1.0
        relation_drug[rel, d1] += 1.0
        relation_drug[rel, drug_num + d2] += 1.0

    norm = np.linalg.norm(relation_drug, axis=1, keepdims=True)
    observed_relation = norm[:, 0] > 0.0
    safe_norm = np.maximum(norm, 1e-8)
    relation_norm = relation_drug / safe_norm
    relation_graph = relation_norm @ relation_norm.T
    relation_graph = np.clip(relation_graph, 0.0, 1.0)
    np.fill_diagonal(relation_graph, 1.0)

    if relation_to_family_override is not None:
        relation_to_family = np.asarray(relation_to_family_override, dtype=np.int64).reshape(-1)
        if relation_to_family.shape[0] != num_relations:
            raise ValueError(
                f"relation_to_family_override mismatch: expected {num_relations}, got {relation_to_family.shape[0]}"
            )
        if relation_to_family.min() < 0:
            raise ValueError("relation_to_family_override must be non-negative")
        num_families = int(relation_to_family.max()) + 1
    else:
        group_size = max(1, int(relation_group_size))
        num_families = max(1, int(math.ceil(num_relations / float(group_size))))
        relation_to_family = np.arange(num_relations, dtype=np.int64) // group_size
        relation_to_family = np.clip(relation_to_family, 0, num_families - 1)

    # Soft mutex derived from the original drug-participation relation graph.
    q = float(np.clip(tau_quantile, 0.0, 1.0))
    tau_row = np.zeros((num_relations,), dtype=np.float32)
    for i in range(num_relations):
        row = relation_graph[i]
        if num_relations > 1:
            row_wo_diag = np.concatenate([row[:i], row[i + 1 :]])
        else:
            row_wo_diag = row
        if row_wo_diag.size == 0:
            tau_row[i] = 0.0
        else:
            tau_row[i] = float(np.quantile(row_wo_diag, q))

    tau_mat = np.minimum(tau_row[:, None], tau_row[None, :])
    temp = float(max(1e-6, mutex_temperature))
    mutex_logit = np.clip(
        (tau_mat - relation_graph) / temp,
        -60.0,
        60.0,
    )
    mutex = 1.0 / (1.0 + np.exp(-mutex_logit))
    mutex = mutex.astype(np.float32)
    np.fill_diagonal(mutex, 0.0)
    unsupported = ~observed_relation
    mutex[unsupported, :] = 0.0
    mutex[:, unsupported] = 0.0

    return {
        "relation_graph": torch.from_numpy(relation_graph).float().to(device),
        "mutex_prior": torch.from_numpy(mutex).float().to(device),
        "relation_to_family": torch.from_numpy(relation_to_family).long().to(device),
        "class_counts": torch.from_numpy(class_counts).float().to(device),
        "observed_relation_mask": torch.from_numpy(observed_relation).bool().to(device),
        "num_families": torch.tensor(num_families, dtype=torch.long, device=device),
    }


def load_drkg_graph(drkg_data_dir: str):
    edges_tsv = os.path.join(drkg_data_dir, "edges.tsv")
    nodes_tsv = os.path.join(drkg_data_dir, "nodes.tsv")
    if not os.path.exists(edges_tsv):
        raise FileNotFoundError(f"edges.tsv not found: {edges_tsv}")
    if not os.path.exists(nodes_tsv):
        raise FileNotFoundError(f"nodes.tsv not found: {nodes_tsv}")

    edge_list = []
    rel_list = []
    seen_edges = set()
    max_node = 0
    max_rel = 0
    with open(edges_tsv, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
            if h == t:
                continue
            edge_key = (h, r, t)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edge_list.append([h, t])
            rel_list.append(r)
            max_node = max(max_node, h, t)
            max_rel = max(max_rel, r)

    with open(nodes_tsv, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 1:
                continue
            max_node = max(max_node, int(parts[0]))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    rel_index = torch.tensor(rel_list, dtype=torch.long).contiguous()
    return edge_index, rel_index, int(max_node + 1), int(max_rel + 1)


def _make_fallback_mol_data() -> Tuple[Data, int]:
    x = torch.zeros((1, 67), dtype=torch.float32)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    sp_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    rel_index = torch.tensor([0], dtype=torch.long)
    sp_rel = torch.tensor([0], dtype=torch.long)
    sp_value = torch.tensor([0.0], dtype=torch.float32)
    data = Data(
        x=x,
        edge_index=edge_index,
        rel_index=rel_index,
        sp_edge_index=sp_edge_index,
        sp_value=sp_value,
        sp_edge_rel=sp_rel,
    )
    data.c_size = torch.tensor([1], dtype=torch.long)
    return data, 0


def _to_edge_tensor(edges: List[List[int]]) -> torch.Tensor:
    if len(edges) == 0:
        return torch.tensor([[0], [0]], dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_mol_cache(smiles_list: List[str]) -> Tuple[Dict[int, Data], int, int]:
    mol_cache = {}
    max_sp_rel = 0
    max_degree = 0
    fallback_ids = []
    for did, smi in enumerate(smiles_list):
        try:
            c_size, features, edge_index, rel_index, sp_edge_index, sp_value, sp_rel, deg = single_smile_to_graph(smi)
        except Exception as exc:
            print(f"[build_mol_cache] drug={did} preprocessing failed: {exc}")
            c_size = 0

        if c_size == 0:
            fallback_ids.append(did)
            data, rel_max = _make_fallback_mol_data()
            mol_cache[did] = data
            max_sp_rel = max(max_sp_rel, rel_max)
            continue

        edge_tensor = _to_edge_tensor(edge_index)
        sp_edge_tensor = _to_edge_tensor(sp_edge_index)
        rel_tensor = (
            torch.tensor(rel_index, dtype=torch.long)
            if len(rel_index) > 0
            else torch.tensor([0], dtype=torch.long)
        )
        sp_rel_tensor = (
            torch.tensor(sp_rel, dtype=torch.long)
            if len(sp_rel) > 0
            else torch.tensor([0], dtype=torch.long)
        )
        sp_value_tensor = (
            torch.tensor(sp_value, dtype=torch.float32)
            if len(sp_value) > 0
            else torch.tensor([0.0], dtype=torch.float32)
        )

        data = Data(
            x=torch.tensor(np.asarray(features), dtype=torch.float32),
            edge_index=edge_tensor,
            rel_index=rel_tensor,
            sp_edge_index=sp_edge_tensor,
            sp_value=sp_value_tensor,
            sp_edge_rel=sp_rel_tensor,
        )
        data.c_size = torch.tensor([int(c_size)], dtype=torch.long)
        mol_cache[did] = data

        max_sp_rel = max(max_sp_rel, int(sp_rel_tensor.max().item()))
        if edge_tensor.numel() > 0:
            deg_vec = degree(edge_tensor[1], num_nodes=int(data.x.shape[0]), dtype=torch.float32)
            max_degree = max(max_degree, int(deg_vec.max().item()))
        max_degree = max(max_degree, int(deg))

    if fallback_ids:
        print(
            f"[build_mol_cache] fallback molecules={len(fallback_ids)} "
            f"drug_ids={fallback_ids}"
        )
    return mol_cache, int(max_sp_rel + 1), int(max_degree + 1)


def _build_single_kg_subgraph_data(
    kg_node_id: int,
    edge_index: torch.Tensor,
    rel_index: torch.Tensor,
    num_rel: int,
    num_nodes_total: int,
    khop: int,
    fixed_num: int,
) -> Tuple[Data, int, int]:
    if kg_node_id < 0 or kg_node_id >= num_nodes_total:
        # Reserve the extra embedding index for drugs that have no KG entity.
        x = torch.tensor([num_nodes_total], dtype=torch.long)
        edge = torch.tensor([[0], [0]], dtype=torch.long)
        rel = torch.tensor([0], dtype=torch.long)
        sp_edge = torch.tensor([[0], [0]], dtype=torch.long)
        sp_value = torch.tensor([0.0], dtype=torch.float32)
        sp_rel = torch.tensor([0], dtype=torch.long)
        mask = torch.tensor([True], dtype=torch.bool)
        data = Data(
            x=x,
            edge_index=edge,
            rel_index=rel,
            id=mask,
            sp_edge_index=sp_edge,
            sp_value=sp_value,
            sp_edge_rel=sp_rel,
        )
        return data, 0, 1

    subset, sub_edge_index, sub_rel_index, mapping_mask = k_hop_subgraph(
        int(kg_node_id),
        int(khop),
        edge_index,
        rel_index,
        fixed_num=int(fixed_num),
        relabel_nodes=True,
    )

    if sub_edge_index.numel() == 0:
        sub_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        sub_rel_index = torch.tensor([0], dtype=torch.long)
        subset = torch.tensor([int(kg_node_id)], dtype=torch.long)
        mapping_mask = [True]

    new_s_edge_index = sub_edge_index.t().cpu().numpy().tolist()
    new_s_value = [1 for _ in range(len(new_s_edge_index))]
    new_s_rel = sub_rel_index.cpu().numpy().tolist()

    s_edge_index = list(new_s_edge_index)
    s_value = list(new_s_value)
    s_rel = list(new_s_rel)

    edge_index_value = calculate_shortest_path(sub_edge_index.t().cpu().numpy())
    if edge_index_value.ndim == 2 and edge_index_value.shape[1] >= 3:
        sp_edge_index = edge_index_value[:, :2]
        sp_value = edge_index_value[:, 2]
        for i in range(len(sp_edge_index)):
            if int(sp_value[i]) == 1:
                continue
            s_edge_index.append(sp_edge_index[i].tolist())
            s_value.append(int(sp_value[i]))
            s_rel.append(int(sp_value[i]) + int(num_rel))

    edge_tensor = _to_edge_tensor(new_s_edge_index)
    sp_edge_tensor = _to_edge_tensor(s_edge_index)
    rel_tensor = (
        torch.tensor(new_s_rel, dtype=torch.long)
        if len(new_s_rel) > 0
        else torch.tensor([0], dtype=torch.long)
    )
    sp_rel_tensor = (
        torch.tensor(s_rel, dtype=torch.long)
        if len(s_rel) > 0
        else torch.tensor([0], dtype=torch.long)
    )
    sp_value_tensor = (
        torch.tensor(s_value, dtype=torch.float32)
        if len(s_value) > 0
        else torch.tensor([0.0], dtype=torch.float32)
    )

    data = Data(
        x=subset.long().cpu(),
        edge_index=edge_tensor,
        rel_index=rel_tensor,
        id=torch.tensor(np.asarray(mapping_mask, dtype=bool)),
        sp_edge_index=sp_edge_tensor,
        sp_value=sp_value_tensor,
        sp_edge_rel=sp_rel_tensor,
    )

    deg_vec = degree(edge_tensor[1], num_nodes=int(data.x.shape[0]), dtype=torch.float32)
    max_deg = int(deg_vec.max().item()) if deg_vec.numel() > 0 else 0
    max_rel = int(sp_rel_tensor.max().item()) if sp_rel_tensor.numel() > 0 else 0
    return data, max_rel, max_deg


def build_kg_cache(
    drug_to_kg_id: np.ndarray,
    edge_index: torch.Tensor,
    rel_index: torch.Tensor,
    num_rel: int,
    num_nodes_total: int,
    khop: int = 2,
    fixed_num: int = 32,
) -> Tuple[Dict[int, Data], int, int]:
    kg_cache = {}
    max_sp_rel = 0
    max_degree = 0
    for did in range(len(drug_to_kg_id)):
        data, rel_max, deg_max = _build_single_kg_subgraph_data(
            kg_node_id=int(drug_to_kg_id[did]),
            edge_index=edge_index,
            rel_index=rel_index,
            num_rel=num_rel,
            num_nodes_total=num_nodes_total,
            khop=khop,
            fixed_num=fixed_num,
        )
        kg_cache[did] = data
        max_sp_rel = max(max_sp_rel, rel_max)
        max_degree = max(max_degree, deg_max)
    return kg_cache, int(max_sp_rel + 1), int(max_degree + 1)


def build_stride_feature_caches(
    smiles_list: List[str],
    drug_to_kg_id: np.ndarray,
    drkg_data_dir: str,
    khop: int = 2,
    fixed_num: int = 32,
    use_disk_cache: bool = True,
    cache_path: str = "",
) -> Tuple[Dict[int, Data], Dict[int, Data], Dict[str, int]]:
    def file_signature(path: str):
        stat = os.stat(path)
        return {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    smiles_signature = hashlib.sha256(
        "\n".join(str(smiles) for smiles in smiles_list).encode("utf-8")
    ).hexdigest()
    kg_signature = {
        "nodes": file_signature(os.path.join(drkg_data_dir, "nodes.tsv")),
        "edges": file_signature(os.path.join(drkg_data_dir, "edges.tsv")),
    }
    cache_file = str(cache_path).strip()
    if cache_file == "":
        cache_file = os.path.join(
            drkg_data_dir,
            f"stride_feature_cache_khop{int(khop)}_fixed{int(fixed_num)}_drug{int(len(smiles_list))}.pt",
        )

    if use_disk_cache and os.path.exists(cache_file):
        try:
            payload = torch.load(cache_file, map_location="cpu")
            if isinstance(payload, dict):
                meta = payload.get("meta", {})
                cache_drug_to_kg_id = np.asarray(meta.get("drug_to_kg_id", []), dtype=np.int64)
                same_meta = (
                    int(meta.get("cache_version", -1)) == FEATURE_CACHE_VERSION
                    and int(meta.get("khop", -1)) == int(khop)
                    and int(meta.get("fixed_num", -1)) == int(fixed_num)
                    and int(meta.get("num_drugs", -1)) == int(len(smiles_list))
                    and meta.get("smiles_signature") == smiles_signature
                    and meta.get("kg_signature") == kg_signature
                    and cache_drug_to_kg_id.shape == np.asarray(drug_to_kg_id, dtype=np.int64).shape
                    and np.array_equal(cache_drug_to_kg_id, np.asarray(drug_to_kg_id, dtype=np.int64))
                )
                if same_meta and "mol_cache" in payload and "kg_cache" in payload and "stats" in payload:
                    print(f"[build_stride_feature_caches] load cache: {cache_file}")
                    return payload["mol_cache"], payload["kg_cache"], payload["stats"]
                print("[build_stride_feature_caches] cache meta mismatch, rebuilding...")
            else:
                print("[build_stride_feature_caches] invalid cache payload, rebuilding...")
        except Exception as e:
            print(f"[build_stride_feature_caches] cache load failed: {e}, rebuilding...")

    edge_index, rel_index, num_nodes_total, num_rel = load_drkg_graph(drkg_data_dir)
    mol_cache, num_rel_mol, max_degree_graph = build_mol_cache(smiles_list)
    kg_cache, num_rel_graph, max_degree_node = build_kg_cache(
        drug_to_kg_id=drug_to_kg_id,
        edge_index=edge_index,
        rel_index=rel_index,
        num_rel=num_rel,
        num_nodes_total=num_nodes_total,
        khop=khop,
        fixed_num=fixed_num,
    )
    stats = {
        "num_features_drug": 67,
        "num_nodes": int(num_nodes_total + 1),
        "num_rel_mol": int(max(1, num_rel_mol)),
        "num_rel_graph": int(max(1, num_rel_graph)),
        "max_degree_graph": int(max(2, max_degree_graph)),
        "max_degree_node": int(max(2, max_degree_node)),
    }
    print(f"[build_stride_feature_caches] stats={stats}")

    if use_disk_cache:
        try:
            payload = {
                "mol_cache": mol_cache,
                "kg_cache": kg_cache,
                "stats": stats,
                "meta": {
                    "cache_version": FEATURE_CACHE_VERSION,
                    "khop": int(khop),
                    "fixed_num": int(fixed_num),
                    "num_drugs": int(len(smiles_list)),
                    "smiles_signature": smiles_signature,
                    "kg_signature": kg_signature,
                    "drug_to_kg_id": np.asarray(drug_to_kg_id, dtype=np.int64),
                },
            }
            torch.save(payload, cache_file)
            print(f"[build_stride_feature_caches] save cache: {cache_file}")
        except Exception as e:
            print(f"[build_stride_feature_caches] cache save failed: {e}")
    return mol_cache, kg_cache, stats


class STRIDEDataset(Dataset):
    def __init__(self, sample: np.ndarray, mol_cache: Dict[int, Data], kg_cache: Dict[int, Data]):
        super().__init__()
        self.sample = np.asarray(sample, dtype=np.int64).reshape(-1, 3)
        self.mol_cache = mol_cache
        self.kg_cache = kg_cache

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index: int):
        d1, d2, y = self.sample[index].tolist()
        mol1 = self.mol_cache[int(d1)].clone()
        kg1 = self.kg_cache[int(d1)].clone()
        mol2 = self.mol_cache[int(d2)].clone()
        kg2 = self.kg_cache[int(d2)].clone()
        return mol1, kg1, mol2, kg2, int(y)


def stride_collate(batch):
    mol1 = Batch.from_data_list([item[0] for item in batch])
    kg1 = Batch.from_data_list([item[1] for item in batch])
    mol2 = Batch.from_data_list([item[2] for item in batch])
    kg2 = Batch.from_data_list([item[3] for item in batch])
    y = torch.tensor([item[4] for item in batch], dtype=torch.long)
    return mol1, kg1, mol2, kg2, y
