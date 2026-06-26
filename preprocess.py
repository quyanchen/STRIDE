import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch import Tensor


RDLogger.DisableLog("rdApp.warning")


BOND_TYPES = [
    "UNSPECIFIED",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "QUADRUPLE",
    "QUINTUPLE",
    "HEXTUPLE",
    "ONEANDAHALF",
    "TWOANDAHALF",
    "THREEANDAHALF",
    "FOURANDAHALF",
    "FIVEANDAHALF",
    "AROMATIC",
    "IONIC",
    "HYDROGEN",
    "THREECENTER",
    "DATIVEONE",
    "DATIVE",
    "DATIVEL",
    "DATIVER",
    "OTHER",
    "ZERO",
]


def one_of_k_encoding_unk(value, allowable_set):
    if value not in allowable_set:
        value = allowable_set[-1]
    return [value == item for item in allowable_set]


def atom_features(atom):
    features = (
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C",
                "N",
                "O",
                "S",
                "F",
                "Si",
                "P",
                "Cl",
                "Br",
                "Mg",
                "Na",
                "Ca",
                "Fe",
                "As",
                "Al",
                "I",
                "B",
                "V",
                "K",
                "Tl",
                "Yb",
                "Sb",
                "Sn",
                "Ag",
                "Pd",
                "Co",
                "Se",
                "Ti",
                "Zn",
                "H",
                "Li",
                "Ge",
                "Cu",
                "Au",
                "Ni",
                "Cd",
                "In",
                "Mn",
                "Zr",
                "Cr",
                "Pt",
                "Hg",
                "Pb",
                "X",
            ],
        )
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + [atom.GetIsAromatic()]
    )
    return np.asarray(features), atom.GetDegree()


def calculate_shortest_path(edge_index):
    graph = nx.DiGraph()
    graph.add_edges_from(edge_index.tolist())
    rows = []
    for source, targets in nx.all_pairs_shortest_path_length(graph):
        for target, length in targets.items():
            rows.append([source, target, length])
    rows.sort()
    return np.asarray(rows, dtype=np.int64)


def single_smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return 0, 0, 0, 0, 0, 0, 0, 0

    features = []
    degrees = []
    for atom in mol.GetAtoms():
        feature, degree = atom_features(atom)
        features.append((feature / sum(feature)).tolist())
        degrees.append(degree)

    mol_edges = []
    for bond in mol.GetBonds():
        bond_type = BOND_TYPES.index(str(bond.GetBondType()))
        mol_edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type])
        mol_edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), bond_type])

    if not mol_edges:
        atom_num = mol.GetNumAtoms()
        self_edges = [[index, index] for index in range(atom_num)]
        zero_relations = [0 for _ in range(atom_num)]
        zero_distances = [0 for _ in range(atom_num)]
        return (
            atom_num,
            features,
            self_edges,
            zero_relations,
            self_edges,
            zero_distances,
            zero_relations,
            max(degrees) if degrees else 0,
        )

    mol_edges = np.asarray(sorted(mol_edges), dtype=np.int64)
    mol_edge_index = mol_edges[:, :2]
    mol_rel_index = mol_edges[:, 2]

    sp_edge_value = calculate_shortest_path(mol_edge_index)
    sp_edge_index = sp_edge_value[:, :2]
    sp_value = sp_edge_value[:, 2]
    sp_rel = sp_value.copy()
    sp_rel[np.where(sp_value == 1)] = mol_rel_index
    sp_rel[np.where(sp_value != 1)] += 23

    return (
        mol.GetNumAtoms(),
        features,
        mol_edge_index.tolist(),
        mol_rel_index.tolist(),
        sp_edge_index.tolist(),
        sp_value.tolist(),
        sp_rel.tolist(),
        max(degrees),
    )


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    if isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    return max(edge_index.size(0), edge_index.size(1))


def k_hop_subgraph(
    node_idx,
    num_hops,
    edge_index,
    rel_index,
    fixed_num,
    relabel_nodes=False,
    num_nodes=None,
    flow="source_to_target",
    seed=42,
):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if flow not in ["source_to_target", "target_to_source"]:
        raise ValueError(f"Unsupported flow: {flow}")

    if flow == "target_to_source":
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]
    root_id = int(node_idx.flatten()[0].item())
    for hop in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        neighbors = col[edge_mask].unique()
        if fixed_num is None or neighbors.size(0) <= fixed_num:
            subsets.append(neighbors)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) + root_id * 1009 + hop)
            selected = torch.randperm(
                neighbors.size(0),
                generator=generator,
            )[:fixed_num].to(neighbors.device)
            subsets.append(neighbors[selected])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[: node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]
    edge_index = edge_index[:, edge_mask]
    rel_index = rel_index[edge_mask] if rel_index is not None else None

    if relabel_nodes:
        node_map = row.new_full((num_nodes,), -1, dtype=torch.long)
        node_map[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_map[edge_index]

    mapping_mask = torch.zeros(subset.size(0), dtype=torch.bool, device=subset.device)
    mapping_mask[inv] = True
    return subset, edge_index, rel_index, mapping_mask.cpu().tolist()
