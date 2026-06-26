import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.nn import Linear
from torch_geometric.utils import degree

from graph_transformer import GraphTransformer


def _pair_feature_dim(hidden_dim: int) -> int:
    return int(hidden_dim) * 4


def _pair_compose(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            left,
            right,
            left * right,
            torch.abs(left - right),
        ],
        dim=-1,
    )


def _safe_row_normalize(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = mat.sum(dim=-1, keepdim=True).clamp_min(eps)
    return mat / denom


class FactorizedRelationExpert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        context_dim: int,
        num_relations: int,
        relation_to_family: torch.Tensor,
        num_families: int,
        relation_embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_relations = int(num_relations)

        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.spec_weight = nn.Parameter(torch.empty(self.num_relations, hidden_dim))
        self.spec_bias = nn.Parameter(torch.zeros(self.num_relations))
        self.group_weight = nn.Parameter(torch.empty(num_families, hidden_dim))
        self.group_bias = nn.Parameter(torch.zeros(num_families))
        nn.init.xavier_uniform_(self.spec_weight)
        nn.init.xavier_uniform_(self.group_weight)

        gate_in_dim = context_dim + relation_embed_dim * 2
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        self.register_buffer("relation_to_family", relation_to_family.long())

    def forward(
        self,
        pair_feat: torch.Tensor,
        pair_context: torch.Tensor,
        relation_embed: torch.Tensor,
        family_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.base(pair_feat)
        spec_score = (hidden * self.spec_weight.unsqueeze(0)).sum(dim=-1) + self.spec_bias.unsqueeze(0)

        fam_idx = self.relation_to_family
        group_w = self.group_weight[fam_idx]
        group_b = self.group_bias[fam_idx]
        group_score = (hidden * group_w.unsqueeze(0)).sum(dim=-1) + group_b.unsqueeze(0)

        bsz = pair_feat.shape[0]
        ctx = pair_context.unsqueeze(1).expand(-1, self.num_relations, -1)
        rel = relation_embed.unsqueeze(0).expand(bsz, -1, -1)
        fam = family_embed[fam_idx].unsqueeze(0).expand(bsz, -1, -1)
        gate = torch.softmax(self.gate(torch.cat([ctx, rel, fam], dim=-1)), dim=-1)
        score = gate[..., 0] * spec_score + gate[..., 1] * group_score
        return score, hidden


def init_params(module, layers=2):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class NodeFeatures(torch.nn.Module):
    def __init__(self, degree, feature_num, embedding_dim, layer=2, type="graph"):
        super().__init__()
        if type == "graph":
            self.node_encoder = Linear(feature_num, embedding_dim)
        else:
            self.node_encoder = torch.nn.Embedding(feature_num, embedding_dim)
        self.degree_encoder = torch.nn.Embedding(degree, embedding_dim, padding_idx=0)
        self.apply(lambda module: init_params(module, layers=layer))

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.degree_encoder.reset_parameters()

    def forward(self, data):
        _, col = data.edge_index
        x_degree = degree(col, data.x.size(0), dtype=data.x.dtype)
        node_feature = self.node_encoder(data.x)
        node_feature += self.degree_encoder(x_degree.long())
        return node_feature


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for module in self.modules():
            self.weights_init(module)

    def weights_init(self, module):
        if isinstance(module, nn.Bilinear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        sc_1 = self.f_k(h_pl, c)
        sc_2 = self.f_k(h_mi, c)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        return torch.cat((sc_1, sc_2), 0)


class STRIDE(nn.Module):
    def __init__(
        self,
        num_features_drug: int,
        num_nodes: int,
        num_relations_mol: int,
        num_relations_graph: int,
        max_degree_graph: int,
        max_degree_node: int,
        class_num: int,
        relation_to_family: torch.Tensor,
        relation_graph: torch.Tensor,
        mutex_prior: torch.Tensor,
        max_layer: int = 2,
        hidden_dim: int = 128,
        num_heads: int = 4,
        expert_hidden: int = 256,
        align_hidden: int = 256,
        decode_steps: int = 3,
        align_chunk_size: int = 8,
        dropout: float = 0.2,
        lambda_pairwise: float = 0.2,
        lambda_smooth: float = 0.2,
        lambda_consistency: float = 0.2,
        lambda_mutex: float = 0.1,
        enable_alignment: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_relations = int(class_num)
        if self.num_relations <= 0:
            raise ValueError("class_num must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(num_heads) <= 0 or self.hidden_dim % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by a positive num_heads")
        self.decode_steps = int(max(1, decode_steps))
        self.align_chunk_size = int(max(1, align_chunk_size))
        self.use_mol = True
        self.use_kg = True

        relation_to_family = relation_to_family.long()
        if relation_to_family.ndim != 1 or relation_to_family.numel() != self.num_relations:
            raise ValueError(
                "relation_to_family must contain one entry per relation"
            )
        if relation_to_family.numel() > 0 and relation_to_family.min().item() < 0:
            raise ValueError("relation_to_family must be non-negative")
        expected_relation_shape = (self.num_relations, self.num_relations)
        if tuple(relation_graph.shape) != expected_relation_shape:
            raise ValueError(
                f"relation_graph must have shape {expected_relation_shape}"
            )
        if tuple(mutex_prior.shape) != expected_relation_shape:
            raise ValueError(
                f"mutex_prior must have shape {expected_relation_shape}"
            )
        if not torch.isfinite(relation_graph).all():
            raise ValueError("relation_graph must contain only finite values")
        if not torch.isfinite(mutex_prior).all():
            raise ValueError("mutex_prior must contain only finite values")
        num_families = int(relation_to_family.max().item()) + 1 if relation_to_family.numel() > 0 else 1
        relation_graph = relation_graph.float().clone()
        relation_affinity = 0.5 * (relation_graph + relation_graph.t())
        relation_affinity.fill_diagonal_(0.0)
        relation_affinity = relation_affinity.clamp_min(0.0)
        identity = torch.eye(
            self.num_relations,
            dtype=relation_affinity.dtype,
            device=relation_affinity.device,
        )
        adjacency = relation_affinity + identity
        degree = adjacency.sum(dim=-1).clamp_min(1e-8)
        propagation = adjacency / degree.unsqueeze(-1)
        inv_sqrt_degree = degree.rsqrt()
        normalized_adjacency = (
            inv_sqrt_degree.unsqueeze(-1)
            * adjacency
            * inv_sqrt_degree.unsqueeze(0)
        )
        relation_laplacian = identity - normalized_adjacency

        mutex_prior = mutex_prior.float().clamp_min(0.0)
        mutex_prior = 0.5 * (mutex_prior + mutex_prior.t())
        mutex_prior.fill_diagonal_(0.0)

        self.register_buffer("relation_to_family", relation_to_family)
        self.register_buffer("relation_affinity", relation_affinity)
        self.register_buffer("relation_graph", propagation)
        self.register_buffer("mutex_prior", mutex_prior)
        self.register_buffer("relation_laplacian", relation_laplacian)

        self.lambda_pairwise = float(lambda_pairwise)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_consistency = float(lambda_consistency)
        self.lambda_mutex = float(lambda_mutex)
        self.enable_alignment = bool(enable_alignment)

        self.mol_atom_feature = NodeFeatures(
            degree=max_degree_graph,
            feature_num=num_features_drug,
            embedding_dim=hidden_dim,
            type="graph",
        )
        self.drug_node_feature = NodeFeatures(
            degree=max_degree_node,
            feature_num=num_nodes,
            embedding_dim=hidden_dim,
            type="node",
        )
        self.mol_representation_learning = GraphTransformer(
            layer_num=max_layer,
            embedding_dim=hidden_dim,
            num_heads=num_heads,
            num_rel=num_relations_mol,
            dropout=dropout,
            type="graph",
        )
        self.node_representation_learning = GraphTransformer(
            layer_num=max_layer,
            embedding_dim=hidden_dim,
            num_heads=num_heads,
            num_rel=num_relations_graph,
            dropout=dropout,
            type="node",
        )

        self.fc_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
        )

        self.disc = Discriminator(hidden_dim)
        self.b_xent = BCEWithLogitsLoss()

        self.relation_embedding = nn.Embedding(self.num_relations, hidden_dim)
        self.family_embedding = nn.Embedding(num_families, hidden_dim)

        pair_dim = _pair_feature_dim(hidden_dim)
        self.pair_dim = pair_dim
        self.mol_rel_pair_proj = nn.Linear(hidden_dim, pair_dim, bias=False)
        self.mol_fam_pair_proj = nn.Linear(hidden_dim, pair_dim, bias=False)
        self.kg_rel_pair_proj = nn.Linear(hidden_dim, pair_dim, bias=False)
        self.kg_fam_pair_proj = nn.Linear(hidden_dim, pair_dim, bias=False)

        self.context_proj = nn.Sequential(
            nn.Linear(pair_dim, expert_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if self.use_mol:
            self.mol_expert = FactorizedRelationExpert(
                input_dim=pair_dim,
                hidden_dim=expert_hidden,
                context_dim=expert_hidden,
                num_relations=self.num_relations,
                relation_to_family=self.relation_to_family,
                num_families=num_families,
                relation_embed_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.mol_expert = None

        if self.use_kg:
            self.kg_expert = FactorizedRelationExpert(
                input_dim=pair_dim,
                hidden_dim=expert_hidden,
                context_dim=expert_hidden,
                num_relations=self.num_relations,
                relation_to_family=self.relation_to_family,
                num_families=num_families,
                relation_embed_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.kg_expert = None

        if self.use_mol and self.use_kg:
            self.cross_expert = FactorizedRelationExpert(
                input_dim=pair_dim * 3,
                hidden_dim=expert_hidden,
                context_dim=expert_hidden,
                num_relations=self.num_relations,
                relation_to_family=self.relation_to_family,
                num_families=num_families,
                relation_embed_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.cross_expert = None

        self.view_names = []
        if self.use_mol:
            self.view_names.append("mol")
        if self.use_kg:
            self.view_names.append("kg")
        if self.cross_expert is not None:
            self.view_names.append("cross")
        self.num_views = len(self.view_names)

        if self.num_views > 1:
            self.view_gate = nn.Sequential(
                nn.Linear(expert_hidden + hidden_dim * 2, expert_hidden),
                nn.ReLU(),
                nn.Linear(expert_hidden, self.num_views),
            )
        else:
            self.view_gate = None

        align_input_dim = pair_dim * 2 if (self.use_mol and self.use_kg) else pair_dim
        self.align_fuse = nn.Sequential(
            nn.Linear(align_input_dim, align_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(align_hidden, align_hidden),
            nn.ReLU(),
        )
        self.align_conditioner = nn.Sequential(
            nn.Linear(align_hidden * 2, align_hidden),
            nn.ReLU(),
            nn.Linear(align_hidden, align_hidden),
        )
        self.align_head_weight = nn.Parameter(torch.empty(self.num_relations, align_hidden))
        self.align_head_bias = nn.Parameter(torch.zeros(self.num_relations))
        nn.init.xavier_uniform_(self.align_head_weight)

        self.pairwise_raw = nn.Parameter(torch.zeros(self.num_relations, self.num_relations))

    def _pairwise_matrix(self) -> torch.Tensor:
        pairwise = 0.5 * (self.pairwise_raw + self.pairwise_raw.t())
        pairwise = pairwise - torch.diag(torch.diag(pairwise))
        return pairwise

    def MI(self, graph_embeddings: torch.Tensor, sub_embeddings: list) -> torch.Tensor:
        n = graph_embeddings.shape[0]
        if n < 2:
            return graph_embeddings.new_empty((0, 1))
        idx = torch.roll(
            torch.arange(n, device=graph_embeddings.device),
            shifts=1,
        )
        shuffle_embeddings = torch.index_select(graph_embeddings, 0, idx)
        c_pos_list = []
        c_neg_list = []
        for c_pos, c_neg, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
            c_pos_list.append(c_pos.expand_as(sub))
            c_neg_list.append(c_neg.expand_as(sub))
        c_pos = torch.cat(c_pos_list)
        c_neg = torch.cat(c_neg_list)
        sub_cat = torch.cat(sub_embeddings)
        return self.disc(sub_cat, c_pos, c_neg)

    def loss_MI(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.sum()
        num_logits = logits.shape[0] // 2
        lbl = torch.cat(
            [
                torch.ones(num_logits, device=logits.device),
                torch.zeros(num_logits, device=logits.device),
            ],
            dim=0,
        )
        return self.b_xent(logits.view(1, -1), lbl.view(1, -1))

    def _prepare_view_pair(
        self,
        pair_base: torch.Tensor,
        relation_embed: torch.Tensor,
        family_embed: torch.Tensor,
        rel_proj: nn.Linear,
        fam_proj: nn.Linear,
    ) -> torch.Tensor:
        base = pair_base.unsqueeze(1).expand(-1, self.num_relations, -1)
        fam_idx = self.relation_to_family
        rel_bias = rel_proj(relation_embed).unsqueeze(0)
        fam_bias = fam_proj(family_embed[fam_idx]).unsqueeze(0)
        return base + rel_bias + fam_bias

    def _forward_alignment(self, z: torch.Tensor, isolate_backward: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor = z.detach()
        helper = z.detach() if isolate_backward else z
        beta = self.relation_graph
        bsz, rel_num, hid = z.shape

        aligned_chunks = []
        chunk = self.align_chunk_size
        for start in range(0, rel_num, chunk):
            end = min(start + chunk, rel_num)
            chunk_size = end - start

            anchor_chunk = anchor[:, start:end, :]
            anchor_expand = anchor_chunk.unsqueeze(2).expand(-1, -1, rel_num, -1)
            helper_expand = helper.unsqueeze(1).expand(-1, chunk_size, -1, -1)
            fitted_chunk = self.align_conditioner(torch.cat([anchor_expand, helper_expand], dim=-1))
            aligned_chunk = z[:, start:end, :] + torch.einsum("cr,bcrd->bcd", beta[start:end, :], fitted_chunk)
            aligned_chunks.append(aligned_chunk)
        aligned = torch.cat(aligned_chunks, dim=1)

        z_sq = (z * z).mean(dim=-1)
        z_cross = torch.matmul(z, z.transpose(1, 2)) / float(hid)
        diff = z_sq.unsqueeze(2) + z_sq.unsqueeze(1) - 2.0 * z_cross
        diff = diff.clamp_min(0.0)
        align_loss = (beta.unsqueeze(0) * diff).mean()
        return aligned, align_loss

    def _constraint_grad(
        self,
        y: torch.Tensor,
        mol_view_logits: Optional[torch.Tensor],
        kg_view_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pairwise = self._pairwise_matrix()
        grad_pair = torch.matmul(y, pairwise)
        grad_smooth = torch.matmul(y, self.relation_laplacian)
        grad_mutex = torch.matmul(y, self.mutex_prior)

        if mol_view_logits is not None and kg_view_logits is not None:
            view_gap = torch.sigmoid(mol_view_logits) - torch.sigmoid(kg_view_logits)
            grad_cons = view_gap.square() * y
        else:
            grad_cons = torch.zeros_like(y)

        grad = (
            self.lambda_pairwise * grad_pair
            + self.lambda_smooth * grad_smooth
            + self.lambda_consistency * grad_cons
            + self.lambda_mutex * grad_mutex
        )
        return grad

    def energy_from_assignment(
        self,
        y: torch.Tensor,
        logits: torch.Tensor,
        mol_view_logits: Optional[torch.Tensor] = None,
        kg_view_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pairwise = self._pairwise_matrix()
        unary_energy = -(y * logits).sum(dim=-1)
        pair_energy = 0.5 * (y * torch.matmul(y, pairwise)).sum(dim=-1)
        smooth_energy = 0.5 * (y * torch.matmul(y, self.relation_laplacian)).sum(dim=-1)
        mutex_energy = 0.5 * (y * torch.matmul(y, self.mutex_prior)).sum(dim=-1)

        if mol_view_logits is not None and kg_view_logits is not None:
            view_gap_sq = (
                torch.sigmoid(mol_view_logits)
                - torch.sigmoid(kg_view_logits)
            ).square()
            consistency_energy = 0.5 * (y.square() * view_gap_sq).sum(dim=-1)
        else:
            consistency_energy = torch.zeros_like(unary_energy)

        total = (
            unary_energy
            + self.lambda_pairwise * pair_energy
            + self.lambda_smooth * smooth_energy
            + self.lambda_mutex * mutex_energy
            + self.lambda_consistency * consistency_energy
        )
        return {
            "total": total,
            "unary": unary_energy,
            "pairwise": pair_energy,
            "smooth": smooth_energy,
            "mutex": mutex_energy,
            "consistency": consistency_energy,
        }

    def _structured_decode(
        self,
        logits: torch.Tensor,
        label_type: str,
        mol_view_logits: Optional[torch.Tensor] = None,
        kg_view_logits: Optional[torch.Tensor] = None,
        decode_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        steps = int(self.decode_steps if decode_steps is None else max(1, decode_steps))
        if label_type == "multi_class":
            y = torch.softmax(logits, dim=-1)
        else:
            y = torch.sigmoid(logits)

        correction = torch.zeros_like(logits)
        for _ in range(steps):
            correction = self._constraint_grad(y, mol_view_logits, kg_view_logits)
            if label_type == "multi_class":
                y = torch.softmax(logits - correction, dim=-1)
            else:
                y = torch.sigmoid(logits - correction)

        structured_logits = logits - correction
        energy = self.energy_from_assignment(y, logits, mol_view_logits, kg_view_logits)
        return structured_logits, y, energy

    def forward(
        self,
        drug1_mol,
        drug1_subgraph,
        drug2_mol,
        drug2_subgraph,
        label_type: str = "multi_class",
        decode_steps: Optional[int] = None,
        isolate_backward: bool = True,
    ) -> Dict[str, torch.Tensor]:
        mol1_atom_feature = self.mol_atom_feature(drug1_mol)
        mol2_atom_feature = self.mol_atom_feature(drug2_mol)
        drug1_node_feature = self.drug_node_feature(drug1_subgraph)
        drug2_node_feature = self.drug_node_feature(drug2_subgraph)

        mol1_graph_embedding, mol1_atom_embedding, _ = self.mol_representation_learning(mol1_atom_feature, drug1_mol)
        mol2_graph_embedding, mol2_atom_embedding, _ = self.mol_representation_learning(mol2_atom_feature, drug2_mol)
        drug1_node_embedding, drug1_sub_embedding, _ = self.node_representation_learning(drug1_node_feature, drug1_subgraph)
        drug2_node_embedding, drug2_sub_embedding, _ = self.node_representation_learning(drug2_node_feature, drug2_subgraph)

        drug1_embedding = self.fc_fuse(torch.cat([drug1_node_embedding, mol1_graph_embedding], dim=-1))
        drug2_embedding = self.fc_fuse(torch.cat([drug2_node_embedding, mol2_graph_embedding], dim=-1))

        mi_mol_loss = self.loss_MI(self.MI(drug1_embedding, mol1_atom_embedding)) + self.loss_MI(
            self.MI(drug2_embedding, mol2_atom_embedding)
        )
        mi_kg_loss = self.loss_MI(self.MI(drug1_embedding, drug1_sub_embedding)) + self.loss_MI(
            self.MI(drug2_embedding, drug2_sub_embedding)
        )

        relation_embed = self.relation_embedding.weight
        family_embed = self.family_embedding.weight
        pair_context = self.context_proj(
            _pair_compose(drug1_embedding, drug2_embedding)
        )

        view_scores = {}
        mol_score = None
        kg_score = None
        cross_score = None

        if self.use_mol:
            mol_pair_base = _pair_compose(
                mol1_graph_embedding,
                mol2_graph_embedding,
            )
            mol_pair = self._prepare_view_pair(
                pair_base=mol_pair_base,
                relation_embed=relation_embed,
                family_embed=family_embed,
                rel_proj=self.mol_rel_pair_proj,
                fam_proj=self.mol_fam_pair_proj,
            )
            mol_score, _ = self.mol_expert(mol_pair, pair_context, relation_embed, family_embed)
            view_scores["mol"] = mol_score
        else:
            mol_pair = None

        if self.use_kg:
            kg_pair_base = _pair_compose(
                drug1_node_embedding,
                drug2_node_embedding,
            )
            kg_pair = self._prepare_view_pair(
                pair_base=kg_pair_base,
                relation_embed=relation_embed,
                family_embed=family_embed,
                rel_proj=self.kg_rel_pair_proj,
                fam_proj=self.kg_fam_pair_proj,
            )
            kg_score, _ = self.kg_expert(kg_pair, pair_context, relation_embed, family_embed)
            view_scores["kg"] = kg_score
        else:
            kg_pair = None

        if self.cross_expert is not None:
            cross_pair = torch.cat([mol_pair, kg_pair, mol_pair * kg_pair], dim=-1)
            cross_score, _ = self.cross_expert(cross_pair, pair_context, relation_embed, family_embed)
            view_scores["cross"] = cross_score

        stacked_scores = torch.stack([view_scores[name] for name in self.view_names], dim=-1)
        if self.num_views == 1:
            unary_logits = stacked_scores.squeeze(-1)
            view_gate = torch.ones_like(stacked_scores)
        else:
            bsz = stacked_scores.shape[0]
            fam_idx = self.relation_to_family
            ctx = pair_context.unsqueeze(1).expand(-1, self.num_relations, -1)
            rel = relation_embed.unsqueeze(0).expand(bsz, -1, -1)
            fam = family_embed[fam_idx].unsqueeze(0).expand(bsz, -1, -1)
            gate_logits = self.view_gate(torch.cat([ctx, rel, fam], dim=-1))
            view_gate = torch.softmax(gate_logits, dim=-1)
            unary_logits = (stacked_scores * view_gate).sum(dim=-1)

        if self.enable_alignment:
            if self.use_mol and self.use_kg:
                align_input = torch.cat([mol_pair, kg_pair], dim=-1)
            elif self.use_mol:
                align_input = mol_pair
            else:
                align_input = kg_pair
            z = self.align_fuse(align_input)
            aligned_z, align_loss = self._forward_alignment(z, isolate_backward=isolate_backward)
            align_logits = (aligned_z * self.align_head_weight.unsqueeze(0)).sum(dim=-1) + self.align_head_bias.unsqueeze(0)
        else:
            align_loss = torch.tensor(0.0, device=unary_logits.device)
            align_logits = torch.zeros_like(unary_logits)
        logits = unary_logits + align_logits

        structured_logits, relaxed_assignment, energy = self._structured_decode(
            logits=logits,
            label_type=label_type,
            mol_view_logits=mol_score,
            kg_view_logits=kg_score,
            decode_steps=decode_steps,
        )

        if mol_score is not None and kg_score is not None:
            consistency_loss = F.mse_loss(torch.sigmoid(mol_score), torch.sigmoid(kg_score))
        else:
            consistency_loss = torch.tensor(0.0, device=logits.device)

        return {
            "logits": logits,
            "structured_logits": structured_logits,
            "relaxed_assignment": relaxed_assignment,
            "energy": energy,
            "align_loss": align_loss,
            "consistency_loss": consistency_loss,
            "mi_mol_loss": mi_mol_loss,
            "mi_kg_loss": mi_kg_loss,
            "mol_view_logits": mol_score if mol_score is not None else torch.zeros_like(logits),
            "kg_view_logits": kg_score if kg_score is not None else torch.zeros_like(logits),
            "cross_view_logits": cross_score if cross_score is not None else torch.zeros_like(logits),
            "unary_logits": unary_logits,
            "align_logits": align_logits,
            "view_gate": view_gate,
        }
