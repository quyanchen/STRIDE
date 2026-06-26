import torch
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class GraphTransformerEncode(torch.nn.Module):
    def __init__(self, num_heads, in_dim, dim_forward, rel_encoder, spatial_encoder, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.dim_forward = dim_forward

        self.ffn = Sequential(
            Linear(self.in_dim, self.dim_forward),
            ReLU(),
            Linear(self.dim_forward, self.in_dim),
        )
        self.multi_head_attention = MultiheadAttention(
            dim_model=self.in_dim,
            num_heads=self.num_heads,
            rel_encoder=rel_encoder,
            spatial_encoder=spatial_encoder,
        )
        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def reset_parameters(self):
        self.ffn[0].reset_parameters()
        self.ffn[2].reset_parameters()
        self.multi_head_attention.reset_parameters()
        self.layernorm1.reset_parameters()
        self.layernorm2.reset_parameters()

    def forward(self, feature, sp_edge_index, sp_value, edge_rel):
        x_norm = self.layernorm1(feature)
        attn_output, attn_weight = self.multi_head_attention(
            x_norm,
            sp_edge_index,
            sp_value,
            edge_rel,
        )
        out1 = feature + self.dropout1(attn_output)

        ffn_output = self.ffn(self.layernorm2(out1))
        out2 = out1 + self.dropout2(ffn_output)
        return out2, attn_weight


class SpatialEncoding(torch.nn.Module):
    def __init__(self, dim_model):
        super().__init__()
        self.fnn = Sequential(
            Linear(1, dim_model),
            ReLU(),
            Linear(dim_model, 1),
            ReLU(),
        )

    def reset_parameters(self):
        self.fnn[0].reset_parameters()
        self.fnn[2].reset_parameters()

    def forward(self, lap):
        lap = lap.to(dtype=self.fnn[0].weight.dtype).unsqueeze(-1)
        return self.fnn(lap)


class MultiheadAttention(MessagePassing):
    def __init__(self, dim_model, num_heads, rel_encoder, spatial_encoder, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        if dim_model % num_heads != 0:
            raise ValueError("dim_model must be divisible by num_heads")

        self.d_model = dim_model
        self.num_heads = num_heads
        self.depth = self.d_model // num_heads
        self.rel_embedding = rel_encoder
        self.spatial_encoding = spatial_encoder
        self.wq = Linear(dim_model, dim_model)
        self.wk = Linear(dim_model, dim_model)
        self.wv = Linear(dim_model, dim_model)
        self.dense = Linear(dim_model, dim_model)

    def reset_parameters(self):
        self.rel_embedding.reset_parameters()
        self.spatial_encoding.reset_parameters()
        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.dense.reset_parameters()

    def forward(self, x, sp_edge_index, sp_value, edge_rel):
        rel_embedding = self.rel_embedding(edge_rel).view(
            -1,
            self.num_heads,
            self.depth,
        )
        q = self.wq(x).view(x.shape[0], self.num_heads, self.depth)
        k = self.wk(x).view(x.shape[0], self.num_heads, self.depth)
        v = self.wv(x).view(x.shape[0], self.num_heads, self.depth)

        row, col = sp_edge_index
        attention_score = ((q[col] + rel_embedding) * (k[row] + rel_embedding)).sum(
            dim=-1,
        ) / (self.depth ** 0.5)
        spatial_bias = self.spatial_encoding(sp_value).squeeze(-1).unsqueeze(-1)
        attention_weight = softmax(
            attention_score + spatial_bias,
            index=col,
            num_nodes=x.shape[0],
        )

        outputs = [
            self.propagate(
                edge_index=sp_edge_index,
                x=v[:, head, :],
                edge_weight=attention_weight[:, head],
                size=None,
            )
            for head in range(self.num_heads)
        ]
        return self.dense(torch.cat(outputs, dim=-1)), attention_weight

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.unsqueeze(-1)


class GraphTransformer(torch.nn.Module):
    def __init__(
        self,
        layer_num=3,
        embedding_dim=64,
        num_heads=4,
        num_rel=10,
        dropout=0.2,
        type="graph",
    ):
        super().__init__()
        if layer_num < 1:
            raise ValueError("layer_num must be at least 1")

        self.type = type
        self.rel_encoder = torch.nn.Embedding(num_rel, embedding_dim)
        self.spatial_encoder = SpatialEncoding(embedding_dim)
        self.encoder = torch.nn.ModuleList(
            GraphTransformerEncode(
                num_heads=num_heads,
                in_dim=embedding_dim,
                dim_forward=embedding_dim * 2,
                rel_encoder=self.rel_encoder,
                spatial_encoder=self.spatial_encoder,
                dropout=dropout,
            )
            for _ in range(layer_num)
        )

    def reset_parameters(self):
        for encoder in self.encoder:
            encoder.reset_parameters()

    def forward(self, feature, data):
        x = feature
        attn_layer = []
        for graph_encoder in self.encoder:
            x, attn = graph_encoder(
                x,
                data.sp_edge_index,
                data.sp_value,
                data.sp_edge_rel,
            )
            attn_layer.append(attn)

        sub_representation = []
        for index, _ in enumerate(data.to_data_list()):
            sub_embedding = x[(data.batch == index).nonzero().flatten()]
            sub_representation.append(sub_embedding)

        if self.type == "graph":
            representation = global_mean_pool(x, batch=data.batch)
        else:
            representation = x[data.id.nonzero().flatten()]

        return representation, sub_representation, attn_layer
