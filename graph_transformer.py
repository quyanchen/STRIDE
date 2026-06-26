
import torch
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class GraphTransformerEncode(torch.nn.Module):
    def __init__(self, num_heads, in_dim, dim_forward, rel_encoder, spatial_encoder, dropout):
        super(GraphTransformerEncode, self).__init__()

        self.num_heads = num_heads
        self.in_dim = in_dim
        self.dim_forward = dim_forward

        self.ffn = Sequential(
            Linear(self.in_dim, self.dim_forward),
            ReLU(),
            Linear(self.dim_forward, self.in_dim)
        )

        self.multiHeadAttention = MultiheadAttention(dim_model = self.in_dim, num_heads = self.num_heads, rel_encoder=rel_encoder, spatial_encoder = spatial_encoder)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def reset_parameters(self):
        self.ffn[0].reset_parameters()
        self.ffn[2].reset_parameters()

        self.multiHeadAttention.reset_parameters()
        self.layernorm1.reset_parameters()
        self.layernorm2.reset_parameters()

    def forward(self, feature, sp_edge_index, sp_value, edge_rel):

        x_norm = self.layernorm1(feature)
        attn_output, attn_weight = self.multiHeadAttention(x_norm, sp_edge_index, sp_value, edge_rel)
        attn_output = self.dropout1(attn_output)
        out1 = attn_output + feature

        residual = out1
        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout2(ffn_output)
        out2 = residual + ffn_output

        return out2, attn_weight

class SpatialEncoding(torch.nn.Module):
    def __init__(self, dim_model):
        super(SpatialEncoding, self).__init__()

        self.dim = dim_model
        self.fnn = Sequential(
            Linear(1, dim_model),
            ReLU(),
            Linear(dim_model, 1),
            ReLU()
        )

    def reset_parameters(self):
        self.fnn[0].reset_parameters()
        self.fnn[2].reset_parameters()

    def forward(self, lap):
        lap_ = torch.unsqueeze(
            lap.to(dtype=self.fnn[0].weight.dtype),
            dim=-1,
        )
        out = self.fnn(lap_)

        return out


class MultiheadAttention(MessagePassing):
    def __init__(self, dim_model, num_heads, rel_encoder, spatial_encoder, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.d_model = dim_model
        self.num_heads = num_heads

        self.rel_embedding = rel_encoder
        self.spatial_encoding = spatial_encoder


        assert dim_model % num_heads == 0
        self.depth = self.d_model // num_heads

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
            -1, self.num_heads, self.depth
        )
        q = self.wq(x).view(x.shape[0], self.num_heads, self.depth)
        k = self.wk(x).view(x.shape[0], self.num_heads, self.depth)
        v = self.wv(x).view(x.shape[0], self.num_heads, self.depth)

        row, col = sp_edge_index
        query_end = q[col] + rel_embedding
        key_start = k[row] + rel_embedding
        attention_score = (
            query_end * key_start
        ).sum(dim=-1) / (self.depth ** 0.5)
        spatial_bias = self.spatial_encoding(sp_value).squeeze(-1).unsqueeze(-1)
        attention_score = attention_score + spatial_bias
        attention_weight = softmax(
            attention_score,
            index=col,
            num_nodes=x.shape[0],
        )

        outputs = []
        for i in range(self.num_heads):
            output_per_head = self.propagate(
                edge_index=sp_edge_index,
                x=v[:, i, :],
                edge_weight=attention_weight[:, i],
                size=None,
            )
            outputs.append(output_per_head)

        out = torch.cat(outputs,dim=-1)

        return self.dense(out), attention_weight

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.unsqueeze(-1)


class GraphTransformer(torch.nn.Module):
    def __init__(self, layer_num = 3, embedding_dim = 64, num_heads = 4, num_rel = 10, dropout = 0.2, type = 'graph'):
        super(GraphTransformer, self).__init__()

        self.type = type
        self.rel_encoder = torch.nn.Embedding(num_rel, embedding_dim)
        self.spatial_encoder = SpatialEncoding(embedding_dim)
        self.encoder = torch.nn.ModuleList()
        if layer_num < 1:
            raise ValueError("layer_num must be at least 1")
        for _ in range(layer_num):
            self.encoder.append(GraphTransformerEncode(num_heads = num_heads, in_dim = embedding_dim, dim_forward = embedding_dim*2,
                                                       rel_encoder = self.rel_encoder, spatial_encoder = self.spatial_encoder, dropout=dropout))

    def reset_parameters(self):
        for e in self.encoder:
            e.reset_parameters()


    def forward(self, feature, data):


        x = feature
        graph_embedding_layer = []
        attn_layer = []
        for graphEncoder in self.encoder:
            x, attn = graphEncoder(x, data.sp_edge_index, data.sp_value, data.sp_edge_rel)
            graph_embedding_layer.append(x)
            attn_layer.append(attn)

        #all_out = torch.stack([x for x in graph_embedding_layer])


        if self.type == 'graph':
            sub_representation = []
            for index, _ in enumerate(data.to_data_list()):
                sub_embedding = x[(data.batch == index).nonzero().flatten()]
                sub_representation.append(sub_embedding)
            representation = global_mean_pool(x, batch=data.batch)
        else:
            sub_representation = []
            for index, _ in enumerate(data.to_data_list()):
                sub_embedding = x[(data.batch == index).nonzero().flatten()]
                sub_representation.append(sub_embedding)
            representation = x[data.id.nonzero().flatten()]

        return representation, sub_representation, attn_layer
