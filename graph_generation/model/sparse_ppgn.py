import torch as th
from torch.nn import Dropout, Linear, Module, ModuleList
from torch_geometric.nn.encoding import PositionalEncoding
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import coalesce
from torch_scatter import scatter

from .mlp import MLP


class SparsePPGN(Module):
    """Our proposed SparsePPGN model.

    Operates on a sparse graph representation.
    """

    def __init__(
        self,
        node_in_features: int,
        edge_in_features: int,
        node_out_features: int,
        edge_out_features: int,
        emb_features: int,
        hidden_features: int,
        ppgn_features: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Embedding layers
        self.node_emb_layer = Linear(node_in_features, emb_features)
        self.edge_emb_layer = Linear(edge_in_features, emb_features)
        self.noise_cond_emb_layer = Linear(1, emb_features)
        self.red_frac_emb_layer = Linear(1, emb_features)
        self.target_size_emb_layer = PositionalEncoding(emb_features)

        # In layers
        self.node_in_mlp = MLP(5 * emb_features, [hidden_features, hidden_features])
        self.edge_in_mlp = MLP(6 * emb_features, [hidden_features, hidden_features])

        # GNN layers
        self.sparse_ppgn_layers = ModuleList(
            [SparsePPGNLayer(hidden_features, ppgn_features) for _ in range(num_layers)]
        )

        # Out layers
        self.node_out_layer = Linear(
            (num_layers + 1) * hidden_features, node_out_features
        )
        self.edge_out_layer = Linear(
            (num_layers + 1) * hidden_features, edge_out_features
        )

        # Dropout
        self.dropout = Dropout(dropout)

    def forward(
        self,
        edge_index,
        batch,
        node_attr,
        edge_attr,
        node_emb,
        noise_cond,
        red_frac,
        target_size,
    ):
        # Embedding
        node_attr_emb = self.node_emb_layer(node_attr)
        edge_attr_emb = self.edge_emb_layer(edge_attr)
        noise_cond_emb = self.noise_cond_emb_layer(noise_cond[..., None])
        red_frac_emb = self.red_frac_emb_layer(red_frac[..., None])
        target_size_emb = self.target_size_emb_layer(target_size[..., None])

        # Input
        x_node = th.cat(
            [
                node_attr_emb,
                node_emb,
                noise_cond_emb[batch],
                red_frac_emb[batch],
                target_size_emb[batch],
            ],
            dim=-1,
        )
        x_node = self.dropout(x_node)
        x_node = self.node_in_mlp(x_node)

        edge_batch = batch[edge_index[0]]
        x_edge = th.cat(
            [
                edge_attr_emb,
                node_emb[edge_index[0]],
                node_emb[edge_index[1]],
                noise_cond_emb[edge_batch],
                red_frac_emb[edge_batch],
                target_size_emb[edge_batch],
            ],
            dim=-1,
        )
        x_edge = self.dropout(x_edge)
        x_edge = self.edge_in_mlp(x_edge)

        # construct triangle_index
        # the indexed elements are the edges of the triangles (including self-loops)
        # for each triangle (a, b, c) the message x[a] * x[b] is sent to x[c]
        # a = (s,u), b = (u,v), c = (v,t)
        # add self-loops
        self_loop_index = th.arange(node_attr.size(0), device=node_attr.device)[
            None, :
        ].expand(2, -1)
        edge_index_ext = th.cat([self_loop_index, edge_index], dim=1)
        x = th.cat([x_node, x_edge], dim=0)

        n = node_attr.size(0)
        edge_id = edge_index_ext[0] * n + edge_index_ext[1]
        edge_id_to_edge_num = th.full((n * n,), -1, dtype=th.long, device=x.device)
        edge_id_to_edge_num[edge_id] = th.arange(edge_id.size(0), device=x.device)

        rowptr, col = SparseTensor.from_edge_index(
            edge_index_ext, sparse_sizes=(n, n)
        ).csr()[:2]
        out_degrees = rowptr[1:] - rowptr[:-1]
        two_hop = edge_index_ext.repeat_interleave(
            out_degrees[edge_index_ext[1]], dim=1
        )
        offsets = th.arange(two_hop.size(1), device=x.device) - (
            (
                th.cat(
                    [
                        th.zeros_like(out_degrees[0])[None],
                        out_degrees[edge_index_ext[1, :-1]],
                    ]
                )
            ).cumsum(0)
        ).repeat_interleave(out_degrees[edge_index_ext[1]], dim=0)
        two_hop = th.cat([two_hop, col[rowptr[two_hop[1]] + offsets][None]])
        triangles = two_hop[:, edge_id_to_edge_num[two_hop[0] * n + two_hop[2]] >= 0]
        triangle_index = th.stack(
            [
                edge_id_to_edge_num[triangles[0] * n + triangles[1]],
                edge_id_to_edge_num[triangles[1] * n + triangles[2]],
                edge_id_to_edge_num[triangles[0] * n + triangles[2]],
            ]
        )

        # Layers
        num_messages = scatter(
            th.ones(triangle_index.size(1), device=x.device), triangle_index[2], dim=0
        )
        norm_factor = 1.0 / num_messages.sqrt()

        skip = [x]
        for layer in self.sparse_ppgn_layers:
            x = layer(x, triangle_index, norm_factor)
            skip.append(x)

        # Skip layer
        x = th.cat(skip, dim=-1)
        x = self.dropout(x)

        # Out layers
        out_node = self.node_out_layer(x[:n])
        out_edge = self.edge_out_layer(x[n:])

        # make out_edge symmetric
        out_edge = coalesce(
            th.cat([edge_index, edge_index.flip(0)], dim=-1),
            th.cat([out_edge, out_edge], dim=0),
            reduce="mean",
        )[1]

        return out_node, out_edge


class SparsePPGNLayer(Module):
    def __init__(self, hidden_features, ppgn_features):
        super().__init__()

        self.mlp1 = MLP(
            in_features=hidden_features,
            hidden_features=[hidden_features, ppgn_features],
        )
        self.mlp2 = MLP(
            in_features=hidden_features,
            hidden_features=[hidden_features, ppgn_features],
        )
        self.mlp3 = MLP(
            in_features=hidden_features + ppgn_features,
            hidden_features=[hidden_features, hidden_features],
        )

    def forward(self, x, triangle_index, norm_factor):
        m1 = self.mlp1(x)
        m2 = self.mlp2(x)

        m = scatter(
            m1[triangle_index[0]] * m2[triangle_index[1]], triangle_index[2], dim=0
        )
        m = m * norm_factor[:, None]

        x = self.mlp3(th.cat([x, m], dim=-1))
        return x
