import torch as th
from torch.nn import Dropout, Linear, Module, ModuleList
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.encoding import PositionalEncoding
from torch_geometric.utils import coalesce

from .mlp import MLP


class GINE(Module):
    """Graph Isomorphism Network (GIN) model with edge features.

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
        self.gine_layers = ModuleList(
            [
                GINEConv(MLP(hidden_features, [hidden_features, hidden_features]))
                for _ in range(num_layers)
            ]
        )
        self.edge_layers = ModuleList(
            [
                MLP(3 * hidden_features, [hidden_features, hidden_features])
                for _ in range(num_layers)
            ]
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

        skip_node = [x_node]
        skip_edge = [x_edge]
        for gin_layer, edge_layer in zip(self.gine_layers, self.edge_layers):
            x_node = gin_layer(x=x_node, edge_index=edge_index, edge_attr=x_edge)
            skip_node.append(x_node)
            x_edge = edge_layer(
                th.cat([x_edge, x_node[edge_index[0]], x_node[edge_index[1]]], dim=-1)
            )
            skip_edge.append(x_edge)

        # Skip layer
        x_node = th.cat(skip_node, dim=-1)
        x_node = self.dropout(x_node)
        x_edge = th.cat(skip_edge, dim=-1)
        x_edge = self.dropout(x_edge)

        # Out layers
        out_node = self.node_out_layer(x_node)
        out_edge = self.edge_out_layer(x_edge)

        # make out_edge symmetric
        out_edge = coalesce(
            th.cat([edge_index, edge_index.flip(0)], dim=-1),
            th.cat([out_edge, out_edge], dim=0),
            reduce="mean",
        )[1]

        return out_node, out_edge
