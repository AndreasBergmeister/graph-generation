import networkx as nx
import torch as th
from torch.nn import Module
from torch_geometric.utils import to_edge_index
from torch_scatter import scatter
from torch_sparse import SparseTensor

from .method import Method


class Expansion(Method):
    """Graph generation method generating graphs by local expansion."""

    def __init__(
        self,
        diffusion,
        spectrum_extractor,
        emb_features,
        augmented_radius=1,
        augmented_dropout=0.0,
        deterministic_expansion=False,
        min_red_frac=0.0,
        max_red_frac=0.5,
        red_threshold=0,
    ):
        self.diffusion = diffusion
        self.spectrum_extractor = spectrum_extractor
        self.emb_features = emb_features
        self.augmented_radius = augmented_radius
        self.augmented_dropout = augmented_dropout
        self.deterministic_expansion = deterministic_expansion
        self.min_red_frac = min_red_frac
        self.max_red_frac = max_red_frac
        self.red_threshold = red_threshold

    def sample_graphs(self, target_size, model: Module, sign_net: Module):
        """Samples a batch of graphs."""
        num_graphs = len(target_size)
        adj = SparseTensor.from_dense(
            th.zeros((num_graphs, num_graphs), device=self.device)
        )

        batch = th.arange(num_graphs, device=self.device)
        node_expansion = th.ones(num_graphs, dtype=th.long, device=self.device)

        while adj.size(0) < target_size.sum():
            adj, batch, node_expansion = self.expand(
                adj,
                batch,
                node_expansion,
                target_size,
                model=model,
                sign_net=sign_net,
            )
            if node_expansion.max() <= 1:
                break

        # return graphs
        adjs = unbatch_adj(adj, batch)
        graphs = [
            nx.from_scipy_sparse_array(adj.to_scipy(layout="coo").astype(bool))
            for adj in adjs
        ]
        return graphs

    @th.no_grad()
    def expand(
        self,
        adj_reduced,
        batch_reduced,
        node_expansion,
        target_size,
        model: Module,
        sign_net: Module,
    ):
        """Expands a graph by a single level."""
        reduced_size = scatter(th.ones_like(batch_reduced), batch_reduced)

        # get node embeddings
        if self.spectrum_extractor is not None:
            spectral_features = th.cat(
                [
                    th.tensor(
                        self.spectrum_extractor(adj.to("cpu").to_scipy(layout="coo")),
                        dtype=th.float32,
                        device=self.device,
                    )
                    for adj in unbatch_adj(adj_reduced, batch_reduced)
                ]
            )
            node_emb_reduced = sign_net(
                spectral_features=spectral_features, edge_index=adj_reduced
            )
        else:
            node_emb_reduced = th.randn(
                adj_reduced.size(0), self.emb_features, device=self.device
            )

        # expand
        # don't expand graphs reached their target size
        node_expansion[(reduced_size >= target_size)[batch_reduced]] = 1
        node_map = th.repeat_interleave(
            th.arange(0, adj_reduced.size(0), device=self.device), node_expansion
        )
        node_emb = node_emb_reduced[node_map]
        batch = batch_reduced[node_map]
        size = scatter(th.ones_like(batch), batch)
        expansion_matrix = SparseTensor(
            row=th.arange(node_map.size(0), device=self.device),
            col=node_map,
            value=th.ones(node_map.size(0), device=self.device),
        )
        adj_augmented = self.get_augmented_graph(adj_reduced, expansion_matrix)
        augmented_edge_index = th.stack(adj_augmented.coo()[:2], dim=0)

        # compute number of nodes in expanded graph
        random_reduction_fraction = (
            th.rand(len(target_size), device=self.device)
            * (self.max_red_frac - self.min_red_frac)
            + self.min_red_frac
        )

        # if expanded number of nodes is less than threshold, use max_red_frac
        max_reduction_mask = (
            th.ceil(size / (1 - self.max_red_frac)) <= self.red_threshold
        ).float()
        random_reduction_fraction = (
            1 - max_reduction_mask
        ) * random_reduction_fraction + max_reduction_mask * self.max_red_frac

        # expanded number of nodes is ⌈n / (1-r)⌉ and at least n+1 and at most target_size
        expanded_size = th.minimum(
            th.maximum(
                th.ceil(size / (1 - random_reduction_fraction)).long(),
                size + 1,
            ),
            target_size,
        )

        # make predictions
        node_pred, augmented_edge_pred = self.diffusion.sample(
            edge_index=augmented_edge_index,
            batch=batch,
            model=model,
            model_kwargs={
                "node_emb": node_emb,
                "red_frac": 1 - size / expanded_size,
                "target_size": target_size.float(),
            },
        )

        # get node attributes
        if self.deterministic_expansion:
            node_attr = th.zeros_like(node_pred, dtype=th.long)
            num_new_nodes = expanded_size - size
            node_range_end = size.cumsum(0)
            node_range_start = node_range_end - size
            # get top-k nodes per graph
            for i in range(len(target_size)):
                new_node_idx = (
                    th.topk(
                        node_pred[node_range_start[i] : node_range_end[i]],
                        num_new_nodes[i],
                        largest=True,
                    )[1]
                    + node_range_start[i]
                )
                node_attr[new_node_idx] = 1
        else:
            node_attr = (node_pred > 0.5).long()

        # construct new graph
        adj = SparseTensor.from_edge_index(
            augmented_edge_index[:, augmented_edge_pred > 0.5],
            sparse_sizes=adj_augmented.sizes(),
        )

        return adj, batch, node_attr + 1

    def get_loss(self, batch, model: Module, sign_net: Module):
        """Returns a weighted sum of the node expansion loss and the augmented edge loss."""
        # get augmented graph
        adj_augmented = self.get_augmented_graph(
            batch.adj_reduced, batch.expansion_matrix
        )

        # construct labels
        node_attr = batch.node_expansion - 1
        augmented_edge_index, edge_val = to_edge_index(adj_augmented + batch.adj)
        augmented_edge_attr = edge_val.long() - 1

        # get node embeddings
        if sign_net is not None:
            node_emb_reduced = sign_net(
                spectral_features=batch.spectral_features_reduced,
                edge_index=batch.adj_reduced,
            )
            node_emb = batch.expansion_matrix @ node_emb_reduced
        else:
            node_emb = th.randn(
                adj_augmented.size(0), self.emb_features, device=self.device
            )

        # reduction fraction
        size = scatter(th.ones_like(batch.batch), batch.batch)
        expanded_size = scatter(batch.node_expansion, batch.batch)
        red_frac = 1 - size / expanded_size

        # loss
        node_loss, edge_loss = self.diffusion.get_loss(
            edge_index=augmented_edge_index,
            batch=batch.batch,
            node_attr=node_attr,
            edge_attr=augmented_edge_attr,
            model=model,
            model_kwargs={
                "node_emb": node_emb,
                "red_frac": red_frac,
                "target_size": batch.target_size.float(),
            },
        )

        # ignore node_loss for first level
        node_loss = node_loss[batch.reduction_level[batch.batch] > 0].mean()
        edge_loss = edge_loss.mean()
        loss = node_loss + edge_loss

        return loss, {
            "node_expansion_loss": node_loss.item(),
            "augmented_edge_loss": edge_loss.item(),
            "loss": loss.item(),
        }

    def get_augmented_graph(self, adj_reduced, expansion_matrix):
        """Returns the expanded adjacency matrix with additional augmented edges.

        All edge weights are set to 1.
        """
        # construct augmented adjacency matrix
        adj_reduced = adj_reduced.set_diag(1)
        adj_reduced_augmented = adj_reduced.copy()
        for _ in range(1, self.augmented_radius):
            adj_reduced_augmented = adj_reduced_augmented @ adj_reduced

        adj_reduced_augmented = adj_reduced_augmented.set_value(
            th.ones(adj_reduced_augmented.nnz(), device=self.device), layout="coo"
        )
        adj_augmented = (
            expansion_matrix @ adj_reduced_augmented @ expansion_matrix.t()
        ).remove_diag()
        adj_expanded = (
            expansion_matrix @ adj_reduced @ expansion_matrix.t()
        ).remove_diag()

        # drop out edges
        if self.augmented_dropout > 0.0:
            adj_required = adj_augmented + adj_expanded
            row, col, val = adj_required.coo()
            edge_mask = th.rand_like(val) >= self.augmented_dropout
            edge_mask = edge_mask | (val > 1)  # keep required edges
            # make undirected
            edge_mask = edge_mask & (row < col)
            edge_index = th.stack([row[edge_mask], col[edge_mask]], dim=0)
            edge_index = th.cat([edge_index, edge_index.flip(0)], dim=1)
            adj_augmented = SparseTensor.from_edge_index(
                edge_index,
                edge_attr=th.ones(edge_index.shape[1], device=self.device),
                sparse_sizes=adj_augmented.sizes(),
            )

        return adj_augmented


def unbatch_adj(adj, batch) -> list:
    size = scatter(th.ones_like(batch), batch)
    graph_end_idx = size.cumsum(0)
    graph_start_idx = graph_end_idx - size
    return [
        adj[graph_start_idx[i] : graph_end_idx[i], :][
            :, graph_start_idx[i] : graph_end_idx[i]
        ]
        for i in range(len(size))
    ]
