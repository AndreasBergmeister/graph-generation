from abc import ABC

import numpy as np
import scipy as sp
import torch as th
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.typing import SparseTensor

from ..reduction import ReductionFactory


class RandRedDataset(IterableDataset, ABC):
    def __init__(self, adjs, red_factory: ReductionFactory, spectrum_extractor):
        super().__init__()

        self.red_factory = red_factory
        self.adjs = adjs
        self.spectrum_extractor = spectrum_extractor

    def get_random_reduction_sequence(self, graph, rng):
        data = []
        while True:
            reduced_graph = graph.get_reduced_graph(rng)
            data.append(
                ReducedGraphData(
                    target_size=graph.n,
                    reduction_level=graph.level,
                    adj=graph.adj.astype(bool).astype(np.float32),
                    node_expansion=graph.node_expansion,
                    adj_reduced=reduced_graph.adj.astype(bool).astype(np.float32),
                    expansion_matrix=reduced_graph.expansion_matrix,
                    spectral_features_reduced=self.spectrum_extractor(reduced_graph.adj)
                    if self.spectrum_extractor is not None
                    else None,
                )
            )
            if graph.n <= 1:
                break
            graph = reduced_graph

        return data


class FiniteRandRedDataset(RandRedDataset):
    def __init__(
        self, adjs, red_factory: ReductionFactory, spectrum_extractor, num_red_seqs
    ):
        super().__init__(adjs, red_factory, spectrum_extractor)
        self.num_red_seqs = num_red_seqs

        self.rng = np.random.default_rng(seed=0)
        self.graph_reduced_data = {i: [] for i in range(len(adjs))}
        for i, adj in enumerate(adjs):
            graph = red_factory(adj)
            for _ in range(num_red_seqs):
                self.graph_reduced_data[i] += self.get_random_reduction_sequence(
                    graph, self.rng
                )

    def __iter__(self):
        while True:
            i = self.rng.integers(len(self.adjs))
            j = self.rng.integers(len(self.graph_reduced_data[i]))
            yield self.graph_reduced_data[i][j]

    @property
    def max_node_expansion(self):
        return max(
            [
                rgd.node_expansion.max().item()
                for seq in self.graph_reduced_data
                for rgd in seq
            ]
        )


class InfiniteRandRedDataset(RandRedDataset):
    def __iter__(self):
        graphs = [self.red_factory(adj.copy()) for adj in self.adjs]

        # get process id
        worker_info = th.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(worker_id)

        # initialize graph_reduced_data
        graph_reduced_data = {
            i: self.get_random_reduction_sequence(graph, rng)
            for i, graph in enumerate(graphs)
        }

        # yield random reduced graph data
        while True:
            i = rng.integers(len(graphs))
            if len(graph_reduced_data[i]) == 0:
                graph_reduced_data[i] = self.get_random_reduction_sequence(
                    graphs[i], rng
                )
                rng.shuffle(graph_reduced_data[i])

            yield graph_reduced_data[i].pop()

    @property
    def max_node_expansion(self):
        raise NotImplementedError


class ReducedGraphData(Data):
    def __init__(self, **kwargs):
        if not kwargs:
            super().__init__()
            return

        super().__init__(x=th.zeros(kwargs["adj"].shape[0]))
        for key, value in kwargs.items():
            if value is None:
                continue
            elif isinstance(value, int):
                value = th.tensor(value).type(th.long)
            elif isinstance(value, np.ndarray):
                value = th.from_numpy(value).type(
                    th.float32 if np.issubdtype(value.dtype, np.floating) else th.long
                )
            elif isinstance(value, sp.sparse.sparray):
                value = SparseTensor.from_scipy(value).type(
                    th.float32 if np.issubdtype(value.dtype, np.floating) else th.long
                )
            else:
                raise ValueError(f"Unsupported type {type(value)} for key {key}")

            setattr(self, key, value)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)  # concatenate along diagonal
        return super().__cat_dim__(key, value, *args, **kwargs)
