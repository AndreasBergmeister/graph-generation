import numpy as np
import torch as th
from torch.utils.data import IterableDataset


class DenseGraphDataset(IterableDataset):
    def __init__(self, adjs):
        super().__init__()

        N: int = max(adj.shape[0] for adj in adjs)

        self.graphs = []
        for adj in adjs:
            padded_adj = np.zeros((N, N), dtype=bool)
            padded_adj[: adj.shape[0], : adj.shape[1]] = adj
            mask = np.zeros((N, N), dtype=bool)
            mask[: adj.shape[0], : adj.shape[1]] = 1.0
            self.graphs.append(GraphData({"adj": padded_adj, "mask": mask}))

    def __iter__(self):
        rng = np.random.default_rng(seed=0)
        while True:
            i = rng.integers(len(self.graphs))
            yield self.graphs[i]


class GraphData(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def to(self, *args, **kwargs):
        for k, v in self.items():
            if isinstance(v, th.Tensor):
                self[k] = v.to(*args, **kwargs)
        return self
