import networkx as nx
import torch as th

from .method import Method


class OneShot(Method):
    """One-shot method for graph generation."""

    def __init__(self, diffusion):
        self.diffusion = diffusion

    def sample_graphs(self, target_size, model, **kwargs):
        # construct mask
        mask_1d = (
            th.arange(target_size.max(), device=target_size.device)[None, :]
            < target_size[:, None]
        )  # N, n
        mask = mask_1d[:, None, :] & mask_1d[:, :, None]  # N, n, n

        # sample
        adjs = self.diffusion.sample(mask=mask, model=model)

        # convert to graphs
        graphs = []
        for i in range(adjs.shape[0]):
            n = target_size[i]
            adj = adjs[i, :n, :n]
            graphs.append(nx.from_numpy_array(adj.numpy(force=True).astype(bool)))

        return graphs

    def get_loss(self, batch, model, **kwargs):
        loss = self.diffusion.get_loss(x=batch.adj, mask=batch.mask, model=model)
        return loss, {"loss": loss.item()}
