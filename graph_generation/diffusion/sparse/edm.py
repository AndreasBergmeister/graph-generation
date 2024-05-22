import numpy as np
import torch as th
from torch.nn import Module
from torch_geometric.utils import sort_edge_index


class EDMModel(Module):
    """Preconditioning model for EDM with optional self-conditioning."""

    def __init__(self, self_conditioning):
        super().__init__()
        self.self_conditioning = self_conditioning

    def forward(
        self,
        edge_index,
        batch,
        node_attr,
        edge_attr,
        t,
        model,
        model_kwargs,
        node_attr_self_cond=None,
        edge_attr_self_cond=None,
    ):
        # compute weights
        sigma_data = EDM.sigma_data
        c_in = (1 / (sigma_data**2 + t**2).sqrt())[:, None]
        c_skip = (sigma_data**2 / (t**2 + sigma_data**2))[:, None]
        c_out = (t * sigma_data / (t**2 + sigma_data**2).sqrt())[:, None]

        # compute input
        model_kwargs = dict(model_kwargs, noise_cond=(t.log() / 4.0).float())
        node_attr_in = c_in[batch] * node_attr
        edge_batch = batch[edge_index[0]]
        edge_attr_in = c_in[edge_batch] * edge_attr

        # self-conditioning
        if self.self_conditioning:
            if not model.training:
                assert (
                    node_attr_self_cond is not None and edge_attr_self_cond is not None
                )
            elif np.random.rand() < 0.5:
                # compute self-conditioning
                with th.no_grad():
                    node_attr_self_cond, edge_attr_self_cond = model(
                        edge_index=edge_index,
                        batch=batch,
                        node_attr=th.cat(
                            [node_attr_in, th.zeros_like(node_attr_in)], dim=-1
                        ).float(),
                        edge_attr=th.cat(
                            [edge_attr_in, th.zeros_like(edge_attr_in)], dim=-1
                        ).float(),
                        **model_kwargs,
                    )
                    node_attr_self_cond = (
                        c_skip[batch] * node_attr + c_out[batch] * node_attr_self_cond
                    ).detach()
                    edge_attr_self_cond = (
                        c_skip[edge_batch] * edge_attr
                        + c_out[edge_batch] * edge_attr_self_cond
                    ).detach()
            else:
                node_attr_self_cond = th.zeros_like(node_attr)
                edge_attr_self_cond = th.zeros_like(edge_attr)

            # scale self-conditioning
            node_attr_self_cond = node_attr_self_cond / sigma_data
            edge_attr_self_cond = edge_attr_self_cond / sigma_data

            # concatenate with input
            node_attr_in = th.cat([node_attr_in, node_attr_self_cond], dim=-1)
            edge_attr_in = th.cat([edge_attr_in, edge_attr_self_cond], dim=-1)

        # compute output
        node_attr_pred, edge_attr_pred = model(
            edge_index=edge_index,
            batch=batch,
            node_attr=node_attr_in.float(),
            edge_attr=edge_attr_in.float(),
            **model_kwargs,
        )
        node_attr_pred = c_skip[batch] * node_attr + c_out[batch] * node_attr_pred
        edge_attr_pred = (
            c_skip[edge_batch] * edge_attr + c_out[edge_batch] * edge_attr_pred
        )

        return node_attr_pred, edge_attr_pred


class EDM:
    P_mean = -1.2
    P_std = 1.2
    sigma_data = 0.5
    sigma_min = 0.002
    sigma_max = 80
    rho = 7
    S_min = 0.05
    S_max = 50
    S_noise = 1.003
    S_churn = 40

    def __init__(self, self_conditioning, num_steps):
        self.model_wrapper = EDMModel(self_conditioning)
        self.num_steps = num_steps

    @property
    def device(self):
        assert hasattr(self, "_device")
        return self._device

    def to(self, device):
        self._device = device
        self.model_wrapper.to(device)
        return self

    def get_loss(self, edge_index, batch, node_attr, edge_attr, model, model_kwargs):
        # rescale attributes to {-1, 1}
        node_attr = node_attr.float() * 2 - 1
        edge_attr = edge_attr.float() * 2 - 1

        # sample noise level
        num_graphs = batch.max().item() + 1
        rnd_normal = th.randn((num_graphs,), device=self.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()

        # sample noise
        edge_batch = batch[edge_index[0]]
        node_noise = th.randn_like(node_attr) * t[batch]
        edge_noise = self.edge_randn(edge_index) * t[edge_batch]

        # make prediction
        node_pred, edge_pred = self.model_wrapper(
            edge_index=edge_index,
            batch=batch,
            node_attr=(node_attr + node_noise).unsqueeze(1),
            edge_attr=(edge_attr + edge_noise).unsqueeze(1),
            t=t,
            model=model,
            model_kwargs=model_kwargs,
        )
        node_pred = node_pred.float().squeeze(1)
        edge_pred = edge_pred.float().squeeze(1)

        # compute loss
        weight = (t**2 + self.sigma_data**2) / (t * self.sigma_data) ** 2
        node_loss = weight[batch] * (node_pred - node_attr) ** 2
        edge_loss = weight[edge_batch] * (edge_pred - edge_attr) ** 2

        return node_loss, edge_loss

    @th.no_grad()
    def sample(self, edge_index, batch, model, model_kwargs):
        num_graphs = batch.max().item() + 1

        # time step discretization
        step_indices = th.arange(self.num_steps, dtype=th.float64, device=self.device)
        t_steps = (
            self.sigma_max ** (1 / self.rho)
            + step_indices
            / (self.num_steps - 1)
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = th.cat([t_steps, th.zeros_like(t_steps[:1])])  # t_N = 0

        # sample latents
        node_attr_next = th.randn_like(batch, dtype=th.float64)[:, None] * t_steps[0]
        edge_attr_next = (
            self.edge_randn(edge_index, dtype=th.float64)[:, None] * t_steps[0]
        )

        node_attr_pred = th.zeros_like(node_attr_next)
        edge_attr_pred = th.zeros_like(edge_attr_next)
        # sample loop
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            node_attr_cur = node_attr_next
            edge_attr_cur = edge_attr_next

            # increase noise temporarily
            gamma = (
                min(self.S_churn / self.num_steps, np.sqrt(2) - 1)
                if self.S_min <= t_cur <= self.S_max
                else 0
            )
            t_hat = t_cur + gamma * t_cur
            node_attr_hat = node_attr_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * self.S_noise * th.randn_like(node_attr_cur)
            edge_attr_hat = (
                edge_attr_cur
                + (t_hat**2 - t_cur**2).sqrt()
                * self.S_noise
                * self.edge_randn(edge_index, dtype=th.float64)[:, None]
            )

            # Euler step
            node_attr_pred, edge_attr_pred = self.model_wrapper(
                edge_index=edge_index,
                batch=batch,
                node_attr=node_attr_hat,
                edge_attr=edge_attr_hat,
                t=t_hat.repeat(num_graphs),
                model=model,
                model_kwargs=model_kwargs,
                node_attr_self_cond=node_attr_pred,
                edge_attr_self_cond=edge_attr_pred,
            )
            node_attr_d = (node_attr_hat - node_attr_pred) / t_hat
            edge_attr_d = (edge_attr_hat - edge_attr_pred) / t_hat
            node_attr_next = node_attr_hat + (t_next - t_hat) * node_attr_d
            edge_attr_next = edge_attr_hat + (t_next - t_hat) * edge_attr_d

            # 2nd order correction
            if i < self.num_steps - 1:
                node_attr_pred, edge_attr_pred = self.model_wrapper(
                    edge_index=edge_index,
                    batch=batch,
                    node_attr=node_attr_next,
                    edge_attr=edge_attr_next,
                    t=t_next.repeat(num_graphs),
                    model=model,
                    model_kwargs=model_kwargs,
                    node_attr_self_cond=node_attr_pred,
                    edge_attr_self_cond=edge_attr_pred,
                )
                node_attr_d_prime = (node_attr_next - node_attr_pred) / t_next
                edge_attr_d_prime = (edge_attr_next - edge_attr_pred) / t_next
                node_attr_next = node_attr_hat + (t_next - t_hat) * (
                    0.5 * node_attr_d + 0.5 * node_attr_d_prime
                )
                edge_attr_next = edge_attr_hat + (t_next - t_hat) * (
                    0.5 * edge_attr_d + 0.5 * edge_attr_d_prime
                )

        # rescale attributes to {0, 1}
        node_attr_out = (node_attr_next + 1) / 2
        edge_attr_out = (edge_attr_next + 1) / 2

        return node_attr_out.squeeze(1), edge_attr_out.squeeze(1)

    @staticmethod
    def edge_randn(edge_index, dtype=th.float32) -> th.Tensor:
        """Sample symmetric Gaussian noise for edges in edge_index."""
        # sample noise for upper triangle
        edge_index_u = edge_index[:, edge_index[0] <= edge_index[1]]
        edge_noise_u = th.randn_like(edge_index_u[0], dtype=dtype)

        # make symmetric
        new_edge_index, edge_noise = sort_edge_index(
            edge_index=th.cat([edge_index_u, edge_index_u.flip(0)], dim=1),
            edge_attr=th.cat([edge_noise_u, edge_noise_u], dim=0),
        )
        assert (edge_index == new_edge_index).all()

        return edge_noise
