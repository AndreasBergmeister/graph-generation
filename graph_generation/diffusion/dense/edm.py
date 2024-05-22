import numpy as np
import torch as th
from torch.nn import Module


class EDMModel(Module):
    """Preconditioning model for EDM with optional self-conditioning.

    Operates on a dense graph representation.
    """

    def __init__(self, self_conditioning):
        super().__init__()
        self.self_conditioning = self_conditioning

    def forward(self, x, mask, t, model, x_self_cond=None):
        """
        Shape:
            x: :math:`(N, n, n, d)`
            mask: :math:`(N, n, n, 1)`
            t: :math:`(N,)`
        """
        # masks
        n = mask.shape[1]
        mask_diag = 1 - th.eye(n, device=mask.device, dtype=th.float64).view(1, n, n, 1)

        # compute weights
        sigma_data = EDM.sigma_data
        c_in = (1 / (sigma_data**2 + t**2).sqrt())[:, None, None, None]
        c_skip = (sigma_data**2 / (t**2 + sigma_data**2))[:, None, None, None]
        c_out = (t * sigma_data / (t**2 + sigma_data**2).sqrt())[
            :, None, None, None
        ]

        # compute input
        noise_cond = (t.log() / 4.0).float()
        x_in = c_in * x * mask * mask_diag

        # self-conditioning
        if self.self_conditioning:
            if not model.training:
                assert x_self_cond is not None
            elif np.random.rand() < 0.5:
                # compute self-conditioning
                with th.no_grad():
                    x_self_cond = model(
                        x=th.cat([x_in, th.zeros_like(x_in)], dim=-1).float(),
                        mask=mask.float(),
                        noise_cond=noise_cond,
                    )
                    x_self_cond = (c_skip * x + c_out * x_self_cond).detach()
            else:
                x_self_cond = th.zeros_like(x)

            # scale self-conditioning
            x_self_cond = x_self_cond / sigma_data * mask * mask_diag

            # concatenate with input
            x_in = th.cat([x_in, x_self_cond], dim=-1)

        # compute output
        x_pred = model(x=x_in.float(), mask=mask.float(), noise_cond=noise_cond)
        x_pred = c_skip * x + c_out * x_pred

        return x_pred


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

    def get_loss(self, x, mask, model):
        # masks
        n = mask.shape[1]
        mask = mask.float()
        mask_diag = 1 - th.eye(n, device=mask.device, dtype=th.float32).view(1, n, n)

        # rescale attributes to {-1, 1}
        x = x.float() * 2 - 1

        # sample noise level
        num_graphs = x.shape[0]
        rnd_normal = th.randn((num_graphs,), device=self.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()

        # sample noise
        noise = (
            self.sym_randn(num_graphs, n, dtype=th.float32, device=self.device)
            * t[:, None, None]
        )

        # make prediction
        x_pred = self.model_wrapper(
            x=(x + noise)[..., None], mask=mask[..., None], t=t, model=model
        )
        x_pred = x_pred.float().squeeze(-1)

        # compute loss
        weight = (t**2 + self.sigma_data**2) / (t * self.sigma_data) ** 2
        loss = weight[:, None, None] * (x_pred - x) ** 2
        loss_mask = mask * mask_diag
        loss = (loss * loss_mask).sum() / loss_mask.sum()

        return loss

    @th.no_grad()
    def sample(self, mask, model):
        num_graphs = mask.shape[0]

        # masks
        n = mask.shape[1]
        mask = mask.double()[..., None]
        mask_diag = 1 - th.eye(n, device=mask.device, dtype=th.float64).view(1, n, n, 1)

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
        x_next = (
            self.sym_randn(num_graphs, n, dtype=th.float64, device=self.device)[
                ..., None
            ]
            * t_steps[0]
        )

        x_pred = th.zeros_like(x_next)
        # sample loop
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            # increase noise temporarily
            gamma = (
                min(self.S_churn / self.num_steps, np.sqrt(2) - 1)
                if self.S_min <= t_cur <= self.S_max
                else 0
            )
            t_hat = t_cur + gamma * t_cur
            x_hat = (
                x_cur
                + (t_hat**2 - t_cur**2).sqrt()
                * self.S_noise
                * self.sym_randn(num_graphs, n, dtype=th.float64, device=self.device)[
                    ..., None
                ]
            )

            # Euler step
            x_pred = self.model_wrapper(
                x=x_hat,
                mask=mask,
                t=t_hat.repeat(num_graphs),
                model=model,
                x_self_cond=x_pred,
            )
            x_d = (x_hat - x_pred) / t_hat
            x_next = x_hat + (t_next - t_hat) * x_d

            # 2nd order correction
            if i < self.num_steps - 1:
                x_pred = self.model_wrapper(
                    x=x_next,
                    mask=mask,
                    t=t_next.repeat(num_graphs),
                    model=model,
                    x_self_cond=x_pred,
                )
                x_d_prime = (x_next - x_pred) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * x_d + 0.5 * x_d_prime)

        # # rescale attributes to {0, 1}
        # x_out = (x_next + 1) / 2

        # threshold
        x_out = (x_next > 0).float()

        # mask
        x_out = x_out * mask * mask_diag

        return x_out.squeeze(-1)

    @staticmethod
    def sym_randn(N, n, dtype=None, device=None):
        """Sample symmetric noise."""
        noise = th.randn((N, n, n), dtype=dtype, device=device)
        noise_triu = th.triu(noise, diagonal=1)
        noise_sym = noise_triu + noise_triu.transpose(1, 2)
        return noise_sym
