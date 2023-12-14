import numpy as np
import torch as th
from torch.nn import Module
from torch.nn import functional as F


class DiscreteGraphDiffusionModel(Module):
    """Preconditioning for discrete diffusion with optional self-conditioning.

    Operates on a dense graph representation.
    """

    def __init__(self, self_conditioning, num_categories, num_steps):
        super().__init__()
        self.self_conditioning = self_conditioning
        self.num_categories = num_categories
        self.num_steps = num_steps

    def forward(self, x, mask, t, model, x_self_cond=None):
        """
        Shape:
            x: :math:`(N, n, n)`
            mask: :math:`(N, n, n)`
            t: :math: `(N,)`
        """
        # masks
        n = mask.shape[1]
        mask = mask.float()[..., None]
        mask_diag = 1 - th.eye(n, device=mask.device, dtype=th.long).view(1, n, n, 1)

        # embed node and edge attributes
        x_in = (F.one_hot(x, self.num_categories).float() * 2 - 1) * mask * mask_diag

        # self-conditioning
        if self.self_conditioning:
            if not model.training:
                assert x_self_cond is not None
                x_self_cond = th.softmax(x_self_cond, dim=-1) * mask * mask_diag
            elif np.random.rand() < 0.5:
                # sample from next time step
                t_next = th.clamp(t + 1, max=self.num_steps)

                # compute self-conditioning
                with th.no_grad():
                    x_self_cond = model(
                        x=th.cat([x_in, th.zeros_like(x_in)], dim=-1),
                        mask=mask,
                        noise_cond=t_next.float() / self.num_steps,
                    )
                    x_self_cond = (
                        x_self_cond.detach() + F.one_hot(x, self.num_categories).float()
                    )
                    x_self_cond = th.softmax(x_self_cond, dim=-1) * mask * mask_diag

            else:
                x_self_cond = th.zeros_like(x_in)

            # concatenate with input
            x_in = th.cat([x_in, x_self_cond], dim=-1)

        # predict
        x_pred = model(x=x_in, mask=mask, noise_cond=t.float() / self.num_steps)

        return x_pred


class DiscreteGraphDiffusion:
    def __init__(self, self_conditioning, num_steps):
        super().__init__()

        self.model_wrapper = DiscreteGraphDiffusionModel(
            self_conditioning, 2, num_steps
        )
        self.num_steps = num_steps
        self.diffusion = CategoricalDiffusion(2, num_steps)

    @property
    def device(self):
        assert hasattr(self, "_device")
        return self._device

    def to(self, device):
        self._device = device
        self.model_wrapper.to(device)
        self.diffusion.to(device)
        return self

    @th.no_grad()
    def sample(self, mask, model):
        """Generate samples using the model.

        Iteratively sample from p(x_{t-1} | x_t) for t = T-1, ..., 0, starting from x_T ~ p(x_T).

        Shape:
            mask: :math:`(N, n, n)`
        """
        # masks
        n = mask.shape[1]
        mask = mask.long()
        mask_diag = 1 - th.eye(n, device=self.device, dtype=th.long).view(1, n, n)

        # sample from p(x_T)
        x_t = sample_categorical(
            self.diffusion.qT.view(1, 1, 1, -1).expand(*mask.shape, -1)
        )
        x_t = (x_t.triu() + x_t.triu().transpose(-1, -2)) * mask * mask_diag

        # sample from p(x_{t-1} | x_t) for t = T-1, ..., 0
        x_pred = th.zeros(*x_t.shape, 2, device=self.device)
        for timestep in reversed(range(self.num_steps)):
            t = timestep * th.ones(mask.shape[0], dtype=th.long, device=self.device)

            # predict
            x_pred = self.model_wrapper(
                x=x_t, mask=mask, t=t, model=model, x_self_cond=x_pred
            )

            # sample ancestor
            x_t = self.diffusion.q_reverse_sample(x_t, x_pred, t)
            x_t = (x_t.triu() + x_t.triu().transpose(-1, -2)) * mask * mask_diag

        return x_t

    def get_loss(self, x, mask, model):
        """Compute loss to train the model.

        Sample x_pred ~ p(x_t, t), where t ~ U(0, T-1) and x_t ~ q(x_t | x),
        and compute the cross entropy loss between x_pred and x.
        """
        # x assumed to contain discrete labels
        x = x.long()  # N, n, n

        # masks
        n = mask.shape[1]
        mask = mask.long()
        mask_diag = 1 - th.eye(n, device=mask.device, dtype=th.long).view(1, n, n)

        # sample symmetric x_t
        t = th.randint(0, self.num_steps, (x.shape[0],), device=self.device)
        x_t = self.diffusion.q_sample(x, t)
        x_t = (x_t.triu() + x_t.triu().transpose(-1, -2)) * mask * mask_diag

        # predict
        x_pred = self.model_wrapper(
            x=x_t, mask=mask, t=t, model=model
        )  # N, n, n, num_categories

        # compute loss
        loss_mask = (mask * mask_diag).float()
        loss = self.diffusion.get_loss(x, x_pred) * loss_mask
        loss = loss.sum() / loss_mask.sum()

        return loss


class CategoricalDiffusion(Module):
    """Discrete diffusion for categorical variables.

    Uses a uniform transition matrix with cosine schedule.
    """

    def __init__(self, num_categories, num_steps):
        super().__init__()

        # stable distribution
        qT = np.ones(num_categories) / num_categories

        # cosine schedule (https://arxiv.org/pdf/2102.05379.pdf)
        steps = np.arange(num_steps + 1, dtype=np.float64) / num_steps
        alphas_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2)
        betas = np.minimum(1 - alphas_bar[1:] / alphas_bar[:-1], 0.999)

        # single step transition matrices
        Qs = (1 - betas)[:, None, None] * np.eye(num_categories) + betas[
            :, None, None
        ] * qT[None, :, None]

        # t-step transition matrices
        Qbs = (
            alphas_bar[:, None, None] * np.eye(num_categories)
            + (1 - alphas_bar)[:, None, None] * qT[None, :, None]
        )

        # register buffers
        self.register_buffer("qT", th.tensor(qT, dtype=th.float64), persistent=False)
        self.register_buffer("Qs", th.tensor(Qs, dtype=th.float64), persistent=False)
        self.register_buffer("Qbs", th.tensor(Qbs, dtype=th.float64), persistent=False)

    def q_sample(self, x, t):
        """Sample from q(x_t | x).

        Shape:
            x: N, n, n
            t: N
            Qbs: T, num_categories, num_categories
        """
        x_t_prob = self.Qbs[t[:, None, None], x]  # N, n, n, |x_t|
        return sample_categorical(x_t_prob)

    def q_reverse_sample(self, x_t, pred, t):
        """Sample from q(x_{t-1} | x_t) = Σ_x q(x_{t-1} | x, x_t) q(x | x_t)."""
        # compute probs of  posterior q(x_{t-1} | x, x_t) ∝ q(x_t | x_{t-1}) q(x_{t-1} | x) for all x
        left_term = self.Qs[t[:, None, None], :, x_t]  # N, n, n, |x_{t-1}|
        right_term = self.Qbs[t - 1]  # N, |x|, |x_{t-1}|
        posterior_probs = (
            left_term[:, :, :, None, :] * right_term[:, None, None, :, :]
        )  # N, n, n |x|, |x_t_1|
        posterior_probs = posterior_probs / posterior_probs.sum(-1, keepdim=True)

        # sample from ancestral distribution q(x_{t-1} | x_t)
        x_probs = F.softmax(pred, dim=-1)  # N, n, n, |x|
        ancestral_probs = (posterior_probs * x_probs[..., None]).sum(
            -2
        )  # N, n, n |x_{t-1}|
        x_t_1 = sample_categorical(ancestral_probs)  # N, n, n

        # no samples for t=0
        nonzero_mask = (t > 0).long()[:, None, None]
        x_0 = x_probs.argmax(-1)
        return x_t_1 * nonzero_mask + x_0 * (1 - nonzero_mask)

    def get_loss(self, x, pred):
        return F.cross_entropy(
            pred.view(-1, pred.shape[-1]), x.view(-1), reduction="none"
        ).view(x.shape)


def sample_categorical(probs):
    """Sample from a categorical distribution.

    Shape:
        probs: :math:`(*, k)`
        out: :math:`(*)`
    """
    return th.multinomial(probs.view(-1, probs.shape[-1]), 1).view(probs.shape[:-1])
