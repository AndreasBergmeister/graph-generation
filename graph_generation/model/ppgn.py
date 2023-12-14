import torch as th
from torch.nn import Dropout, Linear, Module, ModuleList

from .mlp import MLP


class PPGN(Module):
    """Implementation of Provable Powerful Graph Neural Network.

    Operates on a dense graph representation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        emb_features: int,
        hidden_features: int,
        ppgn_features: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Embedding layers
        self.in_emb_layer = Linear(in_features, emb_features)
        self.noise_cond_emb_layer = Linear(1, emb_features)

        # In layer
        self.in_layer = MLP(2 * emb_features, [hidden_features, hidden_features])

        # PPGN layers
        self.ppgn_layers = ModuleList(
            [PPGNLayer(hidden_features, ppgn_features) for _ in range(num_layers)]
        )

        # Out layer
        self.out_layer = Linear((num_layers + 1) * hidden_features, out_features)

        # Dropout
        self.dropout = Dropout(dropout)

    def forward(self, x, mask, noise_cond):
        """
        Shape:
            x: :math:`(N, n, n, d)`
            mask: :math:`(N, n, n, 1)`
            noise_cond: :math: `(N, 1)`
        """
        assert th.allclose(x, x.transpose(1, 2)), "x assumed to be symmetric"
        assert th.allclose(mask, mask.transpose(1, 2)), "mask assumed to be symmetric"

        # Input
        x = self.in_emb_layer(x)  # N, n, n, h
        noise_cond_emb = self.noise_cond_emb_layer(noise_cond[..., None])  # N, h

        x = th.cat([x, noise_cond_emb[:, None, None, :].expand_as(x)], dim=-1)
        x = self.dropout(x)
        x = self.in_layer(x)

        # PPGN layers
        skip = [x]
        for layer in self.ppgn_layers:
            x = layer(x, mask)
            skip.append(x)

        # Output
        x = th.cat(skip, dim=-1)
        x = self.dropout(x)
        x = self.out_layer(x)
        x = (x + x.transpose(1, 2)) / 2  # make symmetric

        return x * mask


class PPGNLayer(Module):
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

    def forward(self, x: th.Tensor, mask: th.Tensor) -> th.Tensor:
        """
        Shape:
            x: :math:`(N, n, n, h)`
            mask: :math:`(N, n, n, 1)`
        """
        m1 = (self.mlp1(x) * mask).permute(0, 3, 1, 2)  # N, p, n, n
        m2 = (self.mlp2(x) * mask).permute(0, 3, 1, 2)  # N, p, n, n

        m = m1 @ m2
        size = mask[:, :, 0, 0].sum(-1)  # N, 1
        m = m / size.sqrt().view(-1, 1, 1, 1)

        x = th.cat((x, m.permute(0, 2, 3, 1)), dim=-1)  # N, n, n, h + p
        x = self.mlp3(x) * mask  # N, n, n, h

        return x
