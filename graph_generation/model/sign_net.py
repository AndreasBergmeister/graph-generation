import torch as th
from torch.nn import Dropout, Linear, Module, ModuleList
from torch_geometric.nn import GINConv

from .mlp import MLP


class SignNet(Module):
    def __init__(
        self,
        num_eigenvectors: int,
        hidden_features: int,
        out_features: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_layer = Linear(2, hidden_features)
        self.conv_layers = ModuleList(
            [
                GINConv(
                    MLP(hidden_features, [hidden_features, hidden_features]),
                    train_eps=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.skip_layer = Linear(hidden_features * (num_layers + 1), hidden_features)
        self.dropout = Dropout(dropout)
        # the following corresponds to the ρ function in the paper
        self.merge_layer = MLP(
            in_features=num_eigenvectors * hidden_features,
            hidden_features=[hidden_features, hidden_features],
            out_features=out_features,
        )

    def forward(self, spectral_features, edge_index):
        """Forward pass of the model.

        Args:
            spectral_features (Tensor): Eigenvalues (repeated) concatenated with eigenvectors. Shape: :math:`(V, num_eigenvectors * 2)`.
            edge_index (Adj): Adjacency matrix given as edge index or sparse tensor. Shape: :math:`(2, E)` or :math:`(V, V)`.

        Returns:
            Tensor: Node features. Shape: :math:`(V, out_features)`.
        """
        # Stack spectral features
        eigenvalues_repeated, eigenvectors = spectral_features.chunk(
            2, dim=-1
        )  # (V, k), (V, k)

        positive_spectral_features = th.stack(
            [eigenvalues_repeated, eigenvectors], dim=-1
        )  # V, k, 2
        negative_spectral_features = th.stack(
            [eigenvalues_repeated, -eigenvectors], dim=-1
        )  # V, k, 2
        combined_spectral_features = th.stack(
            [positive_spectral_features, negative_spectral_features]
        ).transpose(
            1, 2
        )  # 2, k, V, 2

        # Apply layers
        x = self.in_layer(combined_spectral_features)  # 2, k, V, hidden_features
        xs = [x]
        for conv in self.conv_layers:
            # apply conv layer to each spectral feature independently
            x = conv(x=x, edge_index=edge_index)
            xs.append(x)

        # Skip connection
        x = th.cat(xs, dim=-1)
        x = self.dropout(x)
        x = self.skip_layer(x)  # 2, k, V, hidden_features
        # Make sign invariant
        x = x.sum(dim=0)  # k, V, hidden_features

        # Merge features
        x = x.transpose(0, 1)  # V, k, hidden_features
        x = self.merge_layer(x.reshape(x.size(0), -1))  # V, out_features

        return x
