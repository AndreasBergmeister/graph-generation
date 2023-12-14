import torch as th
from torch.nn import LayerNorm, Linear, Module, ModuleList


class MLP(Module):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_features (int): Number of features of the input.
        hidden_features (list[int]): List of the hidden features dimensions.
        out_features (int, optional): If not `None` a projection layer is added at the end of the MLP. Defaults to `None`.
        bias (bool, optional): Whether to use bias in the linear layers. Defaults to `True`.
        norm_layer (Module, optional): Normalization layer to use. Defaults to `norm_layer`.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: list[int],
        out_features: int | None = None,
        bias: bool = True,
        norm_layer=LayerNorm,
    ):
        super().__init__()
        lin_layers = []
        norm_layers = []
        hidden_in_features = in_features
        for hidden_dim in hidden_features:
            lin_layers.append(Linear(hidden_in_features, hidden_dim, bias=bias))
            norm_layers.append(norm_layer(hidden_dim))
            hidden_in_features = hidden_dim

        self.out_layer = (
            Linear(hidden_in_features, out_features, bias=bias)
            if out_features is not None
            else None
        )

        self.lin_layers = ModuleList(lin_layers)
        self.norm_layers = ModuleList(norm_layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        for lin, norm in zip(self.lin_layers, self.norm_layers):
            x = lin(x)
            x = norm(x)
            x = th.relu(x)

        if self.out_layer is not None:
            x = self.out_layer(x)

        return x
