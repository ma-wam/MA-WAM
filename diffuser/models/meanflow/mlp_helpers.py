"""
Helper MLP classes for MeanFlow implementation
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Basic MLP implementation."""

    def __init__(
        self,
        dims,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
    ):
        super().__init__()
        layers = []

        # Get activation function
        if activation_type == "Mish":
            activation = nn.Mish()
        elif activation_type == "ReLU":
            activation = nn.ReLU()
        elif activation_type == "LeakyReLU":
            activation = nn.LeakyReLU()
        else:
            activation = nn.Mish()  # default

        # Get output activation
        if out_activation_type == "Identity":
            out_activation = nn.Identity()
        elif out_activation_type == "Tanh":
            out_activation = nn.Tanh()
        else:
            out_activation = nn.Identity()

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if use_layernorm:
                layers.append(nn.LayerNorm(dims[i + 1]))

            # Add activation for all layers except the last
            if i < len(dims) - 2:
                layers.append(activation)
            else:
                layers.append(out_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualMLP(nn.Module):
    """Residual MLP implementation."""

    def __init__(
        self,
        dims,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
    ):
        super().__init__()

        # Get activation function
        if activation_type == "Mish":
            activation = nn.Mish()
        elif activation_type == "ReLU":
            activation = nn.ReLU()
        elif activation_type == "LeakyReLU":
            activation = nn.LeakyReLU()
        else:
            activation = nn.Mish()  # default

        self.input_dim = dims[0]
        self.output_dim = dims[-1]

        # Input projection if needed
        if self.input_dim != dims[1]:
            self.input_proj = nn.Linear(dims[0], dims[1])
        else:
            self.input_proj = nn.Identity()

        # Residual blocks
        residual_layers = []
        for i in range(1, len(dims) - 1):
            block = []
            if use_layernorm:
                block.append(nn.LayerNorm(dims[i]))
            block.append(nn.Linear(dims[i], dims[i]))
            block.append(activation)
            residual_layers.append(nn.Sequential(*block))

        self.residual_blocks = nn.ModuleList(residual_layers)

        # Output projection
        if dims[-2] != dims[-1]:
            self.output_proj = nn.Linear(dims[-2], dims[-1])
        else:
            self.output_proj = nn.Identity()

        # Output activation
        if out_activation_type == "Identity":
            self.out_activation = nn.Identity()
        elif out_activation_type == "Tanh":
            self.out_activation = nn.Tanh()
        else:
            self.out_activation = nn.Identity()

    def forward(self, x):
        x = self.input_proj(x)

        for block in self.residual_blocks:
            x = x + block(x)

        x = self.output_proj(x)
        x = self.out_activation(x)
        return x