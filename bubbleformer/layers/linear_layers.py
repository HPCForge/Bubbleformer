import torch
import torch.nn as nn

ACTIVATION = {
    "gelu":nn.GELU(),
    "tanh":nn.Tanh(),
    "sigmoid":nn.Sigmoid(),
    "relu":nn.ReLU(),
    "leaky_relu":nn.LeakyReLU(0.1),
    "softplus":nn.Softplus(),
    "ELU":nn.ELU(),
    "silu":nn.SiLU()
}

class MLP(nn.Module):
    """
    Multi-layer perceptron with one hidden layer and customizable activation function.
    Args:
        in_features (int): Number of input features
        hidden_features (int, optional): Number of hidden features
        out_features (int, optional): Number of output features
        act (str): Activation function to use
        drop (float): Dropout rate
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act: str = 'gelu',
        drop: float | None = None
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = ACTIVATION[act]
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop is not None else nn.Identity()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class GeluMLP(nn.Module):
    """
    Multi-layer perceptron with a hidden layer and GELU activation
    specifically designed for use in transformer architectures.
    Args:
        hidden_dim (int): Dimension of the hidden layer
        exp_factor (float): Expansion factor
    """
    def __init__(
        self,
        hidden_dim: int,
        exp_factor: float = 4.0
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim * exp_factor))
        self.fc2 = nn.Linear(int(hidden_dim * exp_factor), hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        return self.fc2(self.act(self.fc1(x)))


class SirenMLP(nn.Module):
    """
    MLP with sine activation as implemented in SIREN paper
    Args:
        hidden_dim (int): Dimension of the hidden layer
        w0 (float): Frequency parameter
    """
    def __init__(
        self,
        hidden_dim: int,
        w0: float = 1.0
    ):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        return torch.sin(self.w0 * self.fc(x))

class FiLMMLP(nn.Module):
    """
    MLP with FiLM (Feature-wise Linear Modulation) layers
    Args:
        param_dim (int): Dimensions of conditioning parameters
        embed_dim (int): Embedding dimension
    """
    def __init__(
        self,
        param_dim: int,
        embed_dim: int
    ):
        super().__init__()
        self.film_net = nn.Sequential(
            nn.LayerNorm(param_dim),
            nn.Linear(param_dim, embed_dim * 2),
        )

    def forward(self, x: torch.Tensor, cond) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor
            cond (torch.Tensor): Conditioning tensor
        Returns:
            torch.Tensor: Output tensor
        """
        gamma_beta = self.film_net(cond)  # (B, 2 * C)
        gamma, beta = gamma_beta.chunk(2, dim=1)  # each (B, C)

        gamma = gamma.view(-1, 1, x.shape[2], 1, 1)  # (B, 1, C, 1, 1)
        beta = beta.view(-1, 1, x.shape[2], 1, 1)

        return gamma * x + beta
