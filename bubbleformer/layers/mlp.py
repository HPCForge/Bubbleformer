import torch
import torch.nn as nn

@torch.compile(fullgraph=True)
class GeluMLP(nn.Module):
    """
    Multi-layer perceptron with a hidden layer and GELU activation
    Args:
        hidden_dim (int): Dimension of the hidden layer
        exp_factor (float): Expansion factor
    """
    def __init__(self, hidden_dim, exp_factor=4.0):
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

@torch.compile(fullgraph=True)
class FiLMMLP(nn.Module):
    """
    MLP with FiLM (Feature-wise Linear Modulation) layers
    Args:
        param_dim (int): Dimensions of conditioning parameters
        embed_dim (int): Embedding dimension
    """
    def __init__(self, param_dim, embed_dim):
        super().__init__()
        self.film_net = nn.Sequential(
            nn.LayerNorm(param_dim),
            nn.Linear(param_dim, embed_dim * 2),
        )

    def forward(self, x: torch.Tensor, cond) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor  (B, T, H, W, C)
            cond (torch.Tensor): Conditioning tensor (B, param_dim)
        Returns:
            torch.Tensor: Output tensor
        """
        assert x.shape[0] == cond.shape[0], "Batch size of input and condition must match"
        batch_size, num_channels = x.shape[0], x.shape[-1]
        
        gamma_beta = self.film_net(cond)  # (B, 2 * C)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        # Every (T, H, W) point gets the same embedding.
        gamma = gamma.view(batch_size, 1, 1, 1, num_channels)
        beta = beta.view(batch_size, 1, 1, 1, num_channels)

        return gamma * x + beta