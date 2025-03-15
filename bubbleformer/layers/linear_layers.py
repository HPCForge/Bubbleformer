import torch
import torch.nn as nn


class GeluMLP(nn.Module):
    """
    Multi-layer perceptron with a hidden layer and GELU activation
    Args:
        hidden_dim (int): Dimension of the hidden layer
        inter_dim (int, optional): Intermediate dimension. If None, uses hidden_dim * exp_factor
        exp_factor (float): Expansion factor used when inter_dim is None
    """
    def __init__(self, hidden_dim, inter_dim=None, exp_factor=4.0):
        super().__init__()
        if inter_dim is None:
            inter_dim = int(hidden_dim * exp_factor)
            
        self.fc1 = nn.Linear(hidden_dim, inter_dim)
        self.fc2 = nn.Linear(inter_dim, hidden_dim)
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
        inter_dim (int, optional): Intermediate dimension. If None, uses hidden_dim * exp_factor
        exp_factor (float): Expansion factor used when inter_dim is None
        w0 (float): Frequency parameter
    """
    def __init__(self, hidden_dim, inter_dim=None, exp_factor=4.0, w0=1.0):
        super().__init__()
        if inter_dim is None:
            inter_dim = int(hidden_dim * exp_factor)
            
        self.fc1 = nn.Linear(hidden_dim, inter_dim)
        self.fc2 = nn.Linear(inter_dim, hidden_dim)
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        return self.fc2(torch.sin(self.w0 * self.fc1(x)))
