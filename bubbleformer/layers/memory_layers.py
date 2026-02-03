import torch
import torch.nn as nn
from einops import rearrange

from .s4 import FFTConv


class MemoryBlock(nn.Module):
    """
    SSM-based temporal memory block using S4/S4D.

    Supports two modes:
    - Convolution mode (forward): Efficient parallel processing, used during training
    - Recurrent mode (step): Sequential stepping, used during inference

    State can be passed between forward calls to maintain memory across chunks.
    """

    def __init__(self, d_model, d_state=64, mode='diag', dropout=0.0):
        """
        Args:
            d_model: Model dimension (number of channels)
            d_state: SSM state dimension
            mode: 'diag' or 's4d' for S4D (diagonal), 'dplr' or 's4' for full S4
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Pre-normalization
        self.norm = nn.LayerNorm(d_model)

        self.ssm = FFTConv(
            d_model=d_model,
            d_state=d_state,
            l_max=None,  # Variable length sequences
            channels=1,
            mode=mode,
            transposed=True,  # Input format (B, C, L)
            activation='gelu',
            dropout=dropout,
        )

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self._step_mode_setup = False

    def forward(self, x, state=None):
        """
        Process sequence with optional initial state (convolution mode).

        Args:
            x: (B, T, H, W, C) input sequence
            state: (B*H*W, C, N) optional initial SSM state from previous chunk

        Returns:
            y: (B, T, H, W, C) output sequence
            next_state: (B*H*W, C, N) state after processing (for next chunk)
        """
        B, T, H, W, C = x.shape

        # Store residual for skip connection
        residual = x

        # Pre-normalization
        x = self.norm(x)

        # Reshape: (B, T, H, W, C) -> (B*H*W, C, T)
        x_flat = rearrange(x, "b t h w c -> (b h w) c t")

        # Initialize state to zeros if not provided
        # This ensures we always compute and return next_state
        if state is None:
            state = self.ssm.default_state(B * H * W, device=x.device)

        # FFTConv with state computes output with state contribution
        # and returns next_state after processing the sequence
        y, next_state = self.ssm(x_flat, state=state)

        # Reshape back: (B*H*W, C, T) -> (B, T, H, W, C)
        y = rearrange(y, "(b h w) c t -> b t h w c", b=B, h=H, w=W)

        # Output projection + residual connection
        y = self.out_proj(y) + residual

        return y, next_state

    def setup_step(self):
        """Prepare for recurrent stepping mode (call once before using step())."""
        self.ssm.setup_step()
        self._step_mode_setup = True

    def default_state(self, batch_size, H, W, device=None):
        """
        Initialize zero state for recurrent mode.

        Args:
            batch_size: Batch size B
            H: Spatial height (in patches)
            W: Spatial width (in patches)
            device: Device for state tensor

        Returns:
            state: (B*H*W, C, N) initial state
        """
        return self.ssm.default_state(batch_size * H * W, device=device)

    def step(self, x, state):
        """
        Single frame recurrent step.

        Args:
            x: (B, H, W, C) single frame (no time dimension)
            state: (B*H*W, C, N) current SSM state

        Returns:
            y: (B, H, W, C) output frame
            next_state: (B*H*W, C, N) updated state
        """
        if not self._step_mode_setup:
            self.setup_step()

        B, H, W, C = x.shape

        # Store residual for skip connection
        residual = x

        # Pre-normalization
        x = self.norm(x)

        # Reshape: (B, H, W, C) -> (B*H*W, C)
        x = rearrange(x, "b h w c -> (b h w) c")

        y, next_state = self.ssm.step(x, state)

        # Reshape back: (B*H*W, C) -> (B, H, W, C)
        y = rearrange(y, "(b h w) c -> b h w c", b=B, h=H, w=W)

        # Output projection + residual connection
        y = self.out_proj(y) + residual

        return y, next_state


class MLP(nn.Module):
    """Simple MLP for use in other memory components."""

    def __init__(self, d_model, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        hidden_dim = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class TimeQueryPool(nn.Module):
    """
    Query-based temporal downsampling using cross-attention.
    (B, T, H, W, D) -> (B, K, H, W, D) where K = time_window
    """

    def __init__(
        self,
        d_model: int,
        time_window: int = 5,
        num_heads: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.time_window = time_window
        self.num_heads = num_heads
        self.dropout = dropout

        self.queries = nn.Parameter(torch.randn(time_window, d_model) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.mlp = MLP(d_model)

        self.norm_x = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """(B*H*W, T, D) -> (B*H*W, K, D)"""

        N, T, D = x.shape
        x = self.norm_x(x)

        q = self.queries.unsqueeze(0).expand(N, -1, -1)
        q = self.norm_q(q)

        y, _ = self.cross_attn(q, x, x)
        y = y + self.mlp(self.norm_mlp(y))
        return y
