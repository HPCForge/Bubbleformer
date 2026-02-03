import torch
import torch.nn as nn

from bubbleformer.layers.attention import TemporalAttention, SpatialNeighborhoodAttention
from bubbleformer.layers.memory_layers import MemoryBlock

class SpaceTimeAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__()
        
        self.temporal = TemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.spatial = SpatialNeighborhoodAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.temporal(x)
        x = self.spatial(x)
        return x

class SpaceSSMAttention(nn.Module):
    """
    Combined spatial and temporal attention using S4-based memory.

    Temporal processing: MemoryBlock (S4/S4D) - maintains state across chunks
    Spatial processing: SpatialNeighborhoodAttention - only sees current window
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        d_state: int = 64,
        ssm_mode: str = 'diag',
        dropout: float = 0.0,
    ):
        """
        Args:
            embed_dim: Model dimension
            num_heads: Number of attention heads for spatial attention
            d_state: SSM state dimension for temporal memory
            ssm_mode: 'diag' for S4D, 'dplr' for full S4
            dropout: Dropout probability
        """
        super().__init__()

        self.temporal = MemoryBlock(
            d_model=embed_dim,
            d_state=d_state,
            mode=ssm_mode,
            dropout=dropout,
        )

        self.spatial = SpatialNeighborhoodAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor, state=None):
        """
        Args:
            x: (B, T, H, W, C) input tensor
            state: SSM state from previous chunk (or None)

        Returns:
            y: (B, T, H, W, C) output tensor
            next_state: SSM state for next chunk
        """
        # Temporal: S4 memory block with state
        x, next_state = self.temporal(x, state)

        # Spatial: neighborhood attention (no state needed, only sees current T frames)
        x = self.spatial(x)

        return x, next_state

    def default_state(self, batch_size: int, H: int, W: int, device=None):
        """Initialize SSM state."""
        return self.temporal.default_state(batch_size, H, W, device)

    def setup_step(self):
        """Prepare for recurrent stepping mode."""
        self.temporal.setup_step()

    def step(self, x: torch.Tensor, state):
        """
        Single frame step for recurrent inference.

        Args:
            x: (B, H, W, C) single frame
            state: SSM state

        Returns:
            y: (B, H, W, C) output frame
            next_state: updated SSM state
        """
        # Temporal step
        x, next_state = self.temporal.step(x, state)

        # Spatial attention on single frame (add dummy time dim)
        x = x.unsqueeze(1)  # (B, 1, H, W, C)
        x = self.spatial(x)
        x = x.squeeze(1)  # (B, H, W, C)

        return x, next_state
