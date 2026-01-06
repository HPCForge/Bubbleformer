import torch
import torch.nn as nn

from bubbleformer.layers.attention import TemporalAttention, SpatialNeighborhoodAttention

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
        x = self.temporal(x)
        x = self.spatial(x)
        return x