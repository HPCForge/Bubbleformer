import torch
import torch.nn as nn
from bubbleformer.layers.attention import TemporalAttention, SpatialAxialAttention, SpatialNeighborhoodAttention
from bubbleformer.layers.mlp import GeluMLP
from torch.profiler import record_function

class SpaceTimeBlock(nn.Module):
    """
    Factored spacetime block with temporal attention followed by spatial neighborhood attention
    Args:
        embed_dim (int): Number of features in the input tensor
        num_heads (int): Number of attention heads
        drop_path (float): Drop path rate
        attn_scale (bool): Whether to use attention scaling
        feat_scale (bool): Whether to use feature scaling
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        drop_path: float = 0.0,
        attn_scale: bool = True,
        feat_scale: bool = True,
    ):
        super().__init__()

        self.pre_norm = nn.LayerNorm(embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)

        self.temporal = TemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.spatial = SpatialNeighborhoodAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.mlp = GeluMLP(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: 
            x (torch.Tensor): Input tensor of shape (B, T, H, W, C)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, H, W, C)
        """
        with record_function("space_time_block"):
            
            # Attention with a skip connection
            inp = x.clone()
            # Force pytorch to use an actual fast implementaiton.
            # This has more requirements on head-dim and requires <=16 bit precision.
            with torch.nn.attention.sdpa_kernel(backends=torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                with record_function("temporal"):
                    x = self.temporal(x)
                with record_function("spatial"):
                    x = self.spatial(x)
            with record_function("pre_norm"):          
                x = self.pre_norm(x) + inp

            # MLP with a skip connection
            intermediate = x.clone()
            with record_function("mlp"):
                x = self.mlp(x)
            with record_function("post_norm"):
                x = self.post_norm(x) + intermediate

        return x