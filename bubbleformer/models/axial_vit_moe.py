import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from bubbleformer.layers import AxialAttentionMoEBlock, TemporalAttentionBlock, HMLPEmbed, HMLPDebed
from ._api import register_model

__all__ = ["AViTMoE"]


class SpaceTimeBlock(nn.Module):
    """
    Factored spacetime block with temporal attention followed by axial attention
    Args:
        embed_dim (int): Number of features in the input tensor
        num_heads (int): Number of attention heads
        drop_path (float): Drop path rate
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        drop_path: float = 0.0,
        n_experts: int = 6,
        n_shared_experts: int = 1,
        top_k: int = 2,
    ):
        super().__init__()

        self.spatial = AxialAttentionMoEBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_path=drop_path,
            n_experts=n_experts,
            n_shared_experts=n_shared_experts,
            top_k=top_k,
        )
        self.temporal = TemporalAttentionBlock(embed_dim, num_heads, drop_path=drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: 
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C, H, W)
        """
        _, t, _, _, _ = x.shape

        # First do temporal attention
        x = self.temporal(x)    # (B, T, C, H, W)

        # Now do spatial attention
        x = rearrange(x, "b t c h w -> (b t) c h w")        # BT sequences
        x = self.spatial(x)                                 # A spatial encoder block
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)

        return x    # (B, T, C, H, W)


@register_model("avit_moe")
class AViTMoE(nn.Module):
    """
    Model that interweaves spatial and temporal attention blocks. Temporal attention
    acts only on the time dimension.

    Args:
        fields (int): Number of fields
        patch_size (int): Size of the square patch
        embed_dim (int): Dimension of the embedding
        num_heads (int): Number of attention heads
        processor_blocks (int): Number of processor blocks
        drop_path (float): Dropout rate
        n_experts (int): Number of experts
        n_shared_experts (int): Number of shared experts
        top_k (int): Number of activated experts to use
    """
    def __init__(
        self,
        fields: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        processor_blocks: int = 12,
        drop_path: int = 0.2,
        n_experts: int = 6,
        n_shared_experts: int = 1,
        top_k: int = 2
    ):
        super().__init__()
        self.drop_path = drop_path
        self.dp = np.linspace(0, drop_path, processor_blocks)
        # Hierarchical Patch Embedding
        self.embed = HMLPEmbed(
            patch_size=patch_size,
            in_channels=fields,
            embed_dim=embed_dim,
        )
        # Factored spacetime block with (space/time axial attention)
        self.blocks = nn.ModuleList(
            [
                SpaceTimeBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    drop_path=self.dp[i],
                    n_experts=n_experts,
                    n_shared_experts=n_shared_experts,
                    top_k=top_k,
                )
                for i in range(processor_blocks)
            ]
        )
        # Patch Debedding
        self.debed = HMLPDebed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=fields
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C, H, W)
        """
        _, t, _, _, _ = x.shape

        # Encode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)

        # Process
        for blk in self.blocks:
            x = blk(x)

        # Decode - It would probably be better to grab the last time here since we're only
        # predicting the last step, but leaving it like this for compatibility to causal masking
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.debed(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)

        return x  # Temporal bundling (B, T, C, H, W)
