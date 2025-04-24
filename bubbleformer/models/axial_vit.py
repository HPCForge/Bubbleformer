import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from bubbleformer.layers import AxialAttentionBlock, AttentionBlock, HMLPEmbed, HMLPDebed
from ._api import register_model

__all__ = ["AViT"]


class SpaceTimeBlock(nn.Module):
    """
    Factored spacetime block with temporal attention followed by axial attention
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

        self.temporal = AttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_path=drop_path,
            attn_scale=attn_scale,
        )

        self.spatial = AxialAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_path=drop_path,
            attn_scale=attn_scale,
            feat_scale=feat_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: 
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C, H, W)
        """
        _, t, _, _, _ = x.shape

        # First do temporal attention
        x = self.temporal(x)    # (B, T, emb, H, W)

        # Now do spatial attention
        x = rearrange(x, "b t emb h w -> (b t) emb h w")        # BT sequences
        x = self.spatial(x)                                 # A spatial encoder block
        x = rearrange(x, "(b t) emb h w -> b t emb h w", t=t)

        return x    # (B, T, emb, H, W)


@register_model("avit")
class AViT(nn.Module):
    """
    Model that interweaves spatial and temporal attention blocks. Temporal attention
    acts only on the time dimension.

    Args:
        fields (int): Number of fields
        time_window (int): Number of time steps
        patch_size (int): Size of the square patch
        embed_dim (int): Dimension of the embedding
        num_heads (int): Number of attention heads
        processor_blocks (int): Number of processor blocks
        drop_path (float): Dropout rate
        attn_scale (bool): Whether to use attention scaling
        feat_scale (bool): Whether to use feature scaling
    """
    def __init__(
        self,
        fields: int = 3,
        time_window: int = 12,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        processor_blocks: int = 12,
        drop_path: int = 0.2,
        attn_scale: bool = True,
        feat_scale: bool = True,
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
                    attn_scale=attn_scale,
                    feat_scale=feat_scale,
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

        # Decode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.debed(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)

        return x  # Temporal bundling (B, T, C, H, W)
