import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np
from einops import rearrange

from bubbleformer.layers import AxialAttentionBlock, AttentionBlock, HMLPEmbed, HMLPDebed, FiLMMLP
from ._api import register_model

__all__ = ["AViT", "FiLMConditionedAViT"]


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
        input_fields: int = 3,
        output_fields: int = 3,
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
        self.patch_size = patch_size
        self.dp = np.linspace(0, drop_path, processor_blocks)
        # Hierarchical Patch Embedding
        self.embed = HMLPEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
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
            out_channels=output_fields
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C, H, W)
        """
        b, t, c, H, W = x.shape

        # --- Pad H/W to be divisible by patch size ---
        p = self.patch_size
        pad_h = (-H) % p  # amount to add on bottom
        pad_w = (-W) % p  # amount to add on right

        # Encode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        if pad_h or pad_w:
            # F.pad pads as (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
        x = self.embed(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)

        # Process
        for blk in self.blocks:
            # x = cp.checkpoint(blk, x, use_reentrant=False)
            x = blk(x)

        # Decode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.debed(x)

        # --- Crop back to original H/W ---
        Hp = H + pad_h
        Wp = W + pad_w
        if pad_h or pad_w:
            x = x[:, :, :Hp - pad_h if pad_h else Hp, :Wp - pad_w if pad_w else Wp]
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)

        return x  # Temporal bundling (B, T, C, H, W)


@register_model("filmavit")
class FiLMConditionedAViT(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) is an expressive and lightweight way to 
    condition neural networks using external information (like fluid parameters).
    This model uses FiLM conditioning on the embeddings and the blocks.
    Args:
        input_fields (int): Number of input fields
        output_fields (int): Number of output fields
        time_window (int): Number of time steps
        patch_size (int): Size of the square patch
        embed_dim (int): Dimension of the embedding
        num_heads (int): Number of attention heads
        processor_blocks (int): Number of processor blocks
        drop_path (float): Dropout rate
        attn_scale (bool): Whether to use attention scaling
        feat_scale (bool): Whether to use feature scaling
        num_fluid_params (int): Number of fluid parameters for conditioning
    """
    def __init__(
        self,
        input_fields: int = 3,
        output_fields: int = 3,
        time_window: int = 12,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        processor_blocks: int = 12,
        drop_path: int = 0.2,
        attn_scale: bool = True,
        feat_scale: bool = True,
        num_fluid_params: int = 8,
    ):
        super().__init__()
        self.embed = HMLPEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
            embed_dim=embed_dim,
        )

        self.film_embed = FiLMMLP(num_fluid_params, embed_dim)
        # self.film_blocks = nn.ModuleList([
        #     FiLMMLP(num_fluid_params, embed_dim) for _ in range(processor_blocks)
        # ])

        self.dp = np.linspace(0, drop_path, processor_blocks)
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                drop_path=self.dp[i],
                attn_scale=attn_scale,
                feat_scale=feat_scale,
            )
            for i in range(processor_blocks)
        ])

        self.debed = HMLPDebed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=output_fields
        )

    def forward(self, x: torch.Tensor, fluid_params: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        fluid_params: (B, num_fluid_params)
        """
        B, T, _, _, _ = x.shape

        # Encode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        # Apply FiLM conditioning on the embeddings
        x = self.film_embed(x, fluid_params)  # (B, T, C, H, W)

        # Process with FiLM-modulated blocks
        # for blk, film in zip(self.blocks, self.film_blocks):
        for blk in self.blocks:
            x = blk(x)
            # x = film(x, fluid_params)

        # Decode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.debed(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)
        return x
