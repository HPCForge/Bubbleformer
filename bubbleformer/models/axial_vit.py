import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import numpy as np
from einops import rearrange
from torch.profiler import record_function

from bubbleformer.layers import AxialAttentionBlock, TemporalAttentionBlock, HMLPEmbed, HMLPDebed, FiLMMLP, GeluMLP
from bubbleformer.layers.positional_encoding import CoordinatePosEncoding
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

        self.pre_norm = nn.LayerNorm(embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)

        self.temporal = TemporalAttentionBlock(
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
                x = self.pre_norm(x + inp)

            # MLP with a skip connection
            intermediate = x.clone()
            with record_function("mlp"):
                x = self.mlp(x)
            with record_function("post_norm"):
                x = self.post_norm(x + intermediate)

        return x


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
        _, t, _, _, _ = x.shape

        # Encode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)

        # Process
        for blk in self.blocks:
            # x = cp.checkpoint(blk, x, use_reentrant=False)
            x = blk(x)

        # Decode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.debed(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)

        return x  # Temporal bundling (B, T, C, H, W)


@register_model("filmavit")
@torch.compile(fullgraph=True)
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
        
        self.coord_enc = CoordinatePosEncoding(embed_dim)

    def forward(self, x: torch.Tensor, fluid_params: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        fluid_params: (B, num_fluid_params)
        """
        B, T, _, _, _ = x.shape

        # Encode
        with record_function("encode"):
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.embed(x)
            x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        # Permute to better order for attention (B, T, H, W, C)
        # TODO: IDK if input should be in this format for the embedding or not...
        # I think conv's do support NHWC layout.
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        
        # TODO: This encoding is temporary and assumes the domain size is always the same.
        with record_function("coord_enc"):
            x = self.coord_enc(x)

        # Apply FiLM conditioning on the embeddings
        with record_function("film_embed"):
            x = self.film_embed(x, fluid_params)  # (B, T, H, W, C)

        # Attention blocks
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
               x = blk(x)

        # Permute back to (B, T, C, H, W)
        x = x.permute(0, 1, 4, 2, 3).contiguous()

        # Decode
        with record_function("decode"):
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.debed(x)
            x = rearrange(x, "(b t) c h w -> b t c h w", t=T)
        return x
