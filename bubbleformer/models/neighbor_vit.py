import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import numpy as np
from einops import rearrange
from torch.profiler import record_function

from bubbleformer.layers import (
    HMLPEmbed, 
    HMLPDebed, 
    FiLMMLP,
    SpaceTimeBlock
)
from bubbleformer.layers.positional_encoding import CoordinatePosEncoding
from ._api import register_model

@register_model("neighbor_vit")
class NeighborViT(nn.Module):
    """
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
        
        input = x.clone()

        # Encode
        with record_function("encode"):
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.embed(x)
            x = rearrange(x, "(b t) c h w -> b t c h w", t=T)
            
        embed = x.clone()

        # Permute to better order for attention (B, T, H, W, C)
        # TODO: IDK if input should be in this format for the embedding or not...
        # I think conv's do support NHWC layout.
        x = rearrange(x, "b t c h w -> b t h w c").contiguous()

        # Apply FiLM conditioning on the embeddings
        with record_function("film_embed"):
            x = self.film_embed(x, fluid_params)

        # Attention blocks
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
               x = blk(x)

        x = rearrange(x, "b t h w c -> b t c h w").contiguous()
        
        # Skip connection from patch embeddings
        x = x + embed
       
        # Decode
        with record_function("decode"):
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.debed(x)
            x = rearrange(x, "(b t) c h w -> b t c h w", t=T)
        
        # Skip connection from the original input
        x = x + input
        
        return x