import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function
from einops import rearrange
import math
from timm.layers import DropPath
import einops
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

import natten

class SpatialNeighborhoodAttention(nn.Module):
    r"""
    This is similar to natten's NaighborhoodAttention2D,
    but includes additional query and key normalization.
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.input_head = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_head = nn.Linear(embed_dim, embed_dim)
        self.qnorm = nn.LayerNorm(self.head_dim)
        self.knorm = nn.LayerNorm(self.head_dim)
        
        # TODO: should each attention block use the same rotary embedding?
        self.rotary_emb = RotaryEmbedding(
            # NOTE: This much be smaller than the head dim. 
            dim=self.head_dim // 3,
            freqs_for="pixel",
            max_freq=256
        )

    def forward(self, x):
        b, t, h, w, c = x.shape
        
        # rotary embedding expects [batch, heads, seq, dim] layout
        heads = einops.rearrange(self.input_head(x), 
                                 "b t h w (heads head_dim) -> (b t) heads h w head_dim", 
                                 heads=self.num_heads).contiguous()
        q, k, v = heads.tensor_split(3, dim=-1)
        q = self.qnorm(q).to(x.dtype)
        k = self.knorm(k).to(x.dtype)
        freqs = self.rotary_emb.get_axial_freqs(h, w)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        
        # natten expects [batch, seq1, seq2, heads, dim] layout
        # TODO: we can just not do the rearrange above.
        q, k, v = map(
            lambda qkv: rearrange(qkv, "bt heads h w head_dim -> bt h w heads head_dim").contiguous(), [q, k, v]
        )

        output = natten.na2d(
            q,
            k,
            v,
            kernel_size=3,
            stride=1,
            dilation=1,   
        )
        
        output = einops.rearrange(output,
                                  "(b t) h w heads head_dim -> b t h w (heads head_dim)", 
                                  b=b, t=t).contiguous()
        output = self.output_head(output)
        return output