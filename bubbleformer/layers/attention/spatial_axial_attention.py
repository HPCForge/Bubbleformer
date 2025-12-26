import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function
from einops import rearrange
import math
from timm.layers import DropPath

class SpatialAxialAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        """
        Args:
            embed_dim (int):Embedding dimension
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.input_head = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_head = nn.Linear(embed_dim, embed_dim)
        self.qnorm = nn.LayerNorm(self.head_dim)
        self.knorm = nn.LayerNorm(self.head_dim)
                
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, H, W, C)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, H, W, C)
        """
        batch_size, t, h, w, c = x.shape
        inp = x.clone()

        x = self.input_head(x)
        q, k, v = x.tensor_split(3, dim=-1)
        q = q.view(batch_size, t, h, w, self.num_heads, self.head_dim)
        k = k.view(batch_size, t, h, w, self.num_heads, self.head_dim)
        v = v.view(batch_size, t, h, w, self.num_heads, self.head_dim)
        q = self.qnorm(q)
        k = self.knorm(k)
       
        # X direction attention
        qx, kx, vx = map(
            lambda x: rearrange(x, "b t h w num_heads head_dim -> (b t h) num_heads w head_dim"), [q, k, v]
        )
        xx = F.scaled_dot_product_attention(
            query=qx.contiguous(),
            key=kx.contiguous(),
            value=vx.contiguous(),
        )
        xx = rearrange(xx, "(b t h) num_heads w head_dim -> b t h w num_heads head_dim", t=t, h=h).contiguous()
        xx = xx.view(batch_size, t, h, w, self.num_heads * self.head_dim)

        # Y direction attention
        qy, ky, vy = map(
            lambda x: rearrange(x, "b t h w num_heads head_dim ->  (b t w) num_heads h head_dim"), [q, k, v]
        )
        xy = F.scaled_dot_product_attention(
            query=qy.contiguous(),
            key=ky.contiguous(),
            value=vy.contiguous(),
        )
        xy = rearrange(xy, "(b t w) num_heads h head_dim -> b t h w num_heads head_dim", t=t, w=w).contiguous()
        xy = xy.view(batch_size, t, h, w, self.num_heads * self.head_dim)
        
        x = (xx + xy) / 2
        x = self.output_head(x)
        return x
    