import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function
from einops import rearrange
import math
from rotary_embedding_torch import RotaryEmbedding

class TemporalAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        """
        Args:
            embed_dim (int): Number of features in the input tensor
            num_heads (int): Number of attention heads
            drop_path (float): Drop path rate
            bias_type (str): Type of bias to use in the attention mechanism
            attn_scale (bool): Whether to apply attention scaling
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.input_head = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_head = nn.Linear(embed_dim, embed_dim)
        self.qnorm = nn.LayerNorm(self.head_dim)
        self.knorm = nn.LayerNorm(self.head_dim)
        
        self.rotary_emb = RotaryEmbedding(dim=32)

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
        x = x.view(batch_size, t, h, w, self.num_heads, 3 * self.head_dim)
        x = rearrange(x, "b t h w heads head_dim -> (b h w) heads t head_dim").contiguous()
        q, k, v = x.tensor_split(3, dim=-1)
        q = self.qnorm(q)
        k = self.knorm(k)
        
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # Sequence length is really small (like 5)... So, just compute the attention manually.
        # This could also be done with a flash attention with a causal mask.
        x = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1) @ v
        x = rearrange(x, "(b h w) num_heads t head_dim -> b t h w (num_heads head_dim)", 
                      t=t, h=h, w=w).contiguous()
        
        x = self.output_head(x)
        return x