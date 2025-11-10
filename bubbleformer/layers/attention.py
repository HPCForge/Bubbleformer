import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function
from einops import rearrange
import math
from timm.layers import DropPath
from bubbleformer.layers import GeluMLP, RelativePositionBias, ContinuousPositionBias1D
from bubbleformer.layers.axial_attention import AxialAttention

class AttentionBlock(nn.Module):
    """
    Attention Block with Optional Scaling for High-Frequency Components
    Takes in tensors of shape (B, n, emb, H, W)
    where n is the token sequence, emb is the embedding dimension
    H and W are the spatial tokens
    and applies self-attention across time dimension
    Args:
        embed_dim (int): Number of features in the input tensor
        num_heads (int): Number of attention heads
        drop_path (float): Drop path rate
        layer_scale_init_value (float): Initial value for layer scale
        bias_type (str): Type of bias to use in the attention mechanism
        attn_scale (bool): Whether to apply attention scaling
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        drop_path: float = 0,
        layer_scale_init_value: float = 1e-6,
        bias_type: str = "rel",
        attn_scale: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_scale = attn_scale
        self.head_dim = embed_dim // num_heads

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((embed_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0 else None
        )
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
        x = self.norm1(x)
        x = self.input_head(x)
        x = x.view(batch_size, t, h, w, self.num_heads, 3 * self.head_dim)
        x = x.permute(0, 2, 3, 4, 1, 5).contiguous().view(batch_size * h * w, self.num_heads, t, 3 * self.head_dim)
        q, k, v = x.tensor_split(3, dim=-1)

        # Sequence length is really small (like 5)... So, just compute the attention manually.
        x = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1) @ v
        x = rearrange(x, "(b h w) num_heads t head_dim -> b t h w (num_heads head_dim)", t=t, h=h, w=w).contiguous()
        
        #q = rearrange(q, "b t h w num_heads head_dim -> (b h w) num_heads t head_dim").contiguous()
        #k = rearrange(k, "b t h w num_heads head_dim -> (b h w) num_heads t head_dim").contiguous()
        #v = rearrange(v, "b t h w num_heads head_dim -> (b h w) num_heads t head_dim").contiguous()
        #x = F.scaled_dot_product_attention(q, k, v)
        #x = rearrange(x, "(b h w) num_heads t head_dim -> b t h w (num_heads head_dim)", t=t, h=h, w=w).contiguous()
        #x = x.view(batch_size, t, h, w, self.num_heads * self.head_dim)
        
        x = self.norm2(x)
        x = self.output_head(x)
        output = x + inp
        return output

class AxialAttentionBlock(nn.Module):
    """
    Axial Attention Block
    Args:
        embed_dim (int):Embedding dimension
        num_heads (int): Number of attention heads
        drop_path (float): Dropout rate
        layer_scale_init_value (float): Initial value for layer scale
        bias_type (str): Type of bias to use
    """
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        drop_path=0,
        layer_scale_init_value=1e-6,
        bias_type="rel",
        attn_scale=True,
        feat_scale=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_scale = attn_scale
        self.feat_scale = feat_scale
        self.head_dim = embed_dim // num_heads

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.gamma_att = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((embed_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.gamma_mlp = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((embed_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )

        self.input_head = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_head = nn.Linear(embed_dim, embed_dim)
        self.qnorm = nn.LayerNorm(self.head_dim)
        self.knorm = nn.LayerNorm(self.head_dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = GeluMLP(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, H, W, C)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, H, W, C)
        """
        batch_size, t, h, w, c = x.shape
        inp = x.clone()
        x = self.norm1(x)

        x = self.input_head(x)
        q, k, v = x.tensor_split(3, dim=-1)
        q = q.view(batch_size, t, h, w, self.num_heads, self.head_dim)
        k = k.view(batch_size, t, h, w, self.num_heads, self.head_dim)
        v = v.view(batch_size, t, h, w, self.num_heads, self.head_dim)
        #q, k = self.qnorm(q), self.knorm(k)
        
        #xx = self.height_attention(q, k, v)
        #xy = self.width_attention(q, k, v)
        #xx = xx.view(batch_size, t, h, w, self.num_heads * self.head_dim)
        #xy = xy.view(batch_size, t, h, w, self.num_heads * self.head_dim)
       
        # X direction attention
        qx, kx, vx = map(
            lambda x: rearrange(x, "b t h w num_heads head_dim ->  (b t h) num_heads w head_dim"), [q, k, v]
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
         
        # Combine
        x = (xx + xy) / 2
        x = self.norm2(x)
        x = self.output_head(x)
        x = x + inp

        # MLP
        inp = x.clone()
        x = self.mlp(x)
        x = self.mlp_norm(x)
        out = x + inp
        
        out = out.view(batch_size, t, h, w, self.num_heads * self.head_dim)

        return out
