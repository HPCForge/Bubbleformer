import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from bubbleformer.layers.s4d import S4D
from einops import rearrange

class MLP(nn.Module):

    def __init__(self, d_model, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        
        hidden_dim = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class TimeQueryPool(nn.Module):

    """
    Query downsamples from length to another
    (B, T, H, W, D) -> (B, K, H, W, D)
    K = time_window
    """

    def __init__(
        self,
        d_model: int,
        time_window: int = 5,
        num_heads: int = 6,
        dropout: float = 0.0,
        
    ):

        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    
        self.d_model = d_model
        self.time_window = time_window
        self.num_heads = num_heads
        self.dropout = dropout 

        self.queries = nn.Parameter(torch.randn(time_window, d_model) * 0.02)


        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.mlp = MLP(d_model)

        self.norm_x = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)

    
    def forward(self, x: torch.Tensor):
        
        """ (B*H*W, T, D) -> (B*H*W, K, D) """
        
        N, T, D = x.shape
        x = self.norm_x(x)

        q = self.queries.unsqueeze(0).expand(N, -1, -1)
        q = self.norm_q(q)

        y, _ = self.cross_attn(q, x, x)
        y = y + self.mlp(self.norm_mlp(y))
        return y


class MemoryBlock(nn.Module):

    def __init__(self, d_model, d_state=64, time_window=5, num_heads=6):
        super().__init__()

        self.time_window = time_window

        self.ssm = S4D(
            d_model=d_model, 
            d_state=d_state
        )
        
        self.pool = TimeQueryPool(
           d_model=d_model,
           time_window=time_window,
           num_heads=num_heads
        )

    def forward(self, x):

        # first reshape
        B, T, H, W, C = x.shape
        x = rearrange(x, "b t h w c -> (b h w) t c", b=B, t=T, c=C, h=H, w=W)

        x = rearrange(x, "n t c -> n c t")
        x, _ = self.ssm(x)
        x = rearrange(x, "n c t -> n t c")
        x = self.pool(x)

        K = x.shape[1]
        assert K == self.time_window, "time window is mismatched"
        x = rearrange(x, "(b h w) k c -> b k h w c", b=B, k=K, h=H, w=W, c=C)
        return x


