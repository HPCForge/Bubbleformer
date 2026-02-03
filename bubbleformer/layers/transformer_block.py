import torch
import torch.nn as nn
from torch.profiler import record_function

from bubbleformer.layers.space_time_attention import SpaceTimeAttention, SpaceSSMAttention
from bubbleformer.layers.mlp import GeluMLP
from bubbleformer.layers.moe.topk_moe import TopkMoE, TopkMoEOutput


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__()

        self.pre_norm = nn.LayerNorm(embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)

        self.attention = SpaceTimeAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.mlp = GeluMLP(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("transformer_block"):
            # Attention with a skip connection
            inp = x.clone()
            with record_function("space_time_attention"):
                x = self.attention(x)
            with record_function("pre_norm"):          
                x = self.pre_norm(x) + inp

            # MLP with a skip connection
            intermediate = x.clone()
            with record_function("mlp"):
                x = self.mlp(x)
            with record_function("post_norm"):
                x = self.post_norm(x) + intermediate

        return x
    
class TransformerMoEBlock(TransformerBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
    ):
        super().__init__(embed_dim, num_heads)
        
        self.mlp = TopkMoE(
            num_experts=num_experts,
            hidden_dim=embed_dim,
            intermediate_dim=embed_dim * 4,
            topk=topk,
            load_balance_loss_weight=load_balance_loss_weight,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("transformer_moe_block"):
            # Attention with a skip connection
            inp = x.clone()
            with record_function("space_time_attention"):
                x = self.attention(x)
            with record_function("pre_norm"):          
                x = self.pre_norm(x) + inp

            # MLP with a skip connection
            intermediate = x.clone()
            with record_function("mlp"):
                moe_output: TopkMoEOutput = self.mlp(x)
                x = moe_output.out
            with record_function("post_norm"):
                x = self.post_norm(x) + intermediate

        return x, moe_output

class TransformerBlockSSM(nn.Module):
    """Basic transformer block with space-time attention and MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        d_state: int = 64,
        ssm_mode: str = 'diag',
    ):
        super().__init__()

        self.pre_norm = nn.LayerNorm(embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)

        self.attention = SpaceSSMAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            d_state=d_state,
            ssm_mode=ssm_mode,
        )

        self.mlp = GeluMLP(embed_dim)

    def forward(self, x: torch.Tensor, state=None):
        """
        Args:
            x: (B, T, H, W, C)
            state: SSM state from previous chunk

        Returns:
            y: (B, T, H, W, C)
            next_state: SSM state for next chunk
        """
        with record_function("transformer_block"):
            # Attention with skip connection
            inp = x.clone()
            with record_function("space_time_attention"):
                x, next_state = self.attention(x, state)
            with record_function("pre_norm"):
                x = self.pre_norm(x) + inp

            # MLP with skip connection
            intermediate = x.clone()
            with record_function("mlp"):
                x = self.mlp(x)
            with record_function("post_norm"):
                x = self.post_norm(x) + intermediate

        return x, next_state

    def default_state(self, batch_size: int, H: int, W: int, device=None):
        return self.attention.default_state(batch_size, H, W, device)


class TransformerMoEBlockSSM(nn.Module):
    """Transformer block with MoE feedforward and S4-based temporal memory."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
        d_state: int = 64,
        ssm_mode: str = 'diag',
    ):
        super().__init__()

        self.pre_norm = nn.LayerNorm(embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)

        self.attention = SpaceSSMAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            d_state=d_state,
            ssm_mode=ssm_mode,
        )

        self.mlp = TopkMoE(
            num_experts=num_experts,
            hidden_dim=embed_dim,
            intermediate_dim=embed_dim * 4,
            topk=topk,
            load_balance_loss_weight=load_balance_loss_weight,
        )

    def forward(self, x: torch.Tensor, state=None):
        """
        Args:
            x: (B, T, H, W, C)
            state: SSM state from previous chunk

        Returns:
            y: (B, T, H, W, C)
            next_state: SSM state for next chunk
            moe_output: TopkMoEOutput with routing info and load balance loss
        """
        with record_function("transformer_moe_block"):
            # Attention with skip connection and state
            inp = x.clone()
            with record_function("space_time_attention"):
                x, next_state = self.attention(x, state)
            with record_function("pre_norm"):
                x = self.pre_norm(x) + inp

            # MoE with skip connection
            intermediate = x.clone()
            with record_function("moe"):
                moe_output: TopkMoEOutput = self.mlp(x)
                x = moe_output.out
            with record_function("post_norm"):
                x = self.post_norm(x) + intermediate

        return x, next_state, moe_output

    def default_state(self, batch_size: int, H: int, W: int, device=None):
        return self.attention.default_state(batch_size, H, W, device)

    def setup_step(self):
        """Prepare for recurrent stepping mode."""
        self.attention.setup_step()

    def step(self, x: torch.Tensor, state):
        """
        Single frame step for recurrent inference.

        Args:
            x: (B, H, W, C) single frame
            state: SSM state

        Returns:
            y: (B, H, W, C) output
            next_state: updated SSM state
            moe_output: routing info
        """
        # Attention with state
        inp = x.clone()
        x, next_state = self.attention.step(x, state)
        x = self.pre_norm(x) + inp

        # MoE
        intermediate = x.clone()
        # Add dummy dims for MoE which expects (B, T, H, W, C)
        x = x.unsqueeze(1)
        moe_output: TopkMoEOutput = self.mlp(x)
        x = moe_output.out.squeeze(1)
        x = self.post_norm(x) + intermediate

        return x, next_state, moe_output
