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
    TransformerMoEBlock,
    TransformerMoEBlockSSM
)
from bubbleformer.layers.memory_layers import MemoryBlock
from ._api import register_model

__all__ = ["NeighborMoE", "NeighborMoESSMOne", "NeighborMoESSMBlock"]


@register_model("neighbor_moe")
class NeighborMoE(nn.Module):

    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        time_window: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_fluid_params: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
    ):
        super().__init__()
        self.embed = HMLPEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
            embed_dim=embed_dim,
        )

        self.film_embed = FiLMMLP(num_fluid_params, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerMoEBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                topk=topk,
                load_balance_loss_weight=load_balance_loss_weight,
            )
            for _ in range(processor_blocks)
        ])

        self.debed = HMLPDebed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=output_fields
        )

    def forward(self, x: torch.Tensor, fluid_params: torch.Tensor):
        """
        Args:
            x: (B, T, C, H, W)
            fluid_params: (B, num_fluid_params)

        Returns:
            pred: (B, T, C, H, W)
            moe_outputs: list of TopkMoEOutput
        """
        B, T, _, _, _ = x.shape

        input = x.clone()

        # Encode
        with record_function("encode"):
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.embed(x)
            x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        embed = x.clone()

        x = rearrange(x, "b t c h w -> b t h w c").contiguous()

        # Apply FiLM conditioning on the embeddings
        with record_function("film_embed"):
            x = self.film_embed(x, fluid_params)

        moe_outputs = []
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
                x, moe_output = blk(x)
                moe_outputs.append(moe_output)

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

        return x, moe_outputs


@register_model("neighbor_moe_ssm_one")
class NeighborMoESSMOne(nn.Module):
    """
    NeighborMoE with a single S4-based temporal memory block in the middle.

    Architecture:
    - Embedding
    - First half of MoE transformer blocks (space-time attention, no state)
    - Single SSM memory block (maintains state across chunks)
    - Second half of MoE transformer blocks (space-time attention, no state)
    - De-embedding

    The SSM state can be passed between forward calls to maintain
    temporal memory across autoregressive chunks during training.
    """

    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        time_window: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_fluid_params: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
        d_state: int = 64,
        ssm_mode: str = 'diag',
    ):
        super().__init__()

        self.time_window = time_window
        self.patch_size = patch_size

        self.embed = HMLPEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
            embed_dim=embed_dim,
        )

        self.film_embed = FiLMMLP(num_fluid_params, embed_dim)

        # Split blocks into pre-SSM and post-SSM
        n_pre = processor_blocks // 2
        n_post = processor_blocks - n_pre

        self.pre_ssm_blocks = nn.ModuleList([
            TransformerMoEBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                topk=topk,
                load_balance_loss_weight=load_balance_loss_weight,
            )
            for _ in range(n_pre)
        ])

        # Single SSM memory block in the middle
        self.ssm_block = MemoryBlock(
            d_model=embed_dim,
            d_state=d_state,
            mode=ssm_mode,
        )

        self.post_ssm_blocks = nn.ModuleList([
            TransformerMoEBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                topk=topk,
                load_balance_loss_weight=load_balance_loss_weight,
            )
            for _ in range(n_post)
        ])

        self.debed = HMLPDebed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=output_fields
        )

    def forward(
        self,
        x: torch.Tensor,
        fluid_params: torch.Tensor,
        states=None,
    ):
        """
        Forward pass with optional SSM state.

        Args:
            x: (B, T, C, H, W) input tensor
            fluid_params: (B, num_fluid_params) conditioning parameters
            states: list containing single SSM state, or None

        Returns:
            pred: (B, T, C, H, W) prediction
            next_states: list containing updated SSM state
            moe_outputs: list of TopkMoEOutput for routing loss
        """
        B, T, C, H, W = x.shape

        input_skip = x.clone()

        # Encode
        with record_function("encode"):
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.embed(x)
            H_p, W_p = x.shape[-2:]  # Patch dimensions
            x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        embed_skip = x.clone()

        # Permute to (B, T, H, W, C)
        x = rearrange(x, "b t c h w -> b t h w c").contiguous()

        # Apply FiLM conditioning
        with record_function("film_embed"):
            x = self.film_embed(x, fluid_params)

        # Pre-SSM blocks (no state)
        moe_outputs = []
        for idx, blk in enumerate(self.pre_ssm_blocks):
            with record_function(f"pre_ssm_block_{idx}"):
                x, moe_output = blk(x)
                moe_outputs.append(moe_output)

        # SSM block (with state)
        # Extract state from list if provided
        state = states[0] if states is not None else None
        with record_function("ssm_block"):
            x, next_state = self.ssm_block(x, state)

        # Post-SSM blocks (no state)
        for idx, blk in enumerate(self.post_ssm_blocks):
            with record_function(f"post_ssm_block_{idx}"):
                x, moe_output = blk(x)
                moe_outputs.append(moe_output)

        x = rearrange(x, "b t h w c -> b t c h w").contiguous()

        # Skip connection from patch embeddings
        x = x + embed_skip

        # Decode
        with record_function("decode"):
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.debed(x)
            x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        # Skip connection from the original input
        x = x + input_skip

        # Return states as list for compatibility with training modules
        return x, [next_state], moe_outputs

    def default_states(self, batch_size: int, H_patch: int, W_patch: int, device=None):
        """
        Initialize SSM states.

        Args:
            batch_size: Batch size
            H_patch: Height in patches
            W_patch: Width in patches
            device: Device for state tensor

        Returns:
            states: list containing single SSM state
        """
        return [self.ssm_block.default_state(batch_size, H_patch, W_patch, device)]

    def setup_step(self):
        """Prepare for recurrent stepping mode."""
        self.ssm_block.setup_step()

    def step(
        self,
        x: torch.Tensor,
        fluid_params: torch.Tensor,
        states,
    ):
        """
        Single frame step for recurrent inference.

        Args:
            x: (B, 1, C, H, W) single frame
            fluid_params: (B, num_fluid_params)
            states: list containing single SSM state

        Returns:
            pred: (B, 1, C, H, W)
            next_states: list containing updated state
            moe_outputs: routing info
        """
        B = x.shape[0]

        input_skip = x.clone()

        # Encode single frame
        x = x.squeeze(1)  # (B, C, H, W)
        x = self.embed(x)
        H_p, W_p = x.shape[-2:]
        embed_skip = x.clone()

        # Permute to (B, H, W, C)
        x = rearrange(x, "b c h w -> b h w c")

        # FiLM conditioning
        x = self.film_embed(x.unsqueeze(1), fluid_params).squeeze(1)

        # Pre-SSM blocks (add dummy time dim for block forward)
        moe_outputs = []
        x = x.unsqueeze(1)  # (B, 1, H, W, C)
        for blk in self.pre_ssm_blocks:
            x, moe_output = blk(x)
            moe_outputs.append(moe_output)
        x = x.squeeze(1)  # (B, H, W, C)

        # SSM step - extract state from list
        state = states[0] if states is not None else None
        x, next_state = self.ssm_block.step(x, state)

        # Post-SSM blocks
        x = x.unsqueeze(1)  # (B, 1, H, W, C)
        for blk in self.post_ssm_blocks:
            x, moe_output = blk(x)
            moe_outputs.append(moe_output)
        x = x.squeeze(1)  # (B, H, W, C)

        # Back to (B, C, H, W)
        x = rearrange(x, "b h w c -> b c h w")
        x = x + embed_skip

        # Decode
        x = self.debed(x)
        x = x.unsqueeze(1)  # (B, 1, C, H, W)
        x = x + input_skip

        # Return states as list
        return x, [next_state], moe_outputs

@register_model("neighbor_moe_ssm_block")
class NeighborMoESSMInBlock(nn.Module):
    """
    NeighborMoE with S4-based temporal memory.

    The SSM state can be passed between forward calls to maintain
    temporal memory across autoregressive chunks during training.
    """

    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        time_window: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_fluid_params: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
        d_state: int = 64,
        ssm_mode: str = 'diag',
    ):
        super().__init__()

        self.time_window = time_window
        self.patch_size = patch_size

        self.embed = HMLPEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
            embed_dim=embed_dim,
        )

        self.film_embed = FiLMMLP(num_fluid_params, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerMoEBlockSSM(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                topk=topk,
                load_balance_loss_weight=load_balance_loss_weight,
                d_state=d_state,
                ssm_mode=ssm_mode,
            )
            for _ in range(processor_blocks)
        ])

        self.debed = HMLPDebed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=output_fields
        )

    def forward(
        self,
        x: torch.Tensor,
        fluid_params: torch.Tensor,
        states=None,
    ):
        """
        Forward pass with optional SSM states.

        Args:
            x: (B, T, C, H, W) input tensor
            fluid_params: (B, num_fluid_params) conditioning parameters
            states: list of SSM states (one per block), or None

        Returns:
            pred: (B, T, C, H, W) prediction
            next_states: list of updated SSM states
            moe_outputs: list of TopkMoEOutput for routing loss
        """
        B, T, C, H, W = x.shape

        input_skip = x.clone()

        # Encode
        with record_function("encode"):
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.embed(x)
            H_p, W_p = x.shape[-2:]  # Patch dimensions
            x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        embed_skip = x.clone()

        # Permute to (B, T, H, W, C)
        x = rearrange(x, "b t c h w -> b t h w c").contiguous()

        # Apply FiLM conditioning
        with record_function("film_embed"):
            x = self.film_embed(x, fluid_params)

        # Initialize states if not provided
        if states is None:
            states = [None] * len(self.blocks)

        # Process blocks with state passing
        next_states = []
        moe_outputs = []

        for idx, (blk, state) in enumerate(zip(self.blocks, states)):
            with record_function(f"block_{idx}"):
                x, next_state, moe_output = blk(x, state)
                next_states.append(next_state)
                moe_outputs.append(moe_output)

        x = rearrange(x, "b t h w c -> b t c h w").contiguous()

        # Skip connection from patch embeddings
        x = x + embed_skip

        # Decode
        with record_function("decode"):
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.debed(x)
            x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        # Skip connection from the original input
        x = x + input_skip

        return x, next_states, moe_outputs

    def default_states(self, batch_size: int, H_patch: int, W_patch: int, device=None):
        """
        Initialize SSM states for all blocks.

        Args:
            batch_size: Batch size
            H_patch: Height in patches
            W_patch: Width in patches
            device: Device for state tensors

        Returns:
            list of states, one per block
        """
        return [blk.default_state(batch_size, H_patch, W_patch, device)
                for blk in self.blocks]

    def setup_step(self):
        """Prepare all blocks for recurrent stepping mode."""
        for blk in self.blocks:
            blk.setup_step()

    def step(
        self,
        x: torch.Tensor,
        fluid_params: torch.Tensor,
        states,
    ):
        """
        Single frame step for recurrent inference.

        Args:
            x: (B, 1, C, H, W) single frame
            fluid_params: (B, num_fluid_params)
            states: list of SSM states

        Returns:
            pred: (B, 1, C, H, W)
            next_states: updated states
            moe_outputs: routing info
        """
        B = x.shape[0]

        input_skip = x.clone()

        # Encode single frame
        x = x.squeeze(1)  # (B, C, H, W)
        x = self.embed(x)
        H_p, W_p = x.shape[-2:]
        embed_skip = x.clone()

        # Permute to (B, H, W, C)
        x = rearrange(x, "b c h w -> b h w c")

        # FiLM conditioning
        x = self.film_embed(x.unsqueeze(1), fluid_params).squeeze(1)

        # Process blocks with state
        next_states = []
        moe_outputs = []

        for blk, state in zip(self.blocks, states):
            x, next_state, moe_output = blk.step(x, state)
            next_states.append(next_state)
            moe_outputs.append(moe_output)

        # Back to (B, C, H, W)
        x = rearrange(x, "b h w c -> b c h w")
        x = x + embed_skip

        # Decode
        x = self.debed(x)
        x = x.unsqueeze(1)  # (B, 1, C, H, W)
        x = x + input_skip

        return x, next_states, moe_outputs
