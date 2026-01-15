import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import numpy as np
from einops import rearrange

from bubbleformer.layers import AxialAttentionBlock, AttentionBlock, HMLPEmbed, HMLPDebed, FiLMMLP
from ._api import register_model
from bubbleformer.layers.VMamba.vmamba import VSSM, VSSBlock
from bubbleformer.layers.s4.models.s4.s4d import S4D
from bubbleformer.layers.s4.models.s4.s4 import S4Block

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
        #print("after temporal ", x.shape)

        # Now do spatial attention
        x = rearrange(x, "b t emb h w -> (b t) emb h w")        # BT sequences
        x = self.spatial(x)
        #print("after spatial ", x.shape)# A spatial encoder block
        x = rearrange(x, "(b t) emb h w -> b t emb h w", t=t)
        #print("after spacetime block", x.shape)
        return x    # (B, T, emb, H, W)


class SpaceTimeSSMBlock(nn.Module):

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
            attn_scale=feat_scale,
        )


        self.ssm = S4D(
            d_model=embed_dim,
            d_state=64
        )
    
    """
    def forward(self, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        #context: [B, T_ctx, C, H, W]
        #x:       [B, K,     C, H, W]  (prediction window)
        #returns: [B, K,     C, H, W]  (processed last K steps)

        #We:
        #  - concatenate context + x along time
        #  - run SSM with burn-in over context tokens
        #  - keep only last K timesteps
        #  - run temporal & spatial attention over those K timesteps
        # Concatenate context + current window
        # If T_ctx == 0, this is just x.
        x_full = torch.cat((context, x), dim=1)   # [B, T_total, C, H, W]
        B, T_total, C, H, W = x_full.shape

        # Number of timesteps we want to keep / predict
        K = x.shape[1]   # prediction window length

        # Flatten time + spatial dims so SSM sees a 1D sequence
        x_flat = rearrange(x_full, "b t emb h w -> b emb (t h w)")   # [B, C, L]
        L = x_flat.size(-1)
        L_tail = K * H * W

        # ---- SSM: burn-in if we have extra context, else just process all ----
        if L <= L_tail:
            # No burn-in possible (e.g. T_total == K)
            # Just run SSM over the full sequence.
            y_full_flat, _ = self.ssm(x_flat)                         # [B, C, L]
            tail_flat = y_full_flat[:, :, -L_tail:]                   # last K steps
        else:
            # Proper burn-in on first L_burn tokens (no grad), then train on tail
            L_burn = L - L_tail                                       # > 0 here
            #with torch.no_grad():
            _, state = self.ssm(x_flat[:, :, :L_burn])            # burn-in
            tail = x_flat[:, :, L_burn:]                              # [B, C, L_tail]
            tail_flat, _ = self.ssm(tail, state=state)               # with grad

        # Reshape back to [B, K, C, H, W]
        y = rearrange(
            tail_flat,
            "b emb (t h w) -> b t emb h w",
            t=K, h=H, w=W
        )

        # ---- temporal attention over K timesteps ----
        y = self.temporal(y)                                          # [B, K, C, H, W]

        # ---- spatial attention ----
        y = rearrange(y, "b t emb h w -> (b t) emb h w")              # [B*K, C, H, W]
        y = self.spatial(y)                                           # [B*K, C, H, W]
        y = rearrange(y, "(b t) emb h w -> b t emb h w", t=K)         # [B, K, C, H, W]

        return y
   """ 
    


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        context: [B, T_ctx, C, H, W]
        x:       [B, K,     C, H, W]   (prediction window)
        returns: [B, K,     C, H, W]

        Design:
          - Concatenate context + x along time.
          - Run S4D once over the *entire* temporal sequence (per spatial location).
          - Take the last K timesteps from the S4D output.
          - Feed those K timesteps to temporal + spatial attention.
        """

        # -----------------------------
        # 1. Concatenate context + x
        # -----------------------------
        context = x[:, :-5, :, :, :]
        x = x[:, -5:, :, :, :]
        B, T, C, H, W = x.shape
        K = 5

        if context is not None and context.shape[1] > 0:
            x_full = torch.cat((context, x), dim=1)   # [B, T_total, C, H, W]
        else:
            x_full = x                                # [B, K, C, H, W]

        B, T_total, C, H, W = x_full.shape

        # -----------------------------------------
        # 2. Reshape so S4D sees time as sequence
        # -----------------------------------------
        # For each spatial location (h, w), we have a length-T_total time-series.
        # S4D (with transposed=True) expects input shape [B_eff, C, L_time].
        x_seq = rearrange(
            x_full,
            "b t c h w -> (b h w) c t"
        )  # [B*H*W, C, T_total]

        # -----------------------------------------
        # 3. Single S4D call over full time axis
        # -----------------------------------------
        # S4D is a pure convolution over the temporal length dim.
        y_seq, _ = self.ssm(x_seq)   # [B*H*W, C, T_total]

        # ------------------------------------------------
        # 4. Take only the last K timesteps for prediction
        # ------------------------------------------------
        # These are the "transformed" versions of the last K input frames,
        # each of which can now depend on *all* previous timesteps (context).
        
        y_tail = y_seq[:, :, -K:]    # [B*H*W, C, K]
        y_pref = y_seq[:, :, :-K]

        # -----------------------------------------
        # 5. Reshape back to [B, K, C, H, W]
        # -----------------------------------------


        y = rearrange(
            y_tail,
            "(b h w) c t -> b t c h w",
            b=B, h=H, w=W
        )  # [B, K, C, H, W]
        
        y_pref = rearrange(
            y_pref,
            "(b h w) c t -> b t c h w",
            b=B, h=H, w=W
        )

        # -----------------------------------------
        # 6. Temporal attention over these K steps
        # -----------------------------------------
        # self.temporal is your existing temporal attention block that
        # expects [B, K, C, H, W] and returns [B, K, C, H, W].
        y = self.temporal(y)         # [B, K, C, H, W]

        # -----------------------------------------
        # 7. Spatial attention (per timestep)
        # -----------------------------------------
        # self.spatial is your existing spatial attention block that
        # operates on [B*K, C, H, W].
        y = rearrange(y, "b t c h w -> (b t) c h w")  # [B*K, C, H, W]
        y = self.spatial(y)                           # [B*K, C, H, W]
        y = rearrange(y, "(b t) c h w -> b t c h w", t=K)

        return torch.cat((y_pref, y), dim=1)
        #return y

    def step(self, x: torch.Tensor, state=None):
        B, T, C, H, W = x.shape

        x_flat = rearrange(x, "b t c h w -> b c (t h w)")
        y_flat, new_state = self.ssm(x_flat, state=state)

        y = rearrange(y_flat, "b c (t h w) -> b t c h w", t=T, h=H, w=W)
        y = self.temporal(y)
        y = self.spatial(x)


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
        block_type: str = "st"
    ):
        super().__init__()
        self.embed = HMLPEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
            embed_dim=embed_dim,
        )

        if block_type == "st":
            BlockType = SpaceTimeBlock
        elif block_type == "st_ssm":
            BlockType = SpaceTimeSSMBlock
        

        self.film_embed = FiLMMLP(num_fluid_params, embed_dim)
        # self.film_blocks = nn.ModuleList([
        #     FiLMMLP(num_fluid_params, embed_dim) for _ in range(processor_blocks)
        # ])
        

        self.dp = np.linspace(0, drop_path, processor_blocks)
        self.blocks = nn.ModuleList([
            BlockType(
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
        #print("after embedding: ", x.shape)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        # Apply FiLM conditioning on the embeddings
        x = self.film_embed(x, fluid_params)  # (B, T, C, H, W)
        #print("after film: ", x.shape)

        # Process with FiLM-modulated blocks
        # for blk, film in zip(self.blocks, self.film_blocks):
        for blk in self.blocks:
            x = blk(x)
            # x = film(x, fluid_params)

        # Decode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.debed(x)
        #print("after debed ", x.shape)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)
        #print("before return ", x.shape)
        return x

@register_model("vmamba_filmavit")
class VMambaFiLMConditionedAViT(FiLMConditionedAViT):
 
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
        vmamba_mlp_ratio: float = 4.0,
        vmamba_dims: int = 48,
        imgsize: int = 512
    ):

        super().__init__(

            input_fields=input_fields,
            output_fields=output_fields,
            time_window=time_window,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            processor_blocks=processor_blocks,
            drop_path=drop_path,
            attn_scale=attn_scale,
            feat_scale=feat_scale,
            num_fluid_params=num_fluid_params,
        )
        
        """


        self.vmamba_block = VSSBlock(
            hidden_dim=embed_dim,
            drop_path=0.2,
            channel_first=True,
            mlp_ratio=vmamba_mlp_ratio,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate = 0.0,
        )
        """

    def forward(self, x, fluid_params):
        
        B, T, C, H, W = x.shape
        
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed(x)
        #x = self.vmamba_block(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

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


@register_model("filmavit_ssm")
class FiLMConditionedAViTSSM(FiLMConditionedAViT):
 
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

        super().__init__(

            input_fields=input_fields,
            output_fields=output_fields,
            time_window=time_window,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            processor_blocks=processor_blocks,
            drop_path=drop_path,
            attn_scale=attn_scale,
            feat_scale=feat_scale,
            num_fluid_params=num_fluid_params,
            block_type="st_ssm"
        )
        
        self.time_window = time_window 

    def forward(self, x, fluid_params):
        
        B, T, C, H, W = x.shape
        
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed(x)
        #x = self.vmamba_block(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        x = self.film_embed(x, fluid_params)  # (B, T, C, H, W)
        # Process with FiLM-modulated blocks
        # for blk, film in zip(self.blocks, self.film_blocks):

        for blk in self.blocks:
            x = blk(x)
            # x = film(x, fluid_params)

        # Decode
        x = x[:, -self.time_window:, :, :, :]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.debed(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=self.time_window)
        print("return shape: ", x.shape)
        return x

