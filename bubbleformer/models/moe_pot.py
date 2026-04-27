"""
Moe-POT architecture. Thanks to the authors of the original implementation.
Paper: "Mixture-of-Experts Operator Transformer for Large-Scale 
        PDE Pre-Training (NeurIPS 2025)"
Based on: "DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale
        PDE Pre-Training (ICML 2024)"
Github: https://github.com/haiyangxin/MoEPOT/tree/main
"""
import numpy as np
import torch
import torch.fft
import torch.nn as nn

from einops import rearrange
from bubbleformer.layers import MoEImage, PatchEmbed

ACTIVATION = {
    "gelu":nn.GELU(),
    "tanh":nn.Tanh(),
    "sigmoid":nn.Sigmoid(),
    "relu":nn.ReLU(),
    "leaky_relu":nn.LeakyReLU(0.1),
    "softplus":nn.Softplus(),
    "elu":nn.ELU(),
    "silu":nn.SiLU()
}

class AFNO2D(nn.Module):
    """
    2D Adaptive Fourier Neural Operator layer.
    Paper: "Adaptive Fourier Neural Operator: Efficient Token Mixers for Transformers (ICLR 2022)"
    Args:
        width (int): Number of input channels
        num_blocks (int): Number of Fourier blocks
        channel_first (bool): Whether the input tensor is in channel-first format
        sparsity_threshold (float): Threshold for sparsity in Fourier domain
        modes (int): Number of Fourier modes to keep
        hidden_size_factor (int): Expansion factor for hidden size
        act (str): Activation function to use
    """
    def __init__(
        self,
        width: int = 32,
        num_blocks: int = 8,
        channel_first: bool = False,
        sparsity_threshold: float = 0.01,
        modes: int = 32,
        hidden_size_factor: int = 1,
        act: str = 'gelu'
    ):
        super().__init__()
        assert width % num_blocks == 0, \
            f"hidden_size {width} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.hidden_size_factor = hidden_size_factor
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)

        self.act = ACTIVATION[act]

        self.w1 = nn.Parameter(
            self.scale * torch.rand(
                2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor
            ))
        self.b1 = nn.Parameter(
            self.scale * torch.rand(
                2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.rand(
                2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(
            self.scale * torch.rand(
                2, self.num_blocks, self.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) or (B, H, W, C)
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W) or (B, H, W, C)
        """
        if self.channel_first:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  ### ->N, X, Y, C
        else:
            B, H, W, C = x.shape
        x_orig = x

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")

        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros([
            B,
            x.shape[1],
            x.shape[2],
            self.num_blocks,
            self.block_size * self.hidden_size_factor
            ], device=x.device)
        o1_imag = torch.zeros([
            B,
            x.shape[1],
            x.shape[2],
            self.num_blocks,
            self.block_size * self.hidden_size_factor
            ], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # total_modes = H*W // 2 + 1
        kept_modes = self.modes

        o1_real[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")

        x = x + x_orig
        if self.channel_first:
            x = x.permute(0, 3, 1, 2)     ### N, C, X, Y

        return x


class TimeAggregator(nn.Module):
    """
    DPOT Time Aggregation Module using MLP or Exponential MLP
    Args:
        n_channels (int): Number of input channels
        n_timesteps (int): Number of time steps
        out_channels (int): Number of output channels
        type (str): Type of aggregation ('mlp' or 'exp_mlp')
    """
    def __init__(
        self,
        n_channels: int,
        n_timesteps: int,
        out_channels: int,
        type: str = 'mlp'
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.out_channels = out_channels
        self.type = type
        if self.type == 'mlp':
            self.w = nn.Parameter(
                1/(n_timesteps * out_channels**0.5) * torch.randn(
                    n_timesteps, out_channels, out_channels
                    ),
                requires_grad=True
            )   # initialization could be tuned
        elif self.type == 'exp_mlp':
            self.w = nn.Parameter(
                1/(n_timesteps * out_channels**0.5) * torch.randn(
                    n_timesteps, out_channels, out_channels
                    ),
                requires_grad=True
            )   # initialization could be tuned
            self.gamma = nn.Parameter(
                2**torch.linspace(-10,10, out_channels).unsqueeze(0),
                requires_grad=True
            )  # 1, C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, T, C)
        Returns:
            torch.Tensor: Output tensor of shape (B, H, W, C)
        """
        if self.type == 'mlp':
            x = torch.einsum('tij, ...ti->...j', self.w, x)
        elif self.type == 'exp_mlp':
            t = torch.linspace(0, 1, x.shape[-2]).unsqueeze(-1).to(x.device) # T, 1
            t_embed = torch.cos(t @ self.gamma)
            x = torch.einsum('tij,...ti->...j', self.w, x * t_embed)

        return x

class MoEPOTBlock(nn.Module):
    def __init__(
        self,
        mixing_type: str = 'afno',
        double_skip: bool = True,
        width: int = 32,
        n_blocks: int = 4,
        mlp_ratio: float = 1.,
        channel_first: bool = True,
        modes: int = 32,
        act: str = 'gelu',
        is_finetune: bool = False
    ):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(8, width)
        self.width = width
        self.modes = modes
        self.act = ACTIVATION[act]

        if mixing_type == "afno":
            self.filter = AFNO2D(
                            width=width,
                            num_blocks=n_blocks,
                            sparsity_threshold=0.01,
                            channel_first=channel_first,
                            modes=modes,
                            hidden_size_factor=1,
                            act=act
                        )
        else:
            raise NotImplementedError(f"Mixing type {mixing_type} not implemented.")

        self.norm2 = torch.nn.GroupNorm(8, width)



        mlp_hidden_dim = int(width * mlp_ratio)
        self.MoE = MoEImage(
                    input_channels=width,
                    hidden_channels=mlp_hidden_dim,
                    output_channels=width,
                    num_experts=16,
                    shared_experts_num=2,
                    top_k=4,
                    is_finetune=is_finetune
        )
        if is_finetune: # Freeze Gate Control Network
            self.MoE.freeze_feature_and_gating()
            self.MoE.feature_extractor.eval()
            self.MoE.gating.eval()
        self.double_skip = double_skip

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
            torch.Tensor: Gate loss tensor
        """
        residual = x
        x = self.norm1(x)
        x = self.filter(x)


        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x, loss_gate = self.MoE(x)

        x = x + residual

        return x, loss_gate


class MoEPOTNet(nn.Module):
    """
    Mixture-of-Experts Path Operator Transformer (MoE-POT) for PDE modeling.
    Args:
        img_size (int): Input image size
        patch_size (int): Patch size
        mixing_type (str): Type of mixer, default is 'afno'
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        in_timesteps (int): Number of input time steps
        out_timesteps (int): Number of output time steps
        n_blocks (int): Number of blocks in the mixer
        embed_dim (int): Embedding dimension
        out_layer_dim (int): Dimension of output convolutional layer
        depth (int): Number of MoEPOT blocks
        modes (int): Number of Fourier modes
        mlp_ratio (float): MLP expansion ratio
        n_cls (int): Number of classes (datasets)
        normalize (bool): Whether to normalize input data
        act (str): Activation function
        time_agg (str): Type of temporal aggregation layer
        is_finetune (bool): Whether in finetuning mode
        """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        mixing_type: str = 'afno',
        in_channels: int = 1,
        out_channels: int = 1,
        in_timesteps: int = 10,
        out_timesteps: int = 1,
        n_blocks: int = 4,
        embed_dim: int = 768,
        out_layer_dim: int = 32,
        depth: int = 12,
        modes: int = 32,
        mlp_ratio: float = 1.,
        n_cls: int = 6,
        normalize: bool = False,
        act: str = 'gelu',
        time_agg: str = 'exp_mlp',
        is_finetune: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.n_blocks = n_blocks
        self.modes = modes
        self.num_features = self.embed_dim = embed_dim  # for consistency with other models
        self.mlp_ratio = mlp_ratio
        self.act = ACTIVATION[act]
        self.patch_embed = PatchEmbed(
                            img_size=img_size,
                            patch_size=patch_size,
                            in_channels=in_channels + 3,
                            embed_dim=out_channels * patch_size + 3,
                            out_dim=embed_dim,act=act
                        )
        self.latent_size = self.patch_embed.out_size
        self.pos_embed = nn.Parameter(
                            torch.zeros(
                                1,
                                embed_dim,
                                self.patch_embed.out_size[0],
                                self.patch_embed.out_size[1]
                            )
                        )
        self.normalize = normalize
        self.time_agg = time_agg
        self.n_cls = n_cls
        self.is_finetune = is_finetune

        self.blocks = nn.ModuleList([
                        MoEPOTBlock(
                            mixing_type=mixing_type,
                            double_skip=False,
                            width=embed_dim,
                            n_blocks=n_blocks,
                            mlp_ratio=mlp_ratio,
                            channel_first=True,
                            modes=modes,
                            act=act,
                            is_finetune=is_finetune)
            for i in range(depth)])


        if self.normalize:
            self.scale_feats_mu = nn.Linear(2 * in_channels, embed_dim)
            self.scale_feats_sigma = nn.Linear(2 * in_channels, embed_dim)


        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.act,
            nn.Linear(embed_dim, embed_dim),
            self.act,
            nn.Linear(embed_dim, n_cls)
        )

        self.time_agg_layer = TimeAggregator(in_channels, in_timesteps, embed_dim, time_agg)

        ### attempt load balancing for high resolution
        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=embed_dim,
                out_channels=out_layer_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            self.act,
            nn.Conv2d(
                in_channels=out_layer_dim,
                out_channels=out_layer_dim,
                kernel_size=1,
                stride=1
            ),
            self.act,
            nn.Conv2d(
                in_channels=out_layer_dim,
                out_channels=self.out_channels * self.out_timesteps,
                kernel_size=1,
                stride=1
            )
        )

        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.mixing_type = mixing_type


    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=.002)    # .02
            if m.bias is not None:
            # if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def _get_grid_3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get 3D grid for spatial coordinates.
        Args:
            x (torch.Tensor): Input tensor of shape (B, X, Y, Z, C)
        Returns:
            torch.Tensor: Grid tensor of shape (B, X, Y, Z, 3)
        """
        batchsize, size_x, size_y, size_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).to(x.device).repeat(
            [batchsize, 1, size_y, size_z, 1]
        )
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).to(x.device).repeat(
            [batchsize, size_x, 1, size_z, 1]
        )
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).to(x.device).repeat(
            [batchsize, size_x, size_y, 1, 1]
        )
        grid = torch.cat((gridx, gridy, gridz), dim=-1)

        return grid

    ### in/out: B, T, C, X, Y
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, X, Y)
        Returns:
            torch.Tensor: Output tensor of shape (B, out_timesteps, out_channels, X, Y)
        """
        # reshape bubbleformer input B, T, C, X, Y to match DPOT input B, X, Y, T, C
        x = x.permute(0, 3, 4, 1, 2)  # B, X, Y, T, C

        B, _, _, T, _ = x.shape # [8,128,128,10,1]
        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3),keepdim=True), \
                        x.std(dim=(1,2,3),keepdim=True) + 1e-6
            x = (x - mu)/ sigma
            scale_mu = self.scale_feats_mu(
                            torch.cat([mu, sigma],dim=-1)
                        ).squeeze(-2).permute(0,3,1,2)
            scale_sigma = self.scale_feats_sigma(
                            torch.cat([mu, sigma], dim=-1)
                        ).squeeze(-2).permute(0, 3, 1, 2)

        grid = self._get_grid_3d(x)
        x = torch.cat((x, grid), dim=-1).contiguous() # B, X, Y, T, C+3
        x = rearrange(x, 'b x y t c -> (b t) c x y')
        x = self.patch_embed(x)
        x = x + self.pos_embed

        x = rearrange(x, '(b t) c x y -> b x y t c', b=B, t=T)

        x = self.time_agg_layer(x)

        x = rearrange(x, 'b x y c -> b c x y')

        if self.normalize:
            x = scale_sigma * x + scale_mu   ### Ada_in layer

        loss_total = 0
        for blk in self.blocks:
            x, loss = blk(x)
            loss_total += loss

        cls_token = x.mean(dim=(2, 3), keepdim=False)
        cls_pred = self.cls_head(cls_token)

        x = self.out_layer(x).permute(0, 2, 3, 1)
        x = x.reshape(*x.shape[:3], self.out_timesteps, self.out_channels).contiguous()

        if self.normalize:
            x = x * sigma  + mu

        # reshape back to bubbleformer output B, out_timesteps, out_channels, X, Y
        x = x.permute(0, 3, 4, 1, 2)  # B, out_timesteps, out_channels, X, Y

        return x, cls_pred, loss_total
