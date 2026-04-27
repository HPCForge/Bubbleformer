import math
import torch
import torch.nn as nn

ACTIVATION = {
    "gelu":nn.GELU(),
    "tanh":nn.Tanh(),
    "sigmoid":nn.Sigmoid(),
    "relu":nn.ReLU(),
    "leaky_relu":nn.LeakyReLU(0.1),
    "softplus":nn.Softplus(),
    "ELU":nn.ELU(),
    "silu":nn.SiLU()
}

class HMLPEmbed(nn.Module):
    """
    Image to Patch Embedding using hierarchical Conv2d.
    It preserves the spatial ordering of the patches
    Args:
        patch_size (int): Size of the square patch
        in_channels (int): Number of input channels
        embed_dim (int): Dimension of the embedding
    """
    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.patch_size = patch_size
        num_layers = int(math.log2(patch_size))
        assert (num_layers - math.log2(patch_size)) == 0, "Patch size must be a power of 2"

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        layers = []
        conv_in = in_channels
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            if num_layers == 1:
                conv_out = embed_dim
            else:
                conv_out = embed_dim if is_last else embed_dim // 4
            layers.append(
                nn.Conv2d(
                    in_channels=conv_in,
                    out_channels=conv_out,
                    kernel_size=2,
                    stride=2,
                    bias=False
                )
            )
            layers.append(nn.InstanceNorm2d(conv_out, affine=True))
            if not is_last:
                layers.append(nn.GELU())
            conv_in = conv_out
        self.in_proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, Emb, H_patches, W_patches)
        """
        x = self.in_proj(x)
        return x


class HMLPDebed(nn.Module):
    """
    Patch to Image De-bedding using hierarchical ConvTranspose2d.
    It takes a spatially ordered tensor of embedded patches and reconstructs the image
    Args:
        patch_size (int): Size of the square patch
        out_channels (int): Number of output channels
        embed_dim (int): Dimension of the embedding
    """
    def __init__(
        self,
        patch_size: int = 16,
        out_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.patch_size = patch_size
        num_layers = int(math.log2(patch_size))
        assert (num_layers - math.log2(patch_size)) == 0, "Patch size must be a power of 2"

        self.out_channels = out_channels
        self.embed_dim = embed_dim
        layers = []
        conv_in = embed_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            if num_layers == 1:
                conv_out = out_channels
            else:
                conv_out = out_channels if is_last else embed_dim // 4
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=conv_in,
                    out_channels=conv_out,
                    kernel_size=2,
                    stride=2,
                    bias=False
                )
            )
            if not is_last:
                layers.append(nn.InstanceNorm2d(conv_out, affine=True))
                layers.append(nn.GELU())
            conv_in = conv_out

        self.out_proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, Emb, H_patches, W_patches)
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        return self.out_proj(x)

class PatchEmbed(nn.Module):
    """
        Vision Transformer style Patch Embedding
        Args:
            img_size (int): Size of the input image
            patch_size (int): Size of the patch
            in_channels (int): Number of input channels
            embed_dim (int): Dimension of the embedding
            out_dim (int): Dimension of the output
            act (str): Activation function
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        out_dim: int = 128,
        act: str = 'gelu'
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.out_dim = out_dim
        self.act = ACTIVATION[act]

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv2d(embed_dim, out_dim, kernel_size=1, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, out_dim, H_patches, W_patches)
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x)
        return x
