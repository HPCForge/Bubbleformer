import math
import torch
import torch.nn as nn

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

class OverlappingEmbed(nn.Module):
    r"""
    Similar to the HMLPEmbed, but the patches are constructed using
    overlapping regions of the input.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        patch_size: int = 4,
        stride: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.stride = stride
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=4,
            stride=2,
            padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )
        
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        return x
    
class OverlappingDebed(nn.Module):
    r"""
    Similar to the HMLPDebed, but the patches are constructed using
    overlapping regions of the input.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        patch_size: int = 4,
        stride: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.stride = stride
        
        # Note: stride and kernel size are swapped vs OverlappingEmbed.
        self.conv1 = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )
        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=4,
            stride=2,
        )
        
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        return x