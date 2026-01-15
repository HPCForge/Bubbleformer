import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Wide Residual Blocks used in modern Unet architectures.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (nn.Module): Activation function to use.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module = nn.GELU(),
        norm: bool = True,
        n_groups: int = 8,
    ):
        super().__init__()
        self.activation: nn.Module = activation

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_ch, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, out_ch, H, W)
        """
        h = self.conv1(self.activation(self.norm1(x)))
        h = self.conv2(self.activation(self.norm2(h)))

        return h + self.shortcut(x)


class MiddleBlock(nn.Module):
    """
    It is a ResidualBlock, followed by another ResidualBlock.
    This block is applied at the lowest resolution of the U-Net.
    Args:
        in_channels (int): Number of channels in the input and output.
        activation (nn.Module): Activation function to use. Defaults to nn.GELU().
        norm (bool, optional): Whether to use normalization. Defaults to False.
    """
    def __init__(
        self,
        in_channels: int,
        activation: nn.Module = nn.GELU(),
        norm: bool = True
    ):
        super().__init__()
        self.res1 = ResidualBlock(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        activation=activation,
                        norm=norm
                    )
        self.res2 = ResidualBlock(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        activation=activation,
                        norm=norm
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        x = self.res1(x)
        x = self.res2(x)
        return x


class ClassicUnetBlock(nn.Module):
    """
    A single block of the classic U-Net architecture from Ronneberger et al. (2015).
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels  
    """
    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(num_features=out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_ch, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, out_ch, H, W)
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x