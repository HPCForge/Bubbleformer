from typing import List
import torch
import torch.nn as nn
from einops import rearrange

from bubbleformer.layers import ClassicUnetBlock, ResidualBlock, MiddleBlock
from ._api import register_model

__all__ = ["ModernUnet", "ClassicUnet"]

class Upsample(nn.Module):
    """Scale up the feature map by 2 times
    Args:
        in_channels (int): Number of channels in the input and output.
    """
    def __init__(
        self,
        in_channels: int
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, C, 2H, 2W)
        """
        return self.conv(x)


class Downsample(nn.Module):
    r"""Scale down the feature map by 1/2 times
    Args:
        in_channels (int): Number of channels in the input and output.
    """
    def __init__(
        self,
        in_channels: int
    ):
        super().__init__()
        self.conv = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H/2, W/2)
        """
        return self.conv(x)


@register_model("unet_modern")
class ModernUnet(nn.Module):
    """
    Modern U-Net architecture with residual connections.
    Args:
        fields (int): Number of fields
        hidden_channels (int): Number of hidden channels
        ch_mults (List[int]): List of channel multipliers for each block
        norm (bool): Whether to use normalization
    """
    def __init__(
        self,
        time_window: int = 5,
        input_fields: int = 4,
        output_fields: int = 4,
        hidden_channels: int = 32,
        ch_mults: List[int] = [],
        norm: bool = True,
    ):
        super().__init__()
        self.time_window = time_window
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.hidden_channels = hidden_channels
        self.ch_mults = ch_mults
        self.norm = norm

        self.activation = nn.GELU()
        in_channels = input_fields * time_window
        final_out = output_fields * time_window

        self.image_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = hidden_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            for _ in range(2):
                down.append(
                    ResidualBlock(in_channels, out_channels)
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(2):
                up.append(
                    ResidualBlock(in_channels + out_channels, out_channels)
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(ResidualBlock(in_channels + out_channels, out_channels))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, in_channels)
        else:
            self.norm = nn.Identity()

        self.final = nn.Conv2d(in_channels, final_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C, H, W)
        """
        x = rearrange(x, "b t c h w -> b (t c) h w")
        x = self.image_proj(x)
        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)

        x = self.final(self.activation(self.norm(x)))
        x = rearrange(x, "b (t c) h w -> b t c h w", t = self.time_window)
        return x


@register_model("unet_classic")
class ClassicUnet(nn.Module):
    """
    Classic U-Net architecture from Ronneberger et al. (2015).
    Args:
        fields (int): Number of fields
        hidden_channels (int): Number of hidden channels
    """
    def __init__(
        self,
        time_window: int = 5,
        input_fields: int = 4,
        output_fields: int = 4,
        hidden_channels: int = 32,
    ):
        super().__init__()
        self.time_window = time_window
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.hidden_channels = hidden_channels

        self.encoder1 = ClassicUnetBlock(
                            in_channels=input_fields * time_window,
                            out_channels=hidden_channels
                        )
        self.pool1 = nn.MaxPool2d(
                            kernel_size=2,
                            stride=2
                        )

        self.encoder2 = ClassicUnetBlock(
                            in_channels=hidden_channels,
                            out_channels=hidden_channels * 2
                        )
        self.pool2 = nn.MaxPool2d(
                            kernel_size=2,
                            stride=2
                        )

        self.encoder3 = ClassicUnetBlock(
                            in_channels=hidden_channels * 2,
                            out_channels=hidden_channels * 4
                        )
        self.pool3 = nn.MaxPool2d(
                            kernel_size=2,
                            stride=2
                        )

        self.encoder4 = ClassicUnetBlock(
                            in_channels=hidden_channels * 4,
                            out_channels=hidden_channels * 8
                        )
        self.pool4 = nn.MaxPool2d(
                            kernel_size=2,
                            stride=2
                        )

        self.bottleneck = ClassicUnetBlock(
                            in_channels=hidden_channels * 8,
                            out_channels=hidden_channels * 16
                        )

        self.upconv4 = nn.ConvTranspose2d(
                            in_channels=hidden_channels * 16,
                            out_channels=hidden_channels * 8,
                            kernel_size=2,
                            stride=2
                        )
        self.decoder4 = ClassicUnetBlock(
                            in_channels=hidden_channels * 16,
                            out_channels=hidden_channels * 8
                        )

        self.upconv3 = nn.ConvTranspose2d(
                            in_channels=hidden_channels * 8,
                            out_channels=hidden_channels * 4,
                            kernel_size=2,
                            stride=2
                        )
        self.decoder3 = ClassicUnetBlock(
                            in_channels=hidden_channels * 8,
                            out_channels=hidden_channels * 4
                        )

        self.upconv2 = nn.ConvTranspose2d(
                            in_channels=hidden_channels * 4,
                            out_channels=hidden_channels * 2,
                            kernel_size=2,
                            stride=2
                        )
        self.decoder2 = ClassicUnetBlock(
                            in_channels=hidden_channels * 4,
                            out_channels=hidden_channels * 2
                        )
        self.upconv1 = nn.ConvTranspose2d(
                            in_channels=hidden_channels * 2,
                            out_channels=hidden_channels,
                            kernel_size=2,
                            stride=2
                        )
        self.decoder1 = ClassicUnetBlock(
                            in_channels=hidden_channels * 2,
                            out_channels=hidden_channels
                        )

        self.conv = nn.Conv2d(
                            in_channels=hidden_channels,
                            out_channels=output_fields * time_window,
                            kernel_size=1
                        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C, H, W)
        """
        x = rearrange(x, "b t c h w -> b (t c) h w")
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        x = self.conv(dec1)
        x = rearrange(x, "b (t c) h w -> b t c h w", t = self.time_window)

        return x
    