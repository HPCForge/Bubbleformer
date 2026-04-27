"""
Layers for Moe-POT architecture.
Credits: Authors of: "Mixture-of-Experts Operator Transformer for Large-Scale 
                                   PDE Pre-Training (NeurIPS 2025)"
Github: https://github.com/haiyangxin/MoEPOT/tree/main
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bubbleformer.layers import ConvFeatureExtractor

class GlobalTopKGating(nn.Module):
    """
    Global Top-K Gating Network for MoE.
    Args:
        input_dim (int): Dimension of the input features
        num_experts (int): Number of experts in the MoE
        top_k (int): Number of top experts to select
        initial_temperature (float): Initial temperature for softmax
        is_finetune (bool): Whether in finetuning mode
    """
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        initial_temperature: float = 2.0,
        is_finetune: bool = False
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = initial_temperature
        self.min_temperature = 0.5
        self.temperature_decay = 0.99

        if is_finetune:
            self.temperature = 0.5

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.gate = nn.Sequential(
            nn.Conv2d(input_dim, input_dim*2, 1),
            nn.BatchNorm2d(input_dim*2),
            nn.GELU(),
            ChannelAttention(input_dim*2),
            nn.Conv2d(input_dim*2, input_dim, 1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, num_experts, 1)
        )

    def update_temperature(self):
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.temperature_decay
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - top_k_indices (torch.Tensor): Indices of top-k experts, shape (B, top_k)
                - top_k_values (torch.Tensor): Weights of top-k experts, shape (B, top_k)
        """
        global_feat = self.global_pool(x)  # [B, C, 1, 1]
        gating_scores = self.gate(global_feat).squeeze(-1).squeeze(-1)  # [B, num_experts]

        top_k_values, top_k_indices = torch.topk(gating_scores, self.top_k, dim=1)  # [B, top_k]
        top_k_values = F.softmax(top_k_values / self.temperature, dim=1)

        return top_k_indices, top_k_values


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.
    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for the intermediate layer
    """
    def __init__(
        self,
        channels: int,
        reduction: int = 16
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class MoEImage(nn.Module):
    """
    Mixture of Experts Convolutional Layer for images.
    Args:
        input_channels (int): Number of input channels
        hidden_channels (int): Number of hidden channels
        output_channels (int): Number of output channels
        num_experts (int): Number of experts in the MoE
        shared_experts_num (int): Number of shared experts
        top_k (int): Number of top experts to select
        is_finetune (bool): Whether in finetuning mode
    """
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int, 
        num_experts: int,
        shared_experts_num: int = 2,
        top_k: int = 4,
        is_finetune: bool = False
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_experts = num_experts
        self.shared_experts_num = shared_experts_num
        self.top_k = top_k
        self.is_finetune = is_finetune

        self.feature_extractor = ConvFeatureExtractor(input_channels, hidden_channels)
        self.gating = GlobalTopKGating(
            hidden_channels,
            num_experts,
            top_k,
            is_finetune=self.is_finetune
        )

        self.shared_experts = nn.ModuleList([
            ConvFeatureExtractor(hidden_channels, output_channels) 
            for _ in range(shared_experts_num)
        ])

        self.experts = nn.ModuleList([
            ConvFeatureExtractor(hidden_channels, output_channels) 
            for _ in range(num_experts)
        ])

    def freeze_feature_and_gating(self, freeze=True):
        """
        Freeze or unfreeze feature extractor and gating network.
        Args:
            freeze (bool): Whether to freeze the layers. Defaults to True.
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze
        for param in self.gating.parameters():
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        features = self.feature_extractor(x)

        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output += expert(features) / self.shared_experts_num

        output = torch.zeros_like(x)
        output = torch.zeros_like(x)

        top_k_indices, top_k_values = self.gating(features)

        for expert_idx in range(self.num_experts):
            mask = (top_k_indices == expert_idx)
            weights = top_k_values * mask
            expert_output = self.experts[expert_idx](features)
            output += expert_output * weights.sum(dim=1).view(-1, 1, 1, 1)

        if self.training and not self.is_finetune:
            loss_gate = self.compute_balance_loss(top_k_values, top_k_indices)
            self.gating.update_temperature()
        else:
            loss_gate = 0

        return shared_output + output, loss_gate

    def compute_balance_loss(
        self,
        gates: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the balance loss to encourage equal expert usage.
        Args:
            gates (torch.Tensor): Gating weights of shape (B, top_k)
            indices (torch.Tensor): Indices of selected experts of shape (B, top_k)
        Returns:
            torch.Tensor: Balance loss scalar
        """
        importance = torch.zeros(self.num_experts, device=gates.device)
        for i in range(self.num_experts):
            mask = (indices == i)
            importance[i] = (gates * mask).sum()

        ideal_load = gates.sum() / self.num_experts
        balance_loss = torch.pow(importance - ideal_load, 2).mean()

        return balance_loss
