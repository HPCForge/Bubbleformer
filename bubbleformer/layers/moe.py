import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
from .linear_layers import GeluMLP, SirenMLP

# # SwiGLU MLP
# class MLP(nn.Module):
#     def __init__(self, dim: int, inter_dim: int):
#         super().__init__()
#         self.w1 = nn.Linear(dim, inter_dim)
#         self.w2 = nn.Linear(inter_dim, dim)
#         self.w3 = nn.Linear(dim, inter_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.w2(F.silu(self.w1(x)) * self.w3(x))

# # GLU MLP Original
# class MLP(nn.Module):
#     def __init__(self, hidden_dim, exp_factor=4.):
#         super().__init__()
#         self.fc1 = nn.Linear(hidden_dim, int(hidden_dim * exp_factor))
#         self.fc2 = nn.Linear(int(hidden_dim * exp_factor), hidden_dim)
#         self.act = nn.GELU()
        
#     def forward(self, x):
#         return self.fc2(self.act(self.fc1(x)))

# GELU MLP
# class MLP(nn.Module):
#     def __init__(self, hidden_dim, inter_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(hidden_dim, inter_dim)
#         self.fc2 = nn.Linear(inter_dim, hidden_dim)
#         self.act = nn.GELU()
        
#     def forward(self, x):
#         return self.fc2(self.act(self.fc1(x)))

# GeGLU MLP
class MLP(nn.Module):
    def __init__(self, hidden_dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, inter_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)) * self.w3(x)) # (batch_size * seq_len, dim)
    
# # GELU activation function
class Expert(nn.Module):
    def __init__(self, hidden_dim, inter_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, inter_dim)
        self.fc2 = nn.Linear(inter_dim, hidden_dim)
        self.act = nn.GELU()
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))  # Keep same shape as input

# # SwiGLU activation function
# class Expert(nn.Module):
#     def __init__(self, hidden_dim: int, inter_dim: int):
#         super().__init__()
#         self.w1 = nn.Linear(hidden_dim, inter_dim)
#         self.w2 = nn.Linear(inter_dim, hidden_dim)
#         self.w3 = nn.Linear(hidden_dim, inter_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.w2(F.silu(self.w1(x)) * self.w3(x)) # (batch_size * seq_len, dim)

# # GeGLU activation function
# class Expert(nn.Module):
#     def __init__(self, hidden_dim: int, inter_dim: int):
#         super().__init__()
#         self.w1 = nn.Linear(hidden_dim, inter_dim)
#         self.w2 = nn.Linear(inter_dim, hidden_dim)
#         self.w3 = nn.Linear(hidden_dim, inter_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.w2(F.gelu(self.w1(x)) * self.w3(x)) # (batch_size * seq_len, dim)

class Gate(nn.Module):
    def __init__(self, hidden_dim, n_routed_experts, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_routed_experts = n_routed_experts
        self.top_k = top_k
        self.weight = nn.Parameter(torch.randn(n_routed_experts, hidden_dim))
    
    def forward(self, x):
        # x shape: (batch_size * height * width, hidden_dim)
        scores = F.softmax(F.linear(x, self.weight), dim=-1)
        top_k_values, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        top_k_values /= top_k_values.sum(dim=-1, keepdim=True)  # Normalize weights
        return top_k_values, top_k_indices  # (batch_size * height * width, top_k), (batch_size * height * width, top_k)

class MoE(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        routed_expert_inter_dim, 
        n_routed_experts=4, 
        n_shared_experts=1, 
        top_k=2, 
        shared_expert_type="gelu",
        shared_expert_inter_dim=None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_routed_experts = n_routed_experts
        self.top_k = top_k
        self.gate = Gate(hidden_dim, n_routed_experts, top_k)
        self.experts = nn.ModuleList([Expert(hidden_dim, routed_expert_inter_dim) for _ in range(n_routed_experts)])
        
        # If shared_expert_inter_dim is not provided, use the same as routed_expert_inter_dim
        if shared_expert_inter_dim is None:
            shared_expert_inter_dim = routed_expert_inter_dim
        
        # Choose shared expert type based on parameter
        if shared_expert_type.lower() == "siren":
            self.shared_experts = SirenMLP(hidden_dim, inter_dim=shared_expert_inter_dim)
        else:  # default to GeluMLP
            self.shared_experts = GeluMLP(hidden_dim, inter_dim=shared_expert_inter_dim)
        
        # Add tracking variables
        self.tracking_enabled = False
        self.expert_token_counts = None
        
    def enable_tracking(self):
        """Enable tracking of token distribution across experts"""
        self.tracking_enabled = True
        self.expert_token_counts = torch.zeros(self.n_routed_experts)
        
    def disable_tracking(self):
        """Disable tracking of token distribution"""
        self.tracking_enabled = False
        
    def get_token_counts(self):
        """Return the current token counts per expert"""
        if self.expert_token_counts is None:
            return None
        return self.expert_token_counts.clone()
        
    def reset_token_counts(self):
        """Reset token counts to zero"""
        if self.expert_token_counts is not None:
            self.expert_token_counts.zero_()
        
    def forward(self, x):
        batch_size, height, width, hidden_dim = x.shape # X Shape: (batch_size, height, width, hidden_dim)
        
        # if self.training:
        #     print("TRAINING - x shape: ", x.shape) # torch.Size([20, 32, 32, 384])
        # else:
        #     print("VALIDATION - x shape: ", x.shape) # torch.Size([20, 32, 32, 384])
        
        x = x.contiguous().reshape(-1, hidden_dim)  # X Shape: (batch_size * height * width, hidden_dim))
        
        weights, indices = self.gate(x)  # Weights Shape: (batch_size * height * width, topk)
        
        output = torch.zeros_like(x)  # Placeholder for MoE output
        # Count the occurrences of each index in the indices tensor and then converting the result to a list
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist() # Shape: (n_routed_experts, )

        # Track token distribution if enabled
        # Number of tokens = batch_size × time_window × height × width
        # Number of tokens = 4 × 5 × 32 × 32 = 20480
        if self.tracking_enabled and self.expert_token_counts is not None:
            token_counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
            self.expert_token_counts += token_counts.to(self.expert_token_counts.device)

        for i in range(self.n_routed_experts):
            if counts[i] ==  0:
                continue
            expert = self.experts[i]
            # torch.where returns two tensors, which represent the row and column indices, respectively
            # where the condition indices == i is true
            idx, top = torch.where(indices == i)
            output[idx] += expert(x[idx]) * weights[idx, top, None]
        shared_expert_output = self.shared_experts(x)
        return (output + shared_expert_output).view(batch_size, height, width, hidden_dim)  # Reshape back to original form

class MoETracker:
    """
    Utility class to track token distribution across experts in MoE layers
    """
    def __init__(self, model):
        self.model = model
        self.moe_layers = []
        self.layer_names = []
        self._find_moe_layers(model)
        self.is_tracking = False
        
    def _find_moe_layers(self, module, prefix=""):
        """Recursively find all MoE layers in the model"""
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, MoE):
                self.moe_layers.append(child)
                self.layer_names.append(full_name)
            else:
                self._find_moe_layers(child, full_name)
    
    def enable_tracking(self):
        """Enable tracking on all MoE layers"""
        for layer in self.moe_layers:
            layer.enable_tracking()
        self.is_tracking = True
        
    def disable_tracking(self):
        """Disable tracking on all MoE layers"""
        for layer in self.moe_layers:
            layer.disable_tracking()
        self.is_tracking = False
        
    def reset_counts(self):
        """Reset token counts on all MoE layers"""
        for layer in self.moe_layers:
            layer.reset_token_counts()
    
    def get_layer_counts(self):
        """Get token counts for each layer"""
        if not self.is_tracking:
            return None
            
        counts = []
        for layer in self.moe_layers:
            counts.append(layer.get_token_counts())
        return counts
    
    def get_total_counts(self):
        """Get total token counts across all layers"""
        if not self.is_tracking:
            return None
            
        total = None
        for layer in self.moe_layers:
            counts = layer.get_token_counts()
            if counts is not None:
                if total is None:
                    total = counts.clone()
                else:
                    total += counts
        return total
    
    def plot_distribution(self, log_wandb=False, epoch=None):
        """Plot token distribution across experts for each layer and in total"""
        if not self.is_tracking:
            return None
            
        layer_counts = self.get_layer_counts()
        total_counts = self.get_total_counts()
        
        if layer_counts is None or total_counts is None:
            return None
            
        # Create a figure with subplots for each layer plus the total
        n_plots = len(layer_counts) + 1
        fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 4, 5))
        
        # If there's only one subplot, axes is not a list
        if n_plots == 2:
            axes = [axes[0], axes[1]]
        
        # Plot individual layer distributions
        for i, (counts, name) in enumerate(zip(layer_counts, self.layer_names)):
            counts_np = counts.cpu().numpy()
            expert_ids = np.arange(len(counts_np))
            
            axes[i].bar(expert_ids, counts_np)
            axes[i].set_title(f"Epoch {epoch} - Layer: {name}")
            axes[i].set_xlabel("Expert ID")
            axes[i].set_ylabel("Token Count")
            # Ensure integer ticks
            axes[i].set_xticks(expert_ids)

        # Plot total distribution
        total_np = total_counts.cpu().numpy()
        expert_ids = np.arange(len(total_np))

        axes[-1].bar(expert_ids, total_np)
        axes[-1].set_title(f"Epoch {epoch} - Total Across All Layers")
        axes[-1].set_xlabel("Expert ID")
        axes[-1].set_ylabel("Token Count")
        axes[-1].set_xticks(expert_ids)
        
        plt.tight_layout()
        
        if log_wandb:
            # Create a log dictionary with the epoch if provided
            log_data = {"Expert Token Distribution": wandb.Image(fig)}
            if epoch is not None:
                log_data["epoch"] = epoch
            wandb.log(log_data)
        
        return fig
    
    def log_distribution_data(self, log_wandb=False, epoch=None):
        """Log token distribution data numerically to wandb"""
        if not self.is_tracking or not log_wandb:
            return None
        
        layer_counts = self.get_layer_counts()
        total_counts = self.get_total_counts()
        
        if layer_counts is None or total_counts is None:
            return None
        
        # Create a dictionary to hold all the data
        log_data = {}
        
        # Log individual layer distributions
        for i, (counts, name) in enumerate(zip(layer_counts, self.layer_names)):
            counts_np = counts.cpu().numpy()
            for expert_id, count in enumerate(counts_np):
                log_data[f"expert_counts/{name}/expert_{expert_id}"] = count
        
        # Log total distribution
        total_np = total_counts.cpu().numpy()
        for expert_id, count in enumerate(total_np):
            log_data[f"expert_counts/total/expert_{expert_id}"] = count
        
        # Add epoch if provided
        if epoch is not None:
            log_data["epoch"] = epoch
        
        # Log all data at once
        wandb.log(log_data)

# Example usage during inference
def inference_with_tracking(model, input_data):
    # Create tracker
    tracker = MoETracker(model)
    
    # Enable tracking
    tracker.enable_tracking()
    tracker.reset_counts()
    
    # Run inference
    with torch.no_grad():
        output = model(input_data)
    
    # Get and visualize results
    fig = tracker.plot_distribution()
    
    # Disable tracking
    tracker.disable_tracking()
    
    return output, fig