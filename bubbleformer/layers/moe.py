import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .linear_layers import GeluMLP, SirenMLP
from einops import rearrange
import torchvision.utils as vutils



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
        # x shape: (batch_size * time_window * height * width, hidden_dim)
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
        shared_expert_inter_dim=None,
        layer_idx=None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.top_k = top_k
        self.gate = Gate(hidden_dim, n_routed_experts, top_k)
        self.experts = nn.ModuleList([Expert(hidden_dim, routed_expert_inter_dim) for _ in range(n_routed_experts)])
        
        # If shared_expert_inter_dim is not provided, use the same as routed_expert_inter_dim
        if shared_expert_inter_dim is None:
            shared_expert_inter_dim = routed_expert_inter_dim
        
        # Create shared experts based on n_shared_experts parameter
        self.shared_experts = nn.ModuleList()
        for _ in range(n_shared_experts):
            if shared_expert_type.lower() == "siren":
                self.shared_experts.append(SirenMLP(hidden_dim, inter_dim=shared_expert_inter_dim))
            else:  # default to GeluMLP
                self.shared_experts.append(GeluMLP(hidden_dim, inter_dim=shared_expert_inter_dim))
    
    def forward(self, x):
        # Extract dimensions from input
        batch_size, time_window, height, width, hidden_dim = x.shape
        
        # Reshape input for gate and experts
        x_flat = rearrange(x, "b t h w d -> (b t h w) d") # X Shape: (batch_size * time_window * height * width, hidden_dim))
        
        # Get weights and indices from gate
        weights, indices = self.gate(x_flat)  # Weights Shape: (batch_size * height * width, topk)
        
        # Process through experts normally
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        output = torch.zeros_like(x_flat)  # Placeholder for MoE output
        
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            output[idx] += expert(x_flat[idx]) * weights[idx, top, None]
        
        # Add shared expert output if there are any shared experts
        if self.n_shared_experts > 0:
            shared_expert_output = torch.zeros_like(x_flat)
            for shared_expert in self.shared_experts:
                shared_expert_output += shared_expert(x_flat)
            
            # If there are multiple shared experts, average their outputs
            if self.n_shared_experts > 1:
                shared_expert_output = shared_expert_output / self.n_shared_experts
                
            # Combined output from both routed and shared experts
            combined_output = output + shared_expert_output
        else:
            # If no shared experts, only use routed experts output
            combined_output = output
        
        # Reshape back to original form
        return combined_output.view(batch_size, time_window, height, width, hidden_dim)