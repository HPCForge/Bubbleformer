import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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
    def __init__(self, hidden_dim, inter_dim, n_routed_experts=4, n_shared_experts=1, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_routed_experts = n_routed_experts
        self.top_k = top_k
        self.gate = Gate(hidden_dim, n_routed_experts, top_k)
        self.experts = nn.ModuleList([Expert(hidden_dim, inter_dim) for _ in range(n_routed_experts)])
        # self.shared_experts = MLP(hidden_dim, n_shared_experts * inter_dim)
        self.shared_experts = MLP(hidden_dim, inter_dim)
        
    def forward(self, x):
        batch_size, height, width, channel = x.shape # X Shape: (batch_size, height, width, hidden_dim)
        x = x.contiguous().reshape(-1, channel)  # X Shape: (batch_size * height * width, hidden_dim))
        
        weights, indices = self.gate(x)  # Weights Shape: (batch_size * height * width, topk)
        
        output = torch.zeros_like(x)  # Placeholder for MoE output
        # Count the occurrences of each index in the indices tensor and then converting the result to a list
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist() # Shape: (n_routed_experts, )

        for i in range(self.n_routed_experts):
            if counts[i] ==  0:
                continue
            expert = self.experts[i]
            # torch.where returns two tensors, which represent the row and column indices, respectively
            # where the condition indices == i is true
            idx, top = torch.where(indices == i)
            output[idx] += expert(x[idx]) * weights[idx, top, None]
        shared_expert_output = self.shared_experts(x)
        return (output + shared_expert_output).view(batch_size, height, width, channel)  # Reshape back to original form