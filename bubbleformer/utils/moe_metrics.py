import torch
from typing import List

def routing_percentage(tokens_per_expert: torch.Tensor):
    assert tokens_per_expert.dim() == 1, "Tokens per expert must be of shape (num_experts,)"
    assert tokens_per_expert.sum() > 0
    return tokens_per_expert / tokens_per_expert.sum()

def topk_indices_to_patch_expert_counts(
    topk_indices: torch.Tensor,
    num_experts: int
):
    batched = True
    if topk_indices.dim() == 4:
        batched = False
        topk_indices = topk_indices.unsqueeze(0)
    
    assert topk_indices.dim() == 5, "Topk indices must be of shape (B, T, H, W, topk)"
    assert num_experts > 0
    
    B, T, H, W, _ = topk_indices.shape
    expert_counts = torch.zeros(num_experts, B, T, H, W, device=topk_indices.device, dtype=torch.int32)
    for e in range(num_experts):
        expert_counts[e] = (topk_indices == e).sum(dim=-1).to(torch.int32) # (B, T, H, W)
    
    if not batched:
        expert_counts = expert_counts.squeeze(1) # (num_experts, T, H, W)
    return expert_counts