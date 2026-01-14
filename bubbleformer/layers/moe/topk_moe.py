import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class TopkMoEOutput:
    tokens: torch.Tensor
    router_logits: torch.Tensor
    counts: torch.Tensor
    load_balance_loss: torch.Tensor

def load_balance_loss(
    router_logits: torch.Tensor,
    expert_counts: torch.Tensor,
    topk: int,
    num_experts: int
):
    r"""
    Switch-transformer load balance loss.
    This computes the dot product of two vectors:
        1. the percentage of tokens routed to each expert.
        2. the average routing probability of each expert.
    Args:
        router_logits: (num_tokens, num_experts)
        expert_counts: (num_experts,)
        num_tokens: int
    """
    mean_tokens_per_expert = expert_counts.to(torch.float32) / expert_counts.to(torch.float32).sum()
    router_probs = F.softmax(router_logits, dim=-1)
    mean_router_prob_per_expert = router_probs.mean(dim=0)
    loss = torch.dot(mean_tokens_per_expert, mean_router_prob_per_expert) * num_experts
    return loss

def get_token_indices(
    topk_expert_indices: torch.Tensor, 
    num_experts: int
):
    # The routing choices are not used in the backward pass, so this
    # can always use no_grad.
    with torch.no_grad():
        flat_expert_indices = topk_expert_indices.view(-1)
        # The argsort returns int64 indices, but we don't need more than 32 bits for the indices.
        indices = flat_expert_indices.argsort().to(torch.int32)
        counts = torch.histc(flat_expert_indices, min=0, max=num_experts - 1, bins=num_experts)
        # group indices initialized like this to use int32.
        group_indices = torch.empty(counts.size(0), dtype=torch.int32, device=topk_expert_indices.device)
        torch.cumsum(counts, dim=0, out=group_indices)
        return group_indices, indices, counts

class TopkMoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        topk: int,
        load_balance_loss_weight: float
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.topk = topk
        self.load_balance_loss_weight = load_balance_loss_weight
        
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, intermediate_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, intermediate_dim, hidden_dim))
        self.router = nn.Linear(hidden_dim, num_experts)
        self.reset_parameters()
        
    def reset_parameters(self):
        with torch.no_grad():
            self.w1.normal_(0, 0.02)
            self.w2.normal_(0, 0.02)
            self.router.weight.data.normal_(0, 0.02)

    def forward(self, x):
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])
        batch_size = x.shape[0]
        
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, k=self.topk, dim=-1)
        group_indices, indices, counts = get_token_indices(topk_indices, self.num_experts)
        
        # Map tokens into groups based on their assigned experts
        groups = x[indices // self.topk]
        
        # Compute all of the experts
        groups = torch._grouped_mm(groups, self.w1, group_indices)
        groups = F.gelu(groups)
        groups = torch._grouped_mm(groups, self.w2, group_indices)
        
        # Scatter the tokens to [B, topk, hidden_dim]
        scattered = torch.empty_like(groups)
        scattered[indices] = groups
        scattered = scattered.view(batch_size, self.topk, self.hidden_dim)
        
        # reduce the output tokens and scale by the routing probability
        out = (scattered * topk_probs.unsqueeze(-1)).sum(dim=1).view(input_shape)
        
        loss = load_balance_loss(router_logits, counts, self.topk, self.num_experts) * self.load_balance_loss_weight
        
        return TopkMoEOutput(out, router_logits, counts, loss)