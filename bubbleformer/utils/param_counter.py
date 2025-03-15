import torch
import torch.nn as nn
from bubbleformer.layers.moe import MoE

def count_parameters(model):
    """
    Count total parameters and activated parameters in a model with MoE layers
    
    Args:
        model (nn.Module): The model to analyze
        
    Returns:
        tuple: (total_params, activated_params, moe_stats)
            - total_params: Total number of parameters in the model
            - activated_params: Number of parameters activated in a forward pass
            - moe_stats: Dictionary with MoE-specific statistics
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count parameters that are always active (non-MoE)
    always_active_params = 0
    moe_params = 0
    moe_layers = []
    
    # Find all MoE layers
    for name, module in model.named_modules():
        if isinstance(module, MoE):
            moe_layers.append((name, module))
            # Count parameters in MoE layers
            expert_params = sum(p.numel() for expert in module.experts for p in expert.parameters())
            gate_params = sum(p.numel() for p in module.gate.parameters())
            shared_expert_params = sum(p.numel() for p in module.shared_experts.parameters())
            
            moe_params += expert_params + gate_params + shared_expert_params
        elif isinstance(module, nn.Parameter):
            always_active_params += module.numel()
    
    # Parameters that are not in MoE layers
    non_moe_params = total_params - moe_params
    
    # For each MoE layer, calculate activated parameters
    activated_moe_params = 0
    moe_stats = {
        "total_moe_layers": len(moe_layers),
        "layers": []
    }
    
    for name, moe_layer in moe_layers:
        n_experts = moe_layer.n_routed_experts
        top_k = moe_layer.top_k
        
        # Parameters per expert
        params_per_expert = sum(p.numel() for expert in moe_layer.experts for p in expert.parameters()) / n_experts
        
        # Gate parameters (always active)
        gate_params = sum(p.numel() for p in moe_layer.gate.parameters())
        
        # Shared expert parameters (always active)
        shared_expert_params = sum(p.numel() for p in moe_layer.shared_experts.parameters())
        
        # Activated parameters for this layer
        # Each token activates top_k experts out of n_experts
        activation_ratio = top_k / n_experts
        activated_expert_params = params_per_expert * top_k
        
        # Total activated parameters for this layer
        layer_activated_params = activated_expert_params + gate_params + shared_expert_params
        activated_moe_params += layer_activated_params
        
        layer_stats = {
            "name": name,
            "n_experts": n_experts,
            "top_k": top_k,
            "params_per_expert": params_per_expert,
            "gate_params": gate_params,
            "shared_expert_params": shared_expert_params,
            "activation_ratio": activation_ratio,
            "activated_params": layer_activated_params
        }
        moe_stats["layers"].append(layer_stats)
    
    # Total activated parameters
    activated_params = non_moe_params + activated_moe_params
    
    return total_params, activated_params, moe_stats

def print_model_stats(model):
    """
    Print detailed statistics about model parameters
    
    Args:
        model (nn.Module): The model to analyze
    """
    total_params, activated_params, moe_stats = count_parameters(model)
    
    print("\n" + "="*80)
    print(f"MODEL PARAMETER STATISTICS")
    print("="*80)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Activated Parameters: {activated_params:,}")
    print(f"Activation Ratio: {activated_params/total_params:.2%}")
    
    if moe_stats["total_moe_layers"] > 0:
        print("\nMixture of Experts (MoE) Layers:")
        print("-"*80)
        
        for i, layer in enumerate(moe_stats["layers"]):
            print(f"\nLayer {i+1}: {layer['name']}")
            print(f"  Experts: {layer['n_experts']}, Top-k: {layer['top_k']}")
            print(f"  Parameters per expert: {layer['params_per_expert']:,}")
            print(f"  Gate parameters: {layer['gate_params']:,}")
            print(f"  Shared expert parameters: {layer['shared_expert_params']:,}")
            print(f"  Expert activation ratio: {layer['activation_ratio']:.2%}")
            print(f"  Activated parameters: {layer['activated_params']:,}")
    
    print("\n" + "="*80) 