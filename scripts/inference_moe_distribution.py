#!/usr/bin/env python3
"""
BubbleFormer MoE Distribution Analysis Script

This script extends inference_autoregressive.py to track and visualize the token distribution
across experts in MoE layers for different datasets.

Usage examples:
    # Run analysis on all datasets (default)
    python inference_moe_distribution.py
    
    # Run on specific dataset
    python inference_moe_distribution.py --dataset sat_92
    
    # Run with custom weights path
    python inference_moe_distribution.py --weights_path /path/to/weights.ckpt
    
    # Save distribution plots
    python inference_moe_distribution.py --save_plots
"""

import os
import torch
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from bubbleformer.models import get_model
from bubbleformer.data import BubblemlForecast
from bubbleformer.layers.moe import MoETracker
from bubbleformer.utils.losses import LpLoss

def generate_save_dir_from_weights_path(weights_path):
    """Generate a save directory base path from a weights path based on the observed pattern"""
    import re
    
    # Extract the epoch number from the checkpoint filename
    match = re.search(r'epoch=(\d+)', os.path.basename(weights_path))
    if not match:
        # If no epoch number found, use a default name
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(weights_path))), "moe_analysis")
    
    epoch_num = match.group(1)
    
    # Extract the base experiment directory (everything before lightning_logs)
    base_dir = weights_path.split('/lightning_logs')[0]
    
    # Construct the save directory base
    save_dir = os.path.join(base_dir, f"epoch_{epoch_num}_moe_analysis")
    
    return save_dir

def run_moe_analysis(dataset_names, custom_weights_path=None, custom_save_dir=None, save_plots=False, test_mode=False):
    """Run MoE distribution analysis on specified datasets"""
    print(f"\n=== Running MoE distribution analysis on {', '.join(dataset_names)} ===\n")
    
    # Configure paths based on the datasets
    dataset_paths = {
        'sat_92': ["/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Saturated-FC72-2D-0.1/Twall-92.hdf5"],
        'subcooled_100': ["/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-SubCooled-FC72-2D-0.1/Twall-100.hdf5"],
        'grav_0.2': ["/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Gravity-FC72-2D-0.1/gravY-0.2.hdf5"]
    }
    
    # We use avit_moe for this analysis
    model_type = 'avit_moe'
    model_kwargs = {
        "fields": 4,
        "patch_size": 16,
        "embed_dim": 384,
        "processor_blocks": 12,
        "num_heads": 6,
        "drop_path": 0.2,
        "n_experts": 6,
        "n_shared_experts": 1,
        "top_k": 2,
        "routed_expert_embed_dim": 48,
        "shared_expert_type": "gelu",
        "shared_expert_embed_dim": 96
    }
    
    # Default weights path for avit_moe model
    default_weights_path = "/share/crsp/lab/amowli/xianwz2/bubbleformer_modify/bubbleformer/bubbleformer_logs/Same_size_shared_experts_E_6_S_1_A_2_avit_moe_poolboiling_combined_36342187/lightning_logs/version_1/checkpoints/epoch=399-step=200000.ckpt"
    
    # Use custom paths if provided, otherwise use defaults
    weights_path = custom_weights_path if custom_weights_path else default_weights_path
    
    # Determine save directory
    if custom_save_dir:
        save_dir_base = custom_save_dir
    elif custom_weights_path:
        save_dir_base = generate_save_dir_from_weights_path(custom_weights_path)
    else:
        save_dir_base = os.path.dirname(os.path.dirname(default_weights_path)) + "/moe_analysis"
    
    print(f"Using weights from: {weights_path}")
    print(f"Saving results to: {save_dir_base}")
    
    # Load model
    model = get_model(model_type, **model_kwargs)
    
    # Load weights
    model_data = torch.load(weights_path, weights_only=False)
    
    diff_term, div_term = model_data['hyper_parameters']['normalization_constants']
    diff_term = torch.tensor(diff_term)
    div_term = torch.tensor(div_term)
    
    weight_state_dict = OrderedDict()
    for key, val in model_data["state_dict"].items():
        name = key[6:]
        weight_state_dict[name] = val
    del model_data
    
    model.load_state_dict(weight_state_dict, strict=False)
    model.eval()
    
    # Create MoE tracker
    tracker = MoETracker(model)
    
    # Store token distributions by dataset
    dataset_distributions = {}
    
    # Process each dataset
    for dataset_name in dataset_names:
        print(f"\nAnalyzing dataset: {dataset_name}")
        
        # Create dataset
        test_path = dataset_paths[dataset_name]
        test_dataset = BubblemlForecast(
            filenames=test_path,
            fields=["dfun", "temperature", "velx", "vely"],
            norm="none",
            time_window=5,
            start_time=95
        )
        
        # Normalize dataset
        _ = test_dataset.normalize(diff_term, div_term)
        
        # Enable tracking
        tracker.enable_tracking()
        tracker.reset_counts()
        
        # Run iterations for token distribution analysis
        start_time = test_dataset.start_time
        skip_itrs = test_dataset.time_window
        
        # Set the number of iterations based on test_mode
        max_itr = 50 if test_mode else 500
        print(f"Running in {'test mode' if test_mode else 'full mode'} with {max_itr} max iterations")
        
        for itr in range(0, max_itr, skip_itrs):
            inp, _ = test_dataset[itr]
            inp = inp.float().unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                _ = model(inp)
            
            print(f"Processed iteration {itr}/{max_itr}")
        
        # Get token distributions
        layer_counts = tracker.get_layer_counts()
        total_counts = tracker.get_total_counts()
        
        # Store distributions for this dataset
        dataset_distributions[dataset_name] = {
            'layer_counts': [count.cpu().numpy() for count in layer_counts],
            'total_counts': total_counts.cpu().numpy()
        }
        
        # Disable tracking
        tracker.disable_tracking()
    
    # Create directory for results
    os.makedirs(save_dir_base, exist_ok=True)
    
    # Save raw token count data
    torch.save(dataset_distributions, os.path.join(save_dir_base, "moe_distributions.pt"))
    
    # Plot distribution for each layer across datasets
    if save_plots:
        num_moe_layers = len(dataset_distributions[dataset_names[0]]['layer_counts'])
        
        # For each MoE layer
        for layer_idx in range(num_moe_layers):
            plt.figure(figsize=(12, 6))
            
            # Get number of experts
            num_experts = len(dataset_distributions[dataset_names[0]]['layer_counts'][layer_idx])
            
            # Setup bar positions
            bar_width = 0.8 / len(dataset_names)
            r = np.arange(num_experts)
            
            # Plot for each dataset
            for i, dataset_name in enumerate(dataset_names):
                counts = dataset_distributions[dataset_name]['layer_counts'][layer_idx]
                plt.bar(r + i * bar_width, counts, width=bar_width, label=dataset_name)
            
            plt.xlabel('Expert ID')
            plt.ylabel('Token Count')
            plt.title(f'MoE Layer {layer_idx} - Token Distribution by Dataset')
            plt.xticks(r + bar_width * (len(dataset_names) - 1) / 2, range(num_experts))
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(save_dir_base, f"layer_{layer_idx}_distribution.png"))
            plt.close()
        
        # Plot total distribution across datasets
        plt.figure(figsize=(12, 6))
        
        # Get number of experts
        num_experts = len(dataset_distributions[dataset_names[0]]['total_counts'])
        
        # Setup bar positions
        bar_width = 0.8 / len(dataset_names)
        r = np.arange(num_experts)
        
        # Plot for each dataset
        for i, dataset_name in enumerate(dataset_names):
            counts = dataset_distributions[dataset_name]['total_counts']
            plt.bar(r + i * bar_width, counts, width=bar_width, label=dataset_name)
        
        plt.xlabel('Expert ID')
        plt.ylabel('Token Count')
        plt.title('Total MoE Token Distribution by Dataset')
        plt.xticks(r + bar_width * (len(dataset_names) - 1) / 2, range(num_experts))
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(save_dir_base, "total_distribution.png"))
        plt.close()
    
    # Create a comparative analysis of expert specialization
    plt.figure(figsize=(14, 8))
    
    # Calculate the proportion of tokens for each expert by dataset
    expert_proportions = {}
    for dataset_name in dataset_names:
        total_counts = dataset_distributions[dataset_name]['total_counts']
        normalized_counts = total_counts / total_counts.sum()
        expert_proportions[dataset_name] = normalized_counts
    
    # Plot heatmap of expert specialization
    data = np.array([expert_proportions[dataset_name] for dataset_name in dataset_names])
    plt.imshow(data, aspect='auto', cmap='viridis')
    plt.colorbar(label='Proportion of Tokens')
    plt.xlabel('Expert ID')
    plt.ylabel('Dataset')
    plt.yticks(range(len(dataset_names)), dataset_names)
    plt.xticks(range(len(expert_proportions[dataset_names[0]])), range(len(expert_proportions[dataset_names[0]])))
    plt.title('Expert Specialization by Dataset')
    plt.tight_layout()
    
    # Save specialization plot
    if save_plots:
        plt.savefig(os.path.join(save_dir_base, "expert_specialization.png"))
    
    print(f"\nMoE distribution analysis completed")
    print(f"Results saved to {save_dir_base}")
    
    return dataset_distributions

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze MoE token distribution across datasets')
    parser.add_argument('--dataset', type=str, choices=['sat_92', 'subcooled_100', 'grav_0.2', 'all'], default='all',
                        help='Dataset to analyze (default: all)')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Custom path to model weights checkpoint file (optional)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Custom base directory to save results (optional)')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save distribution plots (default: False)')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with fewer iterations (default: False)')
    args = parser.parse_args()
    
    # Determine which datasets to analyze
    datasets_to_run = []
    if args.dataset == 'all':
        datasets_to_run = ['sat_92', 'subcooled_100', 'grav_0.2']
    else:
        datasets_to_run = [args.dataset]
    
    # Run analysis
    run_moe_analysis(datasets_to_run, args.weights_path, args.save_dir, args.save_plots, args.test)

if __name__ == "__main__":
    main() 