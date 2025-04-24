#!/usr/bin/env python3
"""
BubbleFormer Inference Script

This script runs inference with BubbleFormer models (avit and avit_moe) on different datasets.
It replaces the need for manually commenting/uncommenting code in Jupyter notebooks.

Usage examples:
    # Run both models on all datasets (default)
    python inference_autoregressive.py
    
    # Run avit model on sat_92 dataset
    python inference_autoregressive.py --model avit --dataset sat_92
    
    # Run avit_moe model on all datasets
    python inference_autoregressive.py --model avit_moe --dataset all
    
    # Run both models on grav_0.2 dataset
    python inference_autoregressive.py --model both --dataset grav_0.2
    
    # Run with custom weights path (save directory will be auto-generated)
    python inference_autoregressive.py --model avit_moe --dataset sat_92 --weights_path /path/to/weights.ckpt
    
    # Run with both custom weights path and save directory
    python inference_autoregressive.py --model avit_moe --dataset sat_92 --weights_path /path/to/weights.ckpt --save_dir /path/to/save
"""

import os
import torch
import argparse
import time
from collections import OrderedDict
from bubbleformer.models import get_model
from bubbleformer.data import BubblemlForecast
from bubbleformer.utils.losses import LpLoss
from bubbleformer.utils.plot_utils import plot_bubbleml

def generate_save_dir_from_weights_path(weights_path):
    """Generate a save directory base path from a weights path based on the observed pattern"""
    import re
    
    # Extract the epoch number from the checkpoint filename
    match = re.search(r'epoch=(\d+)', os.path.basename(weights_path))
    if not match:
        # If no epoch number found, use a default name
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(weights_path))), "outputs")
    
    epoch_num = match.group(1)
    
    # Extract the base experiment directory (everything before lightning_logs)
    base_dir = weights_path.split('/lightning_logs')[0]
    
    # Construct the save directory base
    save_dir = os.path.join(base_dir, f"epoch_{epoch_num}_outputs")
    
    return save_dir

def run_inference(model_type, dataset_name, custom_weights_path=None, custom_save_dir=None):
    """Run inference with specified model on the specified dataset"""
    print(f"\n=== Running inference with {model_type} on {dataset_name} ===\n")
    
    # Configure paths based on the dataset
    dataset_paths = {
        'sat_92': ["/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Saturated-FC72-2D-0.1/Twall-92.hdf5"],
        'subcooled_100': ["/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-SubCooled-FC72-2D-0.1/Twall-100.hdf5"],
        'grav_0.2': ["/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Gravity-FC72-2D-0.1/gravY-0.2.hdf5"]
    }
    
    # Configure model parameters and paths based on the model type
    if model_type == 'avit_moe':
        # model_kwargs = {
        #     "fields": 4,
        #     "patch_size": 16,
        #     "embed_dim": 384,
        #     "processor_blocks": 12,
        #     "num_heads": 6,
        #     "drop_path": 0.2,
        #     "n_experts": 6,
        #     "n_shared_experts": 1,
        #     "top_k": 2
        # }
        model_kwargs = {
            "fields": 4,
            "patch_size": 16,
            "embed_dim": 384,
            "processor_blocks": 12,
            "num_heads": 6,
            "drop_path": 0.2,
            "n_experts": 24,
            "n_shared_experts": 1,
            "top_k": 2,
            "routed_expert_embed_dim": 12,
            "shared_expert_type": "gelu",  # Options: "gelu" or "siren"
            "shared_expert_embed_dim": 96  # Larger dimension for shared expert
        }
        # Default paths for avit_moe model
        default_weights_path = "/share/crsp/lab/amowli/xianwz2/bubbleformer_modify/bubbleformer/bubbleformer_logs/Same_size_shared_experts_E_6_S_1_A_2_avit_moe_poolboiling_combined_36342187/lightning_logs/version_1/checkpoints/epoch=399-step=200000.ckpt"
        default_save_dir_base = "/share/crsp/lab/amowli/xianwz2/bubbleformer_modify/bubbleformer/bubbleformer_logs/Same_size_shared_experts_E_6_S_1_A_2_avit_moe_poolboiling_combined_36342187/epoch_399_outputs"
    else:  # avit
        model_kwargs = {
            "fields": 4,
            "patch_size": 16,
            "embed_dim": 384,
            "processor_blocks": 12,
            "num_heads": 6,
            "drop_path": 0.2
        }
        # Default paths for avit model
        default_weights_path = "/share/crsp/lab/amowli/xianwz2/bubbleformer_modify/bubbleformer/bubbleformer_logs/Same_size_shared_experts_E_6_S_1_A_2_avit_moe_poolboiling_combined_36342187/lightning_logs/version_0/checkpoints/epoch=341-step=171000.ckpt"
        default_save_dir_base = "/share/crsp/lab/amowli/xianwz2/bubbleformer_modify/bubbleformer/bubbleformer_logs/Same_size_shared_experts_E_6_S_1_A_2_avit_moe_poolboiling_combined_36342187/epoch_341_outputs"
    
    # Use custom paths if provided, otherwise use defaults
    weights_path = custom_weights_path if custom_weights_path else default_weights_path
    
    # Determine save directory: 
    # 1. Use custom_save_dir if provided
    # 2. If not provided but custom_weights_path is provided, auto-generate save_dir from weights_path
    # 3. Otherwise use default_save_dir_base
    if custom_save_dir:
        save_dir_base = custom_save_dir
    elif custom_weights_path:
        save_dir_base = generate_save_dir_from_weights_path(custom_weights_path)
    else:
        save_dir_base = default_save_dir_base
    
    # Set save directory based on dataset
    save_dir = os.path.join(save_dir_base, dataset_name)
    
    print(f"Using weights from: {weights_path}")
    print(f"Saving results to: {save_dir}")
    
    # Create dataset
    test_path = dataset_paths[dataset_name]
    test_dataset = BubblemlForecast(
        filenames=test_path,
        fields=["dfun", "temperature", "velx", "vely"],
        norm="none",
        time_window=5,
        start_time=95
    )
    
    # Load model
    model = get_model(model_type, **model_kwargs)
    
    # Load weights
    model_data = torch.load(weights_path, weights_only=False)
    print(model_data.keys())
    
    diff_term, div_term = model_data['hyper_parameters']['normalization_constants']
    diff_term = torch.tensor(diff_term)
    div_term = torch.tensor(div_term)
    
    weight_state_dict = OrderedDict()
    for key, val in model_data["state_dict"].items():
        name = key[6:]
        weight_state_dict[name] = val
    del model_data
    
    model.load_state_dict(weight_state_dict, strict=False)
    
    # Normalize dataset - ignoring return values as we already have the constants
    _ = test_dataset.normalize(diff_term, div_term)
    
    criterion = LpLoss(d=2, p=2, reduce_dims=[0,1], reductions=["mean", "mean"])
    model.eval()
    
    start_time = test_dataset.start_time
    skip_itrs = test_dataset.time_window
    model_preds = []
    model_targets = []
    timesteps = []
    
    # Run inference
    total_start_time = time.time()
    iteration_times = []
    
    for itr in range(0, 500, skip_itrs):
        inp, tgt = test_dataset[itr]
        print(f"Autoreg pred {itr}, inp tw [{start_time+itr}, {start_time+itr+skip_itrs}], tgt tw [{start_time+itr+skip_itrs}, {start_time+itr+2*skip_itrs}]")
        
        if len(model_preds) > 0:
            inp = model_preds[-1]  # T, C, H, W
            
        inp = inp.float().unsqueeze(0)
        
        # Start timing this iteration
        iter_start_time = time.time()
        
        # Run inference
        pred = model(inp)
        
        # End timing this iteration
        iter_end_time = time.time()
        iter_duration = iter_end_time - iter_start_time
        iteration_times.append(iter_duration)
        
        print(f"Iteration time: {iter_duration:.4f} seconds")
        
        pred = pred.squeeze(0).detach().cpu()
        tgt = tgt.detach().cpu()
        
        model_preds.append(pred)
        model_targets.append(tgt)
        timesteps.append(torch.arange(start_time+itr+skip_itrs, start_time+itr+2*skip_itrs))
        
        print(criterion(pred, tgt))
    
    # Calculate total inference time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    avg_iteration_time = sum(iteration_times) / len(iteration_times) if iteration_times else 0
    
    print(f"\nInference Performance Summary:")
    print(f"Total inference time: {total_duration:.4f} seconds")
    print(f"Average iteration time: {avg_iteration_time:.4f} seconds")
    print(f"Number of iterations: {len(iteration_times)}")
    
    # Process results
    model_preds = torch.cat(model_preds, dim=0)         # T, C, H, W
    model_targets = torch.cat(model_targets, dim=0)     # T, C, H, W
    timesteps = torch.cat(timesteps, dim=0)             # T
    
    num_var = len(test_dataset.fields)                  # C
    preds = model_preds * div_term.view(1, num_var, 1, 1) + diff_term.view(1, num_var, 1, 1)     # denormalize
    targets = model_targets * div_term.view(1, num_var, 1, 1) + diff_term.view(1, num_var, 1, 1) # denormalize
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "predictions.pt")
    torch.save({"preds": preds, "targets": targets, "timesteps": timesteps}, save_path)
    
    # Plot results
    plot_bubbleml(preds, targets, timesteps, save_dir)
    
    print(f"Inference completed for {model_type} on {dataset_name}")
    print(f"Results saved to {save_dir}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run inference with bubbleformer models')
    parser.add_argument('--model', type=str, choices=['avit', 'avit_moe', 'both'], default='both',
                        help='Model type: avit, avit_moe, or both (default: both)')
    parser.add_argument('--dataset', type=str, choices=['sat_92', 'subcooled_100', 'grav_0.2', 'all'], default='all',
                        help='Dataset to run inference on (default: all)')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Custom path to model weights checkpoint file (optional)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Custom base directory to save results (optional)')
    args = parser.parse_args()
    
    # Determine which models to run
    models_to_run = []
    if args.model == 'both':
        models_to_run = ['avit', 'avit_moe']
    else:
        models_to_run = [args.model]
    
    # Determine which datasets to run
    datasets_to_run = []
    if args.dataset == 'all':
        datasets_to_run = ['sat_92', 'subcooled_100', 'grav_0.2']
    else:
        datasets_to_run = [args.dataset]
    
    # Run inference for each model on each dataset
    for model_type in models_to_run:
        for dataset_name in datasets_to_run:
            run_inference(model_type, dataset_name, args.weights_path, args.save_dir)

if __name__ == "__main__":
    main()