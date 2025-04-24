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
    
    # Run with plotting enabled
    python inference_autoregressive.py --plot
    
    # Run with custom max iterations
    python inference_autoregressive.py --max_iters 1000
    
    # Run with custom weights path
    python inference_autoregressive.py --weights_path /path/to/weights.ckpt
    
    # Run with custom weights path and save directory
    python inference_autoregressive.py --weights_path /path/to/weights.ckpt --save_dir /path/to/save
"""

import os
import torch
import argparse
import time
import re
import matplotlib.pyplot as plt
from collections import OrderedDict
from bubbleformer.models import get_model
from bubbleformer.data import BubblemlForecast
from bubbleformer.utils.losses import LpLoss
from bubbleformer.utils.plot_utils import plot_bubbleml

def test_eikonal_loss(phi):
    """
    phi = predicted sdf torch.Tensor(T,H,W)
    """
    dx = 1/32
    grad_x = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dx)
    grad_y = (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * dx)

    grad_x = torch.nn.functional.pad(grad_x, (1, 1), mode="replicate")
    grad_y = torch.nn.functional.pad(grad_y, (0, 0, 1, 1), mode="replicate")

    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    loss_map = torch.abs(grad_magnitude - 1)
    mean_loss = torch.mean(loss_map, dim=(1, 2))
    return mean_loss

def test_heat_flux_loss(temperature, phi, k=1.0):
    """
    Calculates heat flux loss across interface boundaries
    
    Args:
        temperature: temperature field torch.Tensor(T, H, W)
        phi: SDF field torch.Tensor(T, H, W) 
        k: thermal conductivity coefficient (default=1.0)
    
    Returns:
        Heat flux loss tensor of shape [T]
    """
    dx = 1/32
    
    # Calculate temperature gradients
    grad_T_x = (temperature[:, :, 2:] - temperature[:, :, :-2]) / (2 * dx)
    grad_T_y = (temperature[:, 2:, :] - temperature[:, :-2, :]) / (2 * dx)
    
    # Pad the gradients to maintain original size
    grad_T_x = torch.nn.functional.pad(grad_T_x, (1, 1), mode="replicate")
    grad_T_y = torch.nn.functional.pad(grad_T_y, (0, 0, 1, 1), mode="replicate")
    
    # Calculate phi gradient (to identify interface)
    grad_phi_x = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dx)
    grad_phi_y = (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * dx)
    
    # Pad the gradients
    grad_phi_x = torch.nn.functional.pad(grad_phi_x, (1, 1), mode="replicate")
    grad_phi_y = torch.nn.functional.pad(grad_phi_y, (0, 0, 1, 1), mode="replicate")
    
    # Calculate heat flux magnitude: q = -k∇T
    heat_flux_magnitude = k * torch.sqrt(grad_T_x**2 + grad_T_y**2)
    
    # Find interface locations (where phi is close to zero)
    interface_mask = torch.abs(phi) < 0.02  # Adjust threshold as needed
    
    # Calculate normal vectors at interface (normalized phi gradient)
    normal_magnitude = torch.sqrt(grad_phi_x**2 + grad_phi_y**2)
    normal_magnitude = torch.clamp(normal_magnitude, min=1e-5)  # Avoid division by zero
    
    # At interface, heat flux should be continuous across the boundary
    # Loss is high where heat flux changes rapidly near interface
    flux_change = torch.abs(
        torch.diff(heat_flux_magnitude, dim=1, prepend=heat_flux_magnitude[:, :1, :]) + 
        torch.diff(heat_flux_magnitude, dim=2, prepend=heat_flux_magnitude[:, :, :1])
    )
    
    # Focus loss on interface regions
    interface_flux_loss = flux_change * interface_mask
    
    # Average loss per timestep
    mean_loss = torch.sum(interface_flux_loss, dim=(1, 2)) / (torch.sum(interface_mask, dim=(1, 2)) + 1e-6)
    
    return mean_loss

def parse_model_params_from_path(weights_path):
    """
    Parse model parameters from the weights path
    For example, in a path like:
    RED_384_SED_768_ST_gelu_E_6_S_1_A_2_
    - RED_384 means routed_expert_embed_dim=384
    - SED_768 means shared_expert_embed_dim=768
    - ST_gelu means shared_expert_type="gelu"
    - E_6 means n_experts=6
    - S_1 means n_shared_experts=1
    - A_2 means top_k=2
    """
    # Extract the filename from the full path
    path_parts = weights_path.split('/')
    # Look for the part that contains the model parameters
    found_part = None
    for part in path_parts:
        if any(marker in part for marker in ['RED_', 'SED_', 'ST_', 'E_', 'S_', 'A_']):
            found_part = part
            break
    
    if not found_part:
        # Try to get the directory containing bubbleformer_logs
        dirname = os.path.dirname(weights_path)
        while dirname and not os.path.basename(dirname).startswith('bubbleformer_logs'):
            dirname = os.path.dirname(dirname)
        
        if not dirname:
            return {}  # Default parameters will be used
        
        found_part = os.path.basename(dirname)
    
    # Dictionary to store extracted parameters
    params = {}
    
    # Extract routed_expert_embed_dim
    red_match = re.search(r'RED_(\d+)', found_part)
    if red_match:
        params['routed_expert_embed_dim'] = int(red_match.group(1))
    
    # Extract shared_expert_embed_dim
    sed_match = re.search(r'SED_(\d+)', found_part)
    if sed_match:
        params['shared_expert_embed_dim'] = int(sed_match.group(1))
    
    # Extract shared_expert_type
    st_match = re.search(r'ST_(\w+)', found_part)
    if st_match:
        params['shared_expert_type'] = st_match.group(1)
    
    # Extract n_experts
    e_match = re.search(r'E_(\d+)', found_part)
    if e_match:
        params['n_experts'] = int(e_match.group(1))
    
    # Extract n_shared_experts
    s_match = re.search(r'S_(\d+)', found_part)
    if s_match:
        params['n_shared_experts'] = int(s_match.group(1))
    
    # Extract top_k
    a_match = re.search(r'A_(\d+)', found_part)
    if a_match:
        params['top_k'] = int(a_match.group(1))
    
    return params

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

def run_inference(model_type, dataset_name, plot=False, max_iters=500, custom_weights_path=None, custom_save_dir=None):
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
        # Default model parameters
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
            "routed_expert_embed_dim": 384,
            "shared_expert_type": "gelu",
            "shared_expert_embed_dim": 768
        }
        
        # Parse model parameters from weights path if available
        if custom_weights_path:
            parsed_params = parse_model_params_from_path(custom_weights_path)
            if parsed_params:
                print("Detected model parameters from weights path:")
                for key, value in parsed_params.items():
                    print(f"  {key}: {value}")
                    model_kwargs[key] = value
        
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
    print(f"Using model parameters:")
    for key, value in model_kwargs.items():
        print(f"  {key}: {value}")
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
    
    model.load_state_dict(weight_state_dict, strict=True)
    
    # Normalize dataset - ignoring return values as we already have the constants
    _ = test_dataset.normalize(diff_term, div_term)
    
    criterion = LpLoss(d=2, p=2, reduce_dims=[0,1], reductions=["mean", "mean"])
    model.eval()
    
    start_time = test_dataset.start_time
    skip_itrs = test_dataset.time_window
    model_preds = []
    model_targets = []
    timesteps = []
    
    # Start timing the inference process
    inference_start_time = time.time()
    step_times = []
    
    # Run inference using the user-specified max_iters
    for itr in range(0, max_iters, skip_itrs):
        step_start_time = time.time()
        
        inp, tgt = test_dataset[itr]
        # print(f"Autoreg pred {itr}, inp tw [{start_time+itr}, {start_time+itr+skip_itrs}], tgt tw [{start_time+itr+skip_itrs}, {start_time+itr+2*skip_itrs}]")
        
        if len(model_preds) > 0:
            inp = model_preds[-1]  # T, C, H, W
            
        inp = inp.float().unsqueeze(0)
        pred = model(inp)
        pred = pred.squeeze(0).detach().cpu()
        tgt = tgt.detach().cpu()
        
        model_preds.append(pred)
        model_targets.append(tgt)
        timesteps.append(torch.arange(start_time+itr+skip_itrs, start_time+itr+2*skip_itrs))
        
        # Record step time
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        step_times.append(step_time)
        
        # print(criterion(pred, tgt))
    
    # End timing and calculate statistics
    inference_end_time = time.time()
    total_inference_time = inference_end_time - inference_start_time
    avg_step_time = sum(step_times) / len(step_times) if step_times else 0
    
    # Print timing results
    print(f"\n=== Timing Results for {model_type} on {dataset_name} ===")
    print(f"Total inference time: {total_inference_time:.4f} seconds")
    print(f"Average time per step: {avg_step_time:.4f} seconds")
    print(f"Number of steps: {len(step_times)}")
    
    # Process results
    model_preds = torch.cat(model_preds, dim=0)         # T, C, H, W
    model_targets = torch.cat(model_targets, dim=0)     # T, C, H, W
    timesteps = torch.cat(timesteps, dim=0)             # T
    
    num_var = len(test_dataset.fields)                  # C
    preds = model_preds * div_term.view(1, num_var, 1, 1) + diff_term.view(1, num_var, 1, 1)     # denormalize
    targets = model_targets * div_term.view(1, num_var, 1, 1) + diff_term.view(1, num_var, 1, 1) # denormalize
    
    # Calculate eikonal loss
    phi_preds = preds[:, 0]  # Shape: [T, H, W] - phi field (dfun)
    eikonal_losses = test_eikonal_loss(phi_preds)
    
    # Calculate heat flux loss
    temp_preds = preds[:, 1]  # Shape: [T, H, W] - temperature field
    heat_flux_losses = test_heat_flux_loss(temp_preds, phi_preds)
    
    # Print loss results
    print(f"\n=== Physics-Based Loss Results for {model_type} on {dataset_name} ===")
    print(f"Average eikonal loss: {eikonal_losses.mean().item():.6f}")
    print(f"Average heat flux loss: {heat_flux_losses.mean().item():.6f}")
    
    # Plot the losses over time if plotting is enabled
    if plot:
        # Eikonal loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(timesteps.cpu().numpy(), eikonal_losses.cpu().numpy())
        plt.xlabel('Time Step')
        plt.ylabel('Eikonal Loss')
        plt.title(f'Eikonal Loss Over Time - {model_type} on {dataset_name}')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'eikonal_loss.png'))
        plt.close()
        
        # Heat flux loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(timesteps.cpu().numpy(), heat_flux_losses.cpu().numpy())
        plt.xlabel('Time Step')
        plt.ylabel('Heat Flux Loss')
        plt.title(f'Heat Flux Loss Over Time - {model_type} on {dataset_name}')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'heat_flux_loss.png'))
        plt.close()
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "predictions.pt")
    torch.save({
        "preds": preds, 
        "targets": targets, 
        "timesteps": timesteps,
        "timing": {
            "total_time": total_inference_time,
            "avg_step_time": avg_step_time,
            "step_times": step_times
        },
        "losses": {
            "eikonal_losses": eikonal_losses,
            "heat_flux_losses": heat_flux_losses,
            "avg_eikonal_loss": eikonal_losses.mean().item(),
            "avg_heat_flux_loss": heat_flux_losses.mean().item()
        }
    }, save_path)
    
    # Plot results only if requested
    if plot:
        plot_bubbleml(preds, targets, timesteps, save_dir)
        print(f"Results plotted to {save_dir}")
    
    print(f"Inference completed for {model_type} on {dataset_name}")
    print(f"Results saved to {save_dir}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run inference with bubbleformer models')
    parser.add_argument('--model', type=str, choices=['avit', 'avit_moe', 'both'], default='both',
                        help='Model type: avit, avit_moe, or both (default: both)')
    parser.add_argument('--dataset', type=str, choices=['sat_92', 'subcooled_100', 'grav_0.2', 'all'], default='all',
                        help='Dataset to run inference on (default: all)')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Whether to plot results using plot_bubbleml (default: False)')
    parser.add_argument('--max_iters', type=int, default=500,
                        help='Maximum number of iterations for inference (default: 500)')
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
            run_inference(model_type, dataset_name, args.plot, args.max_iters, args.weights_path, args.save_dir)

if __name__ == "__main__":
    main()