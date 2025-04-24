#!/usr/bin/env python3
"""
Autoregressive Inference Runner Script

This script runs inference_autoregressive.py for multiple model weights.
It automates the process of evaluating multiple models on specified datasets.

Usage:
    python run_autoregressive_inference.py
    python run_autoregressive_inference.py --dataset sat_92
    python run_autoregressive_inference.py --dataset all
"""

import os
import subprocess
import argparse
import re
from datetime import datetime

# Create output directory if it doesn't exist
output_dir = "./autoregressive_outputs"
os.makedirs(output_dir, exist_ok=True)

# List of weights paths to evaluate
weights_paths = [
    "./bubbleformer_logs/avit_poolboiling_saturated_37856288/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
    "./bubbleformer_logs/0_RED_768_SED_0_ST_gelu_E_6_S_0_A_2_avit_moe_poolboiling_saturated_37904031/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
    "./bubbleformer_logs/1_RED_640_SED_0_ST_gelu_E_6_S_0_A_2_avit_moe_poolboiling_saturated_37904245/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
    "./bubbleformer_logs/2_RED_576_SED_0_ST_gelu_E_6_S_0_A_2_avit_moe_poolboiling_saturated_37906443/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
    "./bubbleformer_logs/3_RED_512_SED_0_ST_gelu_E_6_S_0_A_2_avit_moe_poolboiling_saturated_37906506/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
    "./bubbleformer_logs/4_RED_384_SED_0_ST_gelu_E_6_S_0_A_2_avit_moe_poolboiling_saturated_37906541/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
    "./bubbleformer_logs/5_RED_320_SED_0_ST_gelu_E_6_S_0_A_2_avit_moe_poolboiling_saturated_37906636/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
    "./bubbleformer_logs/6_RED_256_SED_0_ST_gelu_E_6_S_0_A_2_avit_moe_poolboiling_saturated_37906726/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
    "./bubbleformer_logs/7_RED_192_SED_0_ST_gelu_E_6_S_0_A_2_avit_moe_poolboiling_saturated_37906739/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
]

# Function to extract model type from weights path
def extract_model_type(path):
    """
    Extract model type (avit or avit_moe) from the full weights path.
    """
    if 'avit_moe' in path:
        return 'avit_moe'
    else:
        return 'avit'

# Function to generate a meaningful name from the weights path
def generate_model_name(path):
    """
    Extract a meaningful model name from the weights path.
    """
    # Split the path by directory separator
    path_parts = path.split('/')
    
    # Find the part that contains 'bubbleformer_logs'
    for i, part in enumerate(path_parts):
        if 'bubbleformer_logs' in part and i+1 < len(path_parts):
            # The model name is the next part after 'bubbleformer_logs'
            return path_parts[i+1]
    
    # Fallback: return the basename of the directory
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))

# Function to determine the output directory for a given weights path and dataset
def get_output_dir(weights_path, dataset):
    """
    Determine the output directory where results are saved based on weights path and dataset.
    This mimics the logic in inference_autoregressive.py's generate_save_dir_from_weights_path function.
    """
    # Extract the epoch number from the checkpoint filename
    match = re.search(r'epoch=(\d+)', os.path.basename(weights_path))
    if not match:
        return None
    
    epoch_num = match.group(1)
    
    # Extract the base experiment directory (everything before lightning_logs)
    base_dir = weights_path.split('/lightning_logs')[0]
    
    # Construct the save directory
    save_dir = os.path.join(base_dir, f"epoch_{epoch_num}_outputs", dataset)
    
    return save_dir

# Function to generate a GIF from the plots
def generate_gif(output_dir, dataset, model_name):
    """
    Generate a GIF from PNG files in the plots directory.
    """
    plots_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_dir):
        print(f"Warning: Plots directory not found at {plots_dir}")
        return False
    
    # Check if there are PNG files in the directory
    png_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
    if not png_files:
        print(f"Warning: No PNG files found in {plots_dir}")
        return False
    
    # Generate the GIF filename based on dataset and model name
    gif_filename = f"{dataset}_{model_name}.gif"
    gif_path = os.path.join(output_dir, gif_filename)
    
    # Run ffmpeg to create the GIF
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output files without asking
        "-framerate", "20",
        "-pattern_type", "glob",
        "-i", f"{plots_dir}/*.png",
        gif_path
    ]
    
    try:
        print(f"Generating GIF: {gif_path}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error generating GIF: {stderr}")
            return False
        
        print(f"GIF successfully created: {gif_path}")
        return True
    except Exception as e:
        print(f"Exception when generating GIF: {e}")
        return False

# Function to run inference for a single weights path
def run_inference(weights_path, datasets):
    """
    Run inference_autoregressive.py for the specified weights path and datasets.
    """
    model_type = extract_model_type(weights_path)
    model_name = generate_model_name(weights_path)
    
    print(f"\n\n==== Running autoregressive inference for {model_name} ====\n")
    
    for dataset in datasets:
        print(f"\n=== Dataset: {dataset} ===\n")
        
        # Construct the command
        cmd = [
            "python", 
            "scripts/inference_autoregressive.py", 
            "--model", model_type, 
            "--dataset", dataset, 
            "--weights_path", weights_path
        ]
        
        # Run the command
        print(f"Executing: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        # Print the output for monitoring
        print(stdout)
        if stderr:
            print("Errors:", stderr)
        
        print(f"Completed inference for {model_name} on {dataset}")
        
        # Generate GIF from the plots
        output_dir = get_output_dir(weights_path, dataset)
        if output_dir and os.path.exists(output_dir):
            generate_gif(output_dir, dataset, model_name)
        else:
            print(f"Warning: Could not locate output directory for {model_name} on {dataset}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run autoregressive inference for multiple models')
    parser.add_argument('--dataset', type=str, choices=['sat_92', 'subcooled_100', 'grav_0.2', 'all'], default='sat_92',
                        help='Dataset to run inference on (default: sat_92)')
    args = parser.parse_args()
    
    # Determine which datasets to run
    datasets_to_run = []
    if args.dataset == 'all':
        datasets_to_run = ['sat_92', 'subcooled_100', 'grav_0.2']
    else:
        datasets_to_run = [args.dataset]
    
    # Run inference for each weights path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting autoregressive inference run at {timestamp}")
    print(f"Running on datasets: {', '.join(datasets_to_run)}")
    print(f"Processing {len(weights_paths)} model weights")
    
    for weights_path in weights_paths:
        try:
            run_inference(weights_path, datasets_to_run)
        except Exception as e:
            print(f"Error running inference for {weights_path}: {e}")
    
    print(f"\nCompleted all autoregressive inference runs")

if __name__ == "__main__":
    main() 