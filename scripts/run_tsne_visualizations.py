#!/usr/bin/env python3
"""
MoE t-SNE Visualization Script

This script automates running the generate_moe_tsne_visualizations.py script on multiple model weights.
It generates t-SNE visualizations for each model and saves them to a visualization directory.

Usage:
    python run_tsne_visualizations.py
"""

import os
import subprocess
import argparse
import re

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
    "./bubbleformer_logs/8_RED_128_SED_0_ST_gelu_E_6_S_0_A_2_avit_moe_poolboiling_saturated_37906754/lightning_logs/version_1/checkpoints/epoch=399-step=200000.ckpt",
]

# Function to extract model directory from checkpoint path
def extract_model_dir(checkpoint_path):
    """
    Extract the model directory from the checkpoint path.
    For example, from:
    ./bubbleformer_logs/0_RED_384_SED_768_ST_gelu_E_6_S_1_A_2_avit_moe_poolboiling_saturated_37815201/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt
    Extract:
    ./bubbleformer_logs/0_RED_384_SED_768_ST_gelu_E_6_S_1_A_2_avit_moe_poolboiling_saturated_37815201
    """
    parts = checkpoint_path.split('/lightning_logs')
    if len(parts) > 1:
        return parts[0]
    return os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path)))

# Function to run the t-SNE visualization script
def run_tsne_visualization(weights_path, dataset, time_window, perplexity):
    """Run the t-SNE visualization script for a single weights path"""
    model_dir = extract_model_dir(weights_path)
    output_dir = os.path.join(model_dir, "visualization", "tsne")
    
    print(f"\n\n==== Running t-SNE visualization for {os.path.basename(model_dir)} ====\n")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the command
    cmd = [
        "python", 
        "scripts/generate_moe_tsne_visualizations.py", 
        "--checkpoint", weights_path, 
        "--dataset", dataset,
        "--time_window", str(time_window),
        "--perplexity", str(perplexity),
        "--output_dir", output_dir
    ]
    
    # Run the command
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    # Print the output for monitoring
    print(stdout)
    if stderr:
        print("Errors:", stderr)
    
    print(f"t-SNE visualization completed for {os.path.basename(model_dir)}")
    print(f"Outputs saved to {output_dir}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate MoE t-SNE visualizations for multiple models')
    parser.add_argument('--dataset', type=str, default='sat_92', 
                       choices=['sat_92', 'subcooled_100', 'grav_0.2', 'all'],
                       help='Dataset to use for visualization (default: sat_92)')
    parser.add_argument('--time_window', type=int, default=5, 
                       help='Time window for visualizations (default: 5)')
    parser.add_argument('--perplexity', type=int, default=30, 
                       help='Perplexity parameter for t-SNE (default: 30)')
    args = parser.parse_args()
    
    # If dataset is 'all', run for all datasets
    datasets = ['sat_92', 'subcooled_100', 'grav_0.2'] if args.dataset == 'all' else [args.dataset]
    
    # Run t-SNE visualizations for each weights path and dataset
    for weights_path in weights_paths:
        for dataset in datasets:
            try:
                run_tsne_visualization(weights_path, dataset, args.time_window, args.perplexity)
            except Exception as e:
                print(f"Error running t-SNE visualization for {weights_path} on {dataset}: {e}")

if __name__ == "__main__":
    main() 