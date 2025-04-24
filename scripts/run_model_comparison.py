#!/usr/bin/env python3
"""
Model Comparison Script

This script runs inference on multiple model weights and compiles the results into a table.
It runs the Time_loss_compare.py script for each weights path and extracts the key metrics.

Usage:
    python run_model_comparison.py
"""

import os
import subprocess
import re
import csv
import pandas as pd
from datetime import datetime

# Create output directory if it doesn't exist
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# List of weights paths to evaluate
# weights_paths = [
#     "./bubbleformer_logs/0_RED_384_SED_768_ST_gelu_E_6_S_1_A_2_avit_moe_poolboiling_saturated_37815201/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
#     "./bubbleformer_logs/1_RED_384_SED_384_ST_gelu_E_6_S_1_A_2_avit_moe_poolboiling_saturated_37819807/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
#     "./bubbleformer_logs/2_RED_192_SED_384_ST_gelu_E_6_S_1_A_2_avit_moe_poolboiling_saturated_37820603/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
#     "./bubbleformer_logs/3_RED_96_SED_192_ST_gelu_E_6_S_1_A_2_avit_moe_poolboiling_saturated_37820879/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
#     "./bubbleformer_logs/4_RED_48_SED_96_ST_gelu_E_6_S_1_A_2_avit_moe_poolboiling_saturated_37820894/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
#     "./bubbleformer_logs/5_RED_24_SED_96_ST_gelu_E_12_S_1_A_2_avit_moe_poolboiling_saturated_37820622/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
#     "./bubbleformer_logs/6_RED_12_SED_96_ST_gelu_E_24_S_1_A_2_avit_moe_poolboiling_saturated_37820623/lightning_logs/version_0/checkpoints/epoch=399-step=200000.ckpt",
# ]

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

# Parameters for the inference run
model = "avit_moe"
dataset = "sat_92"
max_iters = 500

# Function to extract model name from weights path
def extract_model_name(path):
    """
    Extract model name in format like '0_RED_384_SED_768_ST_gelu_E_6_S_1_A_2_avit_moe'
    from the full weights path.
    """
    # Split the path by directory separator
    path_parts = path.split('/')
    
    # Find the part that contains 'bubbleformer_logs'
    for i, part in enumerate(path_parts):
        if 'bubbleformer_logs' in part and i+1 < len(path_parts):
            # The model name is the next part after 'bubbleformer_logs'
            full_name = path_parts[i+1]
            
            # Check if this is an avit model path (not avit_moe)
            if full_name.startswith('avit_') and 'avit_moe' not in full_name:
                return "avit"
            
            # Extract just up to before 'poolboiling' if it exists
            if 'poolboiling' in full_name:
                parts = full_name.split('_')
                poolboiling_index = parts.index('poolboiling')
                return "_".join(parts[:poolboiling_index])
            else:
                # If 'poolboiling' not found, use a reasonable prefix
                parts = full_name.split('_')
                if len(parts) > 10:
                    return "_".join(parts[:10])
                else:
                    return full_name
    
    # Fallback: just use the first part of the basename
    basename = os.path.basename(os.path.dirname(path))
    if basename == "checkpoints":
        # Go two levels up from checkpoints
        dirname = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
        return dirname.split('_')[0] + "_model"
    
    return basename.split('_')[0] + "_model"

# Function to run inference and parse results
def run_inference(weights_path):
    model_name = extract_model_name(weights_path)
    print(f"\n\n==== Running inference for {model_name} ====\n")
    
    # Determine which model type to use based on the model name
    current_model = "avit" if model_name == "avit" else model
    
    # Construct the command
    cmd = [
        "python", 
        "scripts/Time_loss_compare.py", 
        "--model", current_model, 
        "--dataset", dataset, 
        "--weights_path", weights_path, 
        "--max_iters", str(max_iters)
    ]
    
    # Run the command and capture output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    # Print the output for monitoring
    print(stdout)
    if stderr:
        print("Errors:", stderr)
    
    # Parse the results
    results = {
        "Model": model_name,
    }
    
    # Extract timing information
    avg_time_match = re.search(r'Average time per step: (\d+\.\d+)', stdout)
    if avg_time_match:
        results["Average Time Per Step (s)"] = float(avg_time_match.group(1))
    
    steps_match = re.search(r'Number of steps: (\d+)', stdout)
    if steps_match:
        results["Number of Steps"] = int(steps_match.group(1))
    
    # Extract loss information
    eikonal_loss_match = re.search(r'Average eikonal loss: (\d+\.\d+)', stdout)
    if eikonal_loss_match:
        results["Average Eikonal Loss"] = float(eikonal_loss_match.group(1))
    
    heat_flux_loss_match = re.search(r'Average heat flux loss: (\d+\.\d+)', stdout)
    if heat_flux_loss_match:
        results["Average Heat Flux Loss"] = float(heat_flux_loss_match.group(1))
    
    # Extract model parameters
    red_match = re.search(r'routed_expert_embed_dim: (\d+)', stdout)
    if red_match:
        results["Routed Expert Dim"] = int(red_match.group(1))
    
    sed_match = re.search(r'shared_expert_embed_dim: (\d+)', stdout)
    if sed_match:
        results["Shared Expert Dim"] = int(sed_match.group(1))
    
    e_match = re.search(r'n_experts: (\d+)', stdout)
    if e_match:
        results["Num Experts"] = int(e_match.group(1))
    
    return results

# Run inference for each weights path and collect results
results = []
for weights_path in weights_paths:
    try:
        result = run_inference(weights_path)
        results.append(result)
        print(f"Completed inference for {result['Model']}")
    except Exception as e:
        print(f"Error running inference for {weights_path}: {e}")

# Create a timestamp for the output file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file_csv = os.path.join(output_dir, f"model_comparison_{timestamp}.csv")
output_file_md = os.path.join(output_dir, f"model_comparison_{timestamp}.md")

# Save results as CSV
if results:
    df = pd.DataFrame(results)
    df.to_csv(output_file_csv, index=False)
    
    # Create a markdown table for better readability
    with open(output_file_md, 'w') as f:
        f.write("# Model Comparison Results\n\n")
        f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write the main metrics table
        f.write("## Performance Metrics\n\n")
        f.write("| Model | Routed Expert Dim | Shared Expert Dim | Num Experts | Avg Time (s) | Steps | Eikonal Loss | Heat Flux Loss |\n")
        f.write("|-------|------------------|-------------------|-------------|--------------|-------|--------------|---------------|\n")
        
        for result in results:
            model = result.get("Model", "N/A")
            red = result.get("Routed Expert Dim", "N/A")
            sed = result.get("Shared Expert Dim", "N/A")
            num_experts = result.get("Num Experts", "N/A")
            avg_time = result.get("Average Time Per Step (s)", "N/A")
            steps = result.get("Number of Steps", "N/A")
            eikonal = result.get("Average Eikonal Loss", "N/A")
            heat_flux = result.get("Average Heat Flux Loss", "N/A")
            
            f.write(f"| {model} | {red} | {sed} | {num_experts} | {avg_time} | {steps} | {eikonal} | {heat_flux} |\n")
        
        # Additional information
        f.write("\n## Run Configuration\n\n")
        f.write(f"- Model: {model}\n")
        f.write(f"- Dataset: {dataset}\n")
        f.write(f"- Max Iterations: {max_iters}\n")
    
    print(f"\nResults saved to {output_file_csv} and {output_file_md}")
else:
    print("No results were collected.") 