import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import argparse
from omegaconf import OmegaConf
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from bubbleformer.models import get_model
from bubbleformer.modules import ForecastModule
from einops import rearrange

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Generate MoE Visualizations')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to model config')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save visualizations')
    parser.add_argument('--log_wandb', action='store_true', help='Whether to log to wandb')
    parser.add_argument('--wandb_project', type=str, default='bubbleformer', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity name')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run model on')
    return parser

def load_model(checkpoint_path, config_path, device='cuda'):
    """Load model from checkpoint"""
    config = OmegaConf.load(config_path)
    model_cfg = config.model
    data_cfg = config.data
    optim_cfg = config.optim
    scheduler_cfg = config.scheduler
    
    module = ForecastModule(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        optim_cfg=optim_cfg,
        scheduler_cfg=scheduler_cfg,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    module.load_state_dict(checkpoint['state_dict'])
    module.to(device)
    module.eval()
    
    return module, data_cfg

def generate_similarity_heatmaps(model, sample_input, output_dir, log_wandb=False):
    """Generate similarity heatmaps for MoE layers"""
    print("Generating similarity heatmaps...")
    
    # Create tracker
    tracker = MoETracker(model)
    
    # Store original methods
    original_forward_methods = {}
    for i, layer in enumerate(tracker.moe_layers):
        original_forward_methods[i] = layer.forward
    
    # Get model outputs without generating heatmaps
    with torch.no_grad():
        _ = model(sample_input)
    
    # Generate heatmaps for each layer
    for layer_idx, layer in enumerate(tracker.moe_layers):
        layer_name = tracker.layer_names[layer_idx]
        print(f"Processing layer {layer_idx}: {layer_name}")
        
        # Create output directory
        layer_dir = os.path.join(output_dir, 'heatmaps', f'layer_{layer_idx}_{layer_name.replace(".", "_")}')
        os.makedirs(layer_dir, exist_ok=True)
        
        # Define a custom forward to capture intermediate outputs for this layer
        def custom_forward(x, orig_forward=original_forward_methods[layer_idx], layer=layer):
            batch_size, time_window, height, width, hidden_dim = x.shape
            
            # Reshape for gate and experts
            x_flat = rearrange(x, "b t h w d -> (b t h w) d")
            
            # Get weights and indices
            weights, indices = layer.gate(x_flat)
            
            # Process through experts
            counts = torch.bincount(indices.flatten(), minlength=layer.n_routed_experts).tolist()
            output = torch.zeros_like(x_flat)
            
            for i in range(layer.n_routed_experts):
                if counts[i] == 0:
                    continue
                expert = layer.experts[i]
                idx, top = torch.where(indices == i)
                output[idx] += expert(x_flat[idx]) * weights[idx, top, None]
            
            # Generate heatmap for this layer's output
            heatmap_fig = generate_layer_heatmap(output, batch_size, time_window, height, width, layer.hidden_dim)
            
            # Save heatmap
            plt.savefig(os.path.join(layer_dir, f'heatmap.png'))
            plt.close(heatmap_fig)
            
            if log_wandb:
                wandb.log({f"heatmaps/layer_{layer_idx}": wandb.Image(heatmap_fig)})
            
            # Continue with original forward to get final output
            shared_expert_output = layer.shared_experts(x_flat)
            combined_output = output + shared_expert_output
            return combined_output.view(batch_size, time_window, height, width, hidden_dim)
        
        # Replace forward method temporarily
        layer.forward = custom_forward
        
        # Run model with custom forward
        with torch.no_grad():
            _ = model(sample_input)
        
        # Restore original forward method
        layer.forward = original_forward_methods[layer_idx]
    
    print("Similarity heatmaps generated successfully!")

def generate_layer_heatmap(output_vectors, batch_size, time_window, height, width, hidden_dim):
    """Generate heatmap for a layer's output"""
    # Reshape output vectors to match original dimensions
    output_vectors = output_vectors.reshape(batch_size, time_window, height, width, hidden_dim)
    
    # Calculate center indices
    center_h, center_w = height // 2, width // 2
    
    # Create a grid figure
    fig, axes = plt.subplots(batch_size, time_window, 
                            figsize=(time_window * 3, batch_size * 3),
                            squeeze=False)
    
    # Generate heatmaps for each batch and time step
    for b in range(batch_size):
        for t in range(time_window):
            # Get the center vector as reference
            center_vector = output_vectors[b, t, center_h, center_w]
            
            # Calculate cosine similarity for each position
            similarities = torch.zeros(height, width)
            for h in range(height):
                for w in range(width):
                    current_vector = output_vectors[b, t, h, w]
                    similarity = F.cosine_similarity(center_vector.unsqueeze(0), 
                                                    current_vector.unsqueeze(0), 
                                                    dim=1)
                    similarities[h, w] = similarity.item()
            
            # Plot heatmap
            ax = axes[b, t]
            sns.heatmap(similarities.cpu().numpy(), 
                       ax=ax, 
                       vmin=-1, 
                       vmax=1, 
                       cmap='jet',
                       cbar=(t == time_window-1))
            
            ax.set_title(f"Batch {b}, Time {t}")
            if b == batch_size - 1:
                ax.set_xlabel("Width")
            if t == 0:
                ax.set_ylabel("Height")
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def generate_dim_reduction_visualizations(model, sample_input, output_dir, log_wandb=False):
    """Generate dimensionality reduction visualizations for MoE layers"""
    print("Generating dimensionality reduction visualizations...")
    
    # Create tracker
    tracker = MoETracker(model)
    
    # Store original methods
    original_forward_methods = {}
    for i, layer in enumerate(tracker.moe_layers):
        original_forward_methods[i] = layer.forward
    
    # Get model outputs without generating visualizations
    with torch.no_grad():
        _ = model(sample_input)
    
    # Generate visualizations for each layer
    for layer_idx, layer in enumerate(tracker.moe_layers):
        layer_name = tracker.layer_names[layer_idx]
        print(f"Processing layer {layer_idx}: {layer_name}")
        
        # Create output directory
        layer_dir = os.path.join(output_dir, 'dim_reduction', f'layer_{layer_idx}_{layer_name.replace(".", "_")}')
        os.makedirs(layer_dir, exist_ok=True)
        
        # Define a custom forward to capture intermediate outputs for this layer
        def custom_forward(x, orig_forward=original_forward_methods[layer_idx], layer=layer):
            batch_size, time_window, height, width, hidden_dim = x.shape
            
            # Process input normally to get output
            result = orig_forward(x)
            
            # Get the output in flat form for visualization
            result_flat = rearrange(result, "b t h w d -> (b t h w) d")
            
            # Generate visualization
            vis_fig = visualize_layer_dim_reduction(result_flat, batch_size, time_window, height, width, hidden_dim)
            
            # Save visualization
            plt.savefig(os.path.join(layer_dir, f'tsne.png'))
            plt.close(vis_fig)
            
            if log_wandb:
                wandb.log({f"dim_reduction/layer_{layer_idx}": wandb.Image(vis_fig)})
            
            return result
        
        # Replace forward method temporarily
        layer.forward = custom_forward
        
        # Run model with custom forward
        with torch.no_grad():
            _ = model(sample_input)
        
        # Restore original forward method
        layer.forward = original_forward_methods[layer_idx]
    
    print("Dimensionality reduction visualizations generated successfully!")

def visualize_layer_dim_reduction(output_vectors, batch_size, time_window, height, width, hidden_dim):
    """Visualize dimensionality reduction for a layer's output"""
    # Reshape to separate each batch and time step
    output_vectors = output_vectors.reshape(batch_size, time_window, height, width, hidden_dim)
    
    # Create a grid figure for all batch items and timesteps
    fig, axes = plt.subplots(batch_size, time_window, 
                            figsize=(time_window * 4, batch_size * 4),
                            squeeze=False)
    
    # For each batch and timestep, create a t-SNE visualization
    for b in range(batch_size):
        for t in range(time_window):
            # Extract vectors for this batch and timestep
            vectors = output_vectors[b, t].reshape(height * width, hidden_dim).cpu().numpy()
            
            # Create t-SNE embedding
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, height*width-1))
            tsne_result = tsne.fit_transform(vectors)
            
            # Plot as a scatter plot with points colored by their position
            ax = axes[b, t]
            points = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                               c=np.arange(height * width), 
                               cmap='brg', 
                               alpha=0.7,
                               s=30)
            
            # Add a colorbar to show the position mapping
            plt.colorbar(points, ax=ax)
            
            # Set titles and labels
            ax.set_title(f"Batch {b}, Time {t}")
            if b == batch_size - 1:
                ax.set_xlabel("t-SNE Dimension 1")
            if t == 0:
                ax.set_ylabel("t-SNE Dimension 2")
    
    plt.tight_layout()
    return fig

def main():
    # Parse arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    if args.log_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    
    # Load model and data config
    module, data_cfg = load_model(args.checkpoint, args.config, args.device)
    model = module.model
    model.eval()
    
    # Create dummy input
    time_window = data_cfg.get('time_window', 12)
    fields = len(data_cfg.get('fields', ['velx', 'vely', 'dfun']))
    patch_size = model.embed.patch_size
    height, width = 32, 32  # Adjust based on your data
    sample_input = torch.randn(args.batch_size, time_window, fields, height, width).to(args.device)
    
    # Generate heatmaps
    generate_similarity_heatmaps(model, sample_input, args.output_dir, args.log_wandb)
    
    # Generate dimensionality reduction visualizations
    generate_dim_reduction_visualizations(model, sample_input, args.output_dir, args.log_wandb)
    
    if args.log_wandb:
        wandb.finish()
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
