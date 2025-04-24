import os
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from einops import rearrange
from collections import OrderedDict
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

# Add path to bubbleformer directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bubbleformer.models import get_model
from bubbleformer.modules import ForecastModule
from bubbleformer.layers.moe import Gate, MoE
from bubbleformer.data import BubblemlForecast


class MoETSNEVisualizer:
    """
    Utility class to generate t-SNE visualizations of MoE expert outputs
    for each spatial patch.
    """
    def __init__(self, model):
        self.model = model
        self.moe_layers = []
        self.layer_names = []
        self.expert_outputs = {}
        self._find_moe_layers(model)
        self._register_hooks()
        
    def _find_moe_layers(self, module, prefix="", layer_counter=None):
        """Recursively find all MoE layers in the model"""
        if layer_counter is None:
            layer_counter = [0]  # Use a list as a mutable reference
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, MoE):
                child.layer_idx = layer_counter[0]  # Assign layer index
                self.moe_layers.append(child)
                self.layer_names.append(full_name)
                layer_counter[0] += 1  # Increment counter
            else:
                self._find_moe_layers(child, full_name, layer_counter)
    
    def _register_hooks(self):
        """Register forward hooks to capture expert outputs"""
        def get_expert_outputs_hook(layer_idx):
            def hook(module, input, output):
                # Get the original input
                x = input[0]  # The input tensor to MoE
                batch_size, time_window, height, width, hidden_dim = x.shape
                
                # Reshape input for gate and experts
                x_flat = rearrange(x, "b t h w d -> (b t h w) d")
                
                # Get weights and indices from gate
                weights, indices = module.gate(x_flat)
                
                # Process through experts to generate the combined expert output
                counts = torch.bincount(indices.flatten(), minlength=module.n_routed_experts).tolist()
                expert_output = torch.zeros_like(x_flat)
                
                for i in range(module.n_routed_experts):
                    if counts[i] == 0:
                        continue
                    expert = module.experts[i]
                    idx, top = torch.where(indices == i)
                    expert_output[idx] += expert(x_flat[idx]) * weights[idx, top, None]
                
                # Store the expert output and shape for later visualization
                self.expert_outputs[layer_idx] = (expert_output.detach().clone(), x.shape)
                
                # Let the forward pass continue normally
                return output
            return hook
        
        for layer in self.moe_layers:
            # Register a forward hook (not a pre-hook)
            layer.register_forward_hook(get_expert_outputs_hook(layer.layer_idx))
    
    def generate_tsne_visualizations(self, batch_size=1, time_window=1, perplexity=30):
        """
        Generate t-SNE visualizations of expert outputs for each patch
        
        Args:
            batch_size: Number of batches to visualize
            time_window: Number of time steps to visualize
            perplexity: Perplexity parameter for t-SNE
            
        Returns:
            A list of figures, one for each MoE layer
        """
        figures = []
        
        for layer_idx, layer in enumerate(self.moe_layers):
            if layer_idx not in self.expert_outputs:
                print(f"No expert outputs captured for layer {layer_idx}")
                continue
                
            expert_output, orig_shape = self.expert_outputs[layer_idx]
            b, t, h, w, hidden_dim = orig_shape
            
            # Ensure batch_size and time_window don't exceed the actual dimensions
            batch_size = min(batch_size, b)
            time_window = min(time_window, t)
            
            # Create a grid figure
            fig, axes = plt.subplots(batch_size, time_window, 
                                    figsize=(time_window * 5, batch_size * 5),
                                    squeeze=False)
            
            # Reshape expert outputs to match original dimensions
            expert_output = expert_output.reshape(b, t, h, w, hidden_dim)
            
            # Generate t-SNE visualizations for each batch and time step
            for b_idx in range(batch_size):
                for t_idx in range(time_window):
                    # Get expert outputs for this batch and time step
                    batch_time_outputs = expert_output[b_idx, t_idx]  # Shape: (h, w, hidden_dim)
                    
                    # Reshape to (h*w, hidden_dim)
                    batch_time_outputs_flat = batch_time_outputs.reshape(h*w, hidden_dim)
                    
                    # Calculate t-SNE embedding
                    tsne = TSNE(n_components=2, perplexity=min(perplexity, h*w-1), 
                               random_state=42, init='pca')
                    tsne_result = tsne.fit_transform(batch_time_outputs_flat.cpu().numpy())
                    
                    # Create spatial position indicators
                    positions = np.array([(i, j) for i in range(h) for j in range(w)])
                    
                    # Plot t-SNE results
                    ax = axes[b_idx, t_idx]
                    
                    # Create a colormap representing spatial position
                    # Map the (row,col) position to a single value from 0 to 1
                    position_values = (positions[:, 0] * w + positions[:, 1]) / (h * w - 1)
                    
                    # Create a scatter plot with points colored by their position
                    scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                       c=position_values, 
                                       cmap='brg', 
                                       alpha=0.8,
                                       s=50)
                    
                    # Add a colorbar
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Spatial Position (normalized)')
                    
                    # Add labels for a few selected points to show their spatial position
                    # Select a few points to label (e.g., corners and center)
                    key_positions = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1), (h//2, w//2)]
                    for pos in key_positions:
                        idx = pos[0] * w + pos[1]
                        ax.annotate(f"({pos[0]},{pos[1]})", 
                                   (tsne_result[idx, 0], tsne_result[idx, 1]),
                                   fontsize=9,
                                   xytext=(5, 5),
                                   textcoords='offset points')
                    
                    ax.set_title(f"Batch {b_idx}, Time {t_idx}")
                    ax.set_xlabel("t-SNE Dimension 1")
                    ax.set_ylabel("t-SNE Dimension 2")
                    ax.grid(alpha=0.3)
            
            plt.tight_layout()
            fig.suptitle(f"Layer {layer_idx}: {self.layer_names[layer_idx]} - t-SNE of Expert Outputs", fontsize=16)
            plt.subplots_adjust(top=0.9)
            figures.append(fig)
            
        return figures


def load_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    # Load the checkpoint first
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Print what's available in the checkpoint
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # Get model parameters
    if 'hyper_parameters' in checkpoint:
        hyper_params = checkpoint['hyper_parameters']
        model_cfg = hyper_params.get('model_cfg', {})
        # Get field count from normalization constants or model config
        if 'normalization_constants' in hyper_params:
            diff_term, div_term = hyper_params['normalization_constants']
            if isinstance(diff_term, list):
                fields_count = len(diff_term)
            elif isinstance(diff_term, torch.Tensor):
                fields_count = diff_term.shape[0]
            else:
                fields_count = 4  # Default
        else:
            fields_count = 4  # Default
    else:
        # Default parameters
        model_cfg = {"name": "avit_moe", "params": {}}
        fields_count = 4
    
    # Get model type 
    model_name = model_cfg.get("name", "avit_moe")
    
    # Create model parameters
    model_params = {
        "fields": fields_count,
        "patch_size": 16,
        "embed_dim": 384,
        "processor_blocks": 12,
        "num_heads": 6,
        "drop_path": 0.2,
        "n_experts": 6,
        "n_shared_experts": 1,
        "top_k": 2,
        "routed_expert_embed_dim": 128,
        "shared_expert_type": "gelu",
        "shared_expert_embed_dim": 0
    }
    
    # Update with any available params from checkpoint
    if 'params' in model_cfg:
        for key, value in model_cfg['params'].items():
            model_params[key] = value
    
    # Create model with parameters
    model = get_model(model_name, **model_params)
    
    # Extract state dict from checkpoint
    if 'state_dict' in checkpoint:
        # Extract just the model state dict
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() 
                     if k.startswith('model.')}
    else:
        # It's a direct model state dict
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=True)
    
    # Get normalization constants
    if 'hyper_parameters' in checkpoint and 'normalization_constants' in checkpoint['hyper_parameters']:
        diff_term, div_term = checkpoint['hyper_parameters']['normalization_constants']
        if isinstance(diff_term, list):
            diff_term = torch.tensor(diff_term)
            div_term = torch.tensor(div_term)
    else:
        diff_term = None
        div_term = None
    
    return model, diff_term, div_term, fields_count


def main():
    parser = argparse.ArgumentParser(description='Generate MoE t-SNE visualizations')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='sat_92', 
                       choices=['sat_92', 'subcooled_100', 'grav_0.2'],
                       help='Dataset to use for visualization')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for input tensor')
    parser.add_argument('--time_window', type=int, default=5, help='Time window for input tensor')
    parser.add_argument('--perplexity', type=int, default=30, help='Perplexity parameter for t-SNE')
    parser.add_argument('--output_dir', type=str, default='./moe_tsne', help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Dataset paths based on selection
    dataset_paths = {
        'sat_92': ["/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Saturated-FC72-2D-0.1/Twall-92.hdf5"],
        'subcooled_100': ["/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-SubCooled-FC72-2D-0.1/Twall-100.hdf5"],
        'grav_0.2': ["/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Gravity-FC72-2D-0.1/gravY-0.2.hdf5"]
    }
    
    # Load model from checkpoint
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model, diff_term, div_term, fields_count = load_checkpoint(args.checkpoint)
    model.eval()
    
    # Create dataset
    test_path = dataset_paths[args.dataset]
    print(f"Loading dataset from: {test_path}")
    
    fields = ["dfun", "temperature", "velx", "vely"]
    if fields_count == 3:
        fields = ["dfun", "velx", "vely"]  # Adjust if needed
    
    test_dataset = BubblemlForecast(
        filenames=test_path,
        fields=fields,
        norm="none",
        time_window=args.time_window,
        start_time=95
    )
    
    # Normalize dataset if normalization constants are available
    if diff_term is not None and div_term is not None:
        print("Normalizing dataset with constants from checkpoint")
        test_dataset.normalize(diff_term, div_term)
    else:
        print("Using default normalization")
        test_dataset.normalize()
    
    # Get input sample
    inp, _ = test_dataset[0]
    inp = inp.float().unsqueeze(0)  # Add batch dimension
    
    # Create t-SNE visualizer
    tsne_visualizer = MoETSNEVisualizer(model)
    
    # Run model forward pass to capture expert outputs
    print("Running model forward pass...")
    with torch.no_grad():
        _ = model(inp)
    
    # Generate t-SNE visualizations
    print("Generating t-SNE visualizations...")
    figures = tsne_visualizer.generate_tsne_visualizations(
        batch_size=1, 
        time_window=args.time_window,
        perplexity=args.perplexity
    )
    
    # Save visualizations
    os.makedirs(args.output_dir, exist_ok=True)
    for i, fig in enumerate(figures):
        output_path = os.path.join(args.output_dir, f"moe_layer_{i}_tsne_{args.dataset}.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE visualization to {output_path}")
    
    print(f"Generated {len(figures)} t-SNE visualizations")


if __name__ == "__main__":
    main() 