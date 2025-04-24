import os
import sys
import argparse
import torch
from datetime import datetime
import lightning as L
from lightning.pytorch.utilities.model_summary import ModelSummary, summarize

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from bubbleformer.models import get_model

def parse_args():
    parser = argparse.ArgumentParser(description='Display AViT model structure')
    parser.add_argument('--fields', type=int, default=4, help='Number of fields')
    parser.add_argument('--time-window', type=int, default=5, help='Time window size')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch size')
    parser.add_argument('--embed-dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--num-heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--processor-blocks', type=int, default=12, help='Number of processor blocks')
    parser.add_argument('--drop-path', type=float, default=0.2, help='Drop path rate')
    parser.add_argument('--input-shape', type=str, default='4,5,4,512,512', 
                        help='Input shape as comma-separated values: batch_size,time_window,fields,height,width')
    parser.add_argument('--max-depth', type=int, default=-1, help='Maximum depth of layers to show (-1 for all layers)')
    parser.add_argument('--save-dir', type=str, default='model_summaries', 
                        help='Directory to save the model summary output')
    parser.add_argument('--filename', type=str, default='', 
                        help='Filename to save the summary (default: auto-generated timestamp)')
    return parser.parse_args()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

class AvitLightningModule(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)

def main():
    args = parse_args()
    
    # Create AViT model
    model = get_model(
        "avit",
        fields=args.fields,
        time_window=args.time_window,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        processor_blocks=args.processor_blocks,
        drop_path=args.drop_path,
    )
    
    # Create a Lightning module wrapper
    lightning_model = AvitLightningModule(model)
    
    # Set example input array for shape inference
    input_shape = tuple(map(int, args.input_shape.split(',')))
    lightning_model.example_input_array = torch.randn(input_shape)
    
    # Create the model summary
    summary = ModelSummary(lightning_model, max_depth=args.max_depth)
    
    # Print model configuration
    print("\nAViT Model Configuration:")
    print("-" * 80)
    print(f"Fields: {args.fields}")
    print(f"Time Window: {args.time_window}")
    print(f"Patch Size: {args.patch_size}")
    print(f"Embedding Dimension: {args.embed_dim}")
    print(f"Number of Attention Heads: {args.num_heads}")
    print(f"Number of Processor Blocks: {args.processor_blocks}")
    print(f"Drop Path Rate: {args.drop_path}")
    print(f"Input Shape: {input_shape}")
    print("-" * 80)
    
    # Print the summary
    print("\nModel Summary:")
    print("-" * 80)
    print(summary)
    
    # Save summary to file if requested
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Generate filename if not provided
        if not args.filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"avit_model_summary_{timestamp}.txt"
        else:
            filename = args.filename
            if not filename.endswith('.txt'):
                filename += '.txt'
        
        filepath = os.path.join(args.save_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("AViT Model Configuration:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Fields: {args.fields}\n")
            f.write(f"Time Window: {args.time_window}\n")
            f.write(f"Patch Size: {args.patch_size}\n")
            f.write(f"Embedding Dimension: {args.embed_dim}\n")
            f.write(f"Number of Attention Heads: {args.num_heads}\n")
            f.write(f"Number of Processor Blocks: {args.processor_blocks}\n")
            f.write(f"Drop Path Rate: {args.drop_path}\n")
            f.write(f"Input Shape: {input_shape}\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("Model Summary:\n")
            f.write("-" * 80 + "\n")
            f.write(str(summary))
            
        print(f"\nModel summary saved to: {filepath}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            dummy_input = torch.randn(input_shape)
            output = model(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"\nError when running model inference: {e}")

if __name__ == "__main__":
    main() 