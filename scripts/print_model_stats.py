import os
import sys
import argparse
import pprint
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from bubbleformer.data import BubblemlForecast
from bubbleformer.modules import ForecastModule
from bubbleformer.utils.param_counter import print_model_stats

@hydra.main(version_base=None, config_path="../bubbleformer/config", config_name="default")
def main(cfg: DictConfig) -> None:
    # Set up configuration
    params = {}
    params["data_cfg"] = cfg.data_cfg
    params["model_cfg"] = cfg.model_cfg
    params["optim_cfg"] = cfg.optim_cfg
    params["scheduler_cfg"] = cfg.scheduler_cfg
    
    # Create datasets to get normalization constants
    train_dataset = BubblemlForecast(
        filenames=cfg.data_cfg.train_paths,
        fields=cfg.data_cfg.fields,
        norm=cfg.data_cfg.normalize,
        time_window=cfg.data_cfg.time_window,
    )
    normalization_constants = train_dataset.normalize()
    diff_term = normalization_constants[0].tolist()
    div_term = normalization_constants[1].tolist()
    
    # Create model
    train_module = ForecastModule(
        model_cfg=cfg.model_cfg,
        data_cfg=cfg.data_cfg,
        optim_cfg=cfg.optim_cfg,
        scheduler_cfg=cfg.scheduler_cfg,
        log_wandb=False,
        normalization_constants=(diff_term, div_term),
    )
    
    # Print model config
    print("\nModel Configuration:")
    print("-" * 80)
    print(OmegaConf.to_yaml(cfg.model_cfg))
    print("-" * 80)
    
    # Print model parameter statistics
    print_model_stats(train_module.model)
    
    # Save to file if --output is specified
    if len(sys.argv) > 1 and sys.argv[1] == "--output":
        if len(sys.argv) > 2:
            output_path = sys.argv[2]
            with open(output_path, 'w') as f:
                # Redirect stdout to file
                old_stdout = sys.stdout
                sys.stdout = f
                
                # Print model config
                print("\nModel Configuration:")
                print("-" * 80)
                print(OmegaConf.to_yaml(cfg.model_cfg))
                print("-" * 80)
                
                # Print model parameter statistics
                print_model_stats(train_module.model)
                
                # Restore stdout
                sys.stdout = old_stdout
                print(f"Model statistics saved to {output_path}")

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
