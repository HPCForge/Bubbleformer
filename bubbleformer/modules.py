from typing import Tuple, Optional, List

import wandb
from omegaconf import OmegaConf, DictConfig
import torch
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import lightning as L

from bubbleformer.models import get_model
from bubbleformer.utils.losses import LpLoss
from bubbleformer.utils.lr_schedulers import CosineWarmupLR
from bubbleformer.utils.plot_utils import wandb_sdf_plotter, wandb_temp_plotter, wandb_vel_plotter
import time
from bubbleformer.layers.moe import MoETracker
import os
import pandas as pd

class ForecastModule(L.LightningModule):
    """
    Module for training forecasting models with equal
    input and output time windows.
    Args:
        model_cfg (DictConfig): YAML Model config loaded using OmegaConf
        data_cfg (DictConfig): YAML Data config loaded using OmegaConf
        optim_cfg (DictConfig): YAML Optimizer config loaded using OmegaConf
        scheduler_cfg (DictConfig): YAML Scheduler config loaded using OmegaConf
        log_wandb (bool): Whether to log to wandb
        normalization_constants (Tuple[List, List]): 
                    Difference and Division constants for normalization
    """
    def __init__(
        self,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        optim_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        log_wandb: bool = False,
        normalization_constants: Tuple[List, List] = None
    ):
        super().__init__()
        self.model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        self.data_cfg = OmegaConf.to_container(data_cfg, resolve=True)
        self.optimizer_cfg = OmegaConf.to_container(optim_cfg, resolve=True)
        self.scheduler_cfg = OmegaConf.to_container(scheduler_cfg, resolve=True)
        if normalization_constants is not None:
            self.normalization_constants = normalization_constants
        self.log_wandb = log_wandb

        self.model_cfg["params"]["fields"] = len(self.data_cfg["fields"])
        self.model_cfg["params"]["time_window"] = self.data_cfg["time_window"]
        self.save_hyperparameters()

        self.criterion = LpLoss(d=2, p=2, reduce_dims=[0,1,2], reductions=["mean", "mean", "sum"])
        self.model = get_model(self.model_cfg["name"], **self.model_cfg["params"])
        # self.model = torch.compile(self.model)
        self.t_max = None
        self.validation_sample = None
        self.train_start_time = None
        self.val_start_time = None
        
        # Initialize MoE tracker
        self.moe_tracker = MoETracker(self.model)
    
    def setup(
        self,
        stage: Optional[str] = None
    ):
        if stage == "fit":
            self.t_max = self.trainer.estimated_stepping_batches

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        inp, tgt = batch 
        pred = self.model(inp)
        loss = self.criterion(pred, tgt)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
        self.log(
            "learning_rate",
            current_lr,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True
        )
        if self.log_wandb and self.trainer.is_global_zero:
            wandb.log({"train_loss": loss, "learning_rate": current_lr})

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        inp, tgt = batch
        pred = self.model(inp)
        loss = self.criterion(pred, tgt)
        if batch_idx == 0:
            self.validation_sample = (inp.detach(), tgt.detach(), pred.detach())

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        if self.log_wandb and self.trainer.is_global_zero:
            wandb.log({"val_loss": loss})

        return loss

    def configure_optimizers(self):
        opt_name = self.optimizer_cfg["name"]
        opt_params = self.optimizer_cfg["params"]
        if opt_name == "adamw":
            optimizer = AdamW(self.model.parameters(), **opt_params)
        elif opt_name == "adam":
            optimizer = Adam(self.model.parameters(), **opt_params)
        else:
            raise ValueError(f"Optimizer {opt_name} not supported")

        scheduler_name = self.scheduler_cfg["name"]
        scheduler_params = self.scheduler_cfg["params"]
        if scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(
                            optimizer,
                            T_max=self.t_max,
                            eta_min=scheduler_params["eta_min"],
                            last_epoch=self.trainer.global_step - 1
                        )
        if scheduler_name == "cosine_warmup":
            scheduler = CosineWarmupLR(
                            optimizer,
                            warmup_iters=scheduler_params["warmup_iters"],
                            max_iters=self.t_max,
                            eta_min=scheduler_params["eta_min"],
                            last_epoch=self.trainer.global_step - 1
                        )
        else:
            raise ValueError(f"Scheduler {scheduler_name} not supported")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def on_train_epoch_start(self):
        self.train_start_time = time.perf_counter()  
        
    def on_train_epoch_end(self):
        if self.train_start_time is not None: # when resuming from middle of epoch, variable stays none
            train_time = time.perf_counter() - self.train_start_time  
            if self.log_wandb and self.trainer.is_global_zero:
                wandb.log({"train_epoch_time": train_time, "epoch": self.current_epoch})  
        
    def on_validation_epoch_start(self):
        self.val_start_time = time.perf_counter()  
        if self.log_wandb and self.trainer.is_global_zero:
            train_loss = self.trainer.callback_metrics["train_loss"].item()
            wandb.log({"train_loss_epoch": train_loss, "epoch": self.current_epoch})
        
        # Enable MoE tracking for validation
        self.moe_tracker.enable_tracking()
        self.moe_tracker.reset_counts()

    def on_validation_epoch_end(self):
        if self.val_start_time is not None:
            val_time = time.perf_counter() - self.val_start_time  
            if self.log_wandb and self.trainer.is_global_zero:
                wandb.log({"val_epoch_time": val_time, "epoch": self.current_epoch})  
        
        # Plot and log MoE token distribution
        if self.log_wandb and self.trainer.is_global_zero:
            self.moe_tracker.plot_distribution(log_wandb=True, epoch=self.current_epoch)
            self.moe_tracker.log_distribution_data(log_wandb=True, epoch=self.current_epoch)
            
        # Save MoE distribution data to disk
        if self.trainer.is_global_zero:
            save_dir = os.path.join(self.trainer.log_dir, f"moe_distribution/epoch_{self.current_epoch}")
            self.save_distribution_data(save_dir)
        
        # Disable MoE tracking after validation
        self.moe_tracker.disable_tracking()
        
        fields = self.data_cfg["fields"]
        if self.validation_sample is None:
            return
        _, targets, predictions = self.validation_sample

        target_sample = targets[0] # T, C, H, W
        pred_sample = predictions[0] # T, C, H, W

        if self.log_wandb and self.trainer.is_global_zero:
            try:
                sdf_idx = fields.index("dfun")
                target_sdfs = wandb_sdf_plotter(target_sample[:,sdf_idx,:,:])
                pred_sdfs = wandb_sdf_plotter(pred_sample[:,sdf_idx,:,:])
                wandb.log({
                    "Target SDF": wandb.Image(target_sdfs, caption=f"Epc {self.current_epoch}"),
                    "Prediction SDF": wandb.Image(pred_sdfs, caption=f"Epc {self.current_epoch}"),
                })

            except ValueError:
                pass
            try:
                temp_idx = fields.index("temperature")
                target_temps = wandb_temp_plotter(target_sample[:,temp_idx,:,:])
                pred_temps = wandb_temp_plotter(pred_sample[:,temp_idx,:,:])
                wandb.log({
                    "Target Temp": wandb.Image(target_temps, caption=f"Epc {self.current_epoch}"),
                    "Prediction Temp": wandb.Image(pred_temps, caption=f"Epc {self.current_epoch}")
                })
            except ValueError:
                pass
            try:
                velx_idx = fields.index("velx")
                vely_idx = fields.index("vely")
                target_vel_field = torch.stack([
                                        target_sample[:,velx_idx,:,:],
                                        target_sample[:,vely_idx,:,:]
                                    ],
                                    dim=1
                                )
                pred_vel_field = torch.stack([
                                        pred_sample[:,velx_idx,:,:],
                                        pred_sample[:,vely_idx,:,:]
                                    ],
                                    dim=1
                                )
                #input_vels = wandb_vel_plotter(input_vel_field)
                target_vels = wandb_vel_plotter(target_vel_field)
                pred_vels = wandb_vel_plotter(pred_vel_field)
                wandb.log({
                    #"Input Velocity": wandb.Image(input_vels),
                    "Target Vel": wandb.Image(target_vels, caption=f"Epc {self.current_epoch}"),
                    "Prediction Vel": wandb.Image(pred_vels, caption=f"Epc {self.current_epoch}")
                })
            except ValueError:
                pass

        plt.close("all")

        if self.log_wandb and self.trainer.is_global_zero:
            val_loss = self.trainer.callback_metrics["val_loss"].item()
            wandb.log({"val_loss_epoch": val_loss, "epoch": self.current_epoch})

    def save_distribution_data(self, save_dir):
        """Save MoE token distribution data to disk as CSV files"""
        if not self.moe_tracker.is_tracking:
            return None
        
        layer_counts = self.moe_tracker.get_layer_counts()
        total_counts = self.moe_tracker.get_total_counts()
        
        if layer_counts is None or total_counts is None:
            return None
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual layer distributions
        for i, (counts, name) in enumerate(zip(layer_counts, self.moe_tracker.layer_names)):
            counts_np = counts.cpu().numpy()
            df = pd.DataFrame({
                'Expert ID': pd.Series(range(len(counts_np))).astype(int),
                'Token Count': counts_np
            })
            
            # Clean up layer name for filename
            clean_name = name.replace('.', '_')
            df.to_csv(os.path.join(save_dir, f"layer_{clean_name}_distribution.csv"), index=False)
        
        # Save total distribution
        total_np = total_counts.cpu().numpy()
        df = pd.DataFrame({
            'Expert ID': pd.Series(range(len(total_np))).astype(int),
            'Token Count': total_np
        })
        df.to_csv(os.path.join(save_dir, "total_distribution.csv"), index=False)
        
        return save_dir

    
