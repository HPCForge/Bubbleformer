import random
import time
from typing import Tuple, Optional, List

import wandb
from omegaconf import OmegaConf, DictConfig
import torch
from torch.optim import AdamW, Adam
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import lightning as L

from bubbleformer.models import get_model
from bubbleformer.utils.losses import LpLoss, L1Loss
from bubbleformer.utils.lr_schedulers import CosineWarmupLR
from bubbleformer.utils.plot_utils import wandb_sdf_plotter, wandb_temp_plotter, wandb_vel_plotter

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

        self.criterion = L1Loss() #LpLoss(d=2, p=2, reduce_dims=[0,1,2], reductions=["mean", "mean", "sum"])
        self.model_cfg["params"]["input_fields"] = len(self.data_cfg["input_fields"])
        self.model_cfg["params"]["output_fields"] = len(self.data_cfg["output_fields"])
        self.model_cfg["params"]["time_window"] = self.data_cfg["time_window"]
        self.model = get_model(self.model_cfg["name"], **self.model_cfg["params"])
        #self.model = torch.compile(self.model)

        self.save_hyperparameters()
        self.t_max = None
        self.validation_sample = None
        self.train_start_time = None
        self.val_start_time = None

    def default_log(self, key, value, **kwargs):
        kwargs["on_step"] = True
        kwargs["on_epoch"] = True
        kwargs["prog_bar"] = True
        kwargs["logger"] = True
        self.log(key, value, **kwargs)
        if self.log_wandb and self.trainer.is_global_zero:
            wandb.log({key: value})
            
    def default_log_dict(self, dict, **kwargs):
        kwargs["on_step"] = True
        kwargs["on_epoch"] = True
        kwargs["prog_bar"] = True
        kwargs["logger"] = True
        self.log_dict(dict, **kwargs)
        if self.log_wandb and self.trainer.is_global_zero:
            wandb.log(dict)
        
    def get_current_lr(self):
        opt = self.optimizers()
        return opt.param_groups[0]['lr']
        
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
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        inp, tgt = batch 
        pred = self.model(inp)
        loss = self.criterion(pred, tgt)

        self.default_log_dict({
            "train_loss": loss,
            "learning_rate": self.get_current_lr()
        })

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

        self.default_log_dict({"val_loss": loss})
        
        return loss

    def configure_optimizers(self):
        opt_name = self.optimizer_cfg["name"]
        opt_params = self.optimizer_cfg["params"]
        if opt_name == "adamw":
            optimizer = AdamW(self.model.parameters(), **opt_params, fused=True)
        elif opt_name == "adam":
            optimizer = Adam(self.model.parameters(), **opt_params)
        elif opt_name == "lion":
            optimizer = Lion(self.model.parameters(), **opt_params)
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
        self.train_start_time = time.time()

    def on_train_epoch_end(self):
        if self.train_start_time is not None: # when resuming from middle of epoch, var is None
            train_time = time.time() - self.train_start_time
            if self.log_wandb and self.trainer.is_global_zero:
                wandb.log({"train_epoch_time": train_time, "epoch": self.current_epoch})

    def on_validation_epoch_start(self):
        self.val_start_time = time.time()  
        if self.log_wandb and self.trainer.is_global_zero:
            try:
                train_loss = self.trainer.callback_metrics["train_loss"].item()
                wandb.log({"train_loss_epoch": train_loss, "epoch": self.current_epoch})
            except:
                pass

    def on_validation_epoch_end(self):
        if self.val_start_time is not None:
            val_time = time.time() - self.val_start_time
            if self.log_wandb and self.trainer.is_global_zero:
                wandb.log({"val_epoch_time": val_time, "epoch": self.current_epoch})

        fields = self.data_cfg["output_fields"]
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
            try:
                val_loss = self.trainer.callback_metrics["val_loss"].item()
                wandb.log({"val_loss_epoch": val_loss, "epoch": self.current_epoch})
            except:
                pass


class ConditionedForecastModule(ForecastModule):
    """
    Module for training forecasting models with different
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
        super().__init__(
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            log_wandb=log_wandb,
            normalization_constants=normalization_constants
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        return self.model(x, cond)

    def training_step(
        self,
        batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        inp, tgt, cond = batch
        pred = self.model(inp, cond)
        loss = self.criterion(pred, tgt)
        
        self.default_log_dict({
            "train_loss": loss,
            "learning_rate": self.get_current_lr()
        })

        return loss

    def validation_step(
        self,
        batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        inp, tgt, cond = batch
        pred = self.model(inp, cond)
        loss = self.criterion(pred, tgt)
        if batch_idx == 0:
            self.validation_sample = (inp.detach(), tgt.detach(), pred.detach())

        self.default_log_dict({"val_loss": loss})

        return loss
    
class MoEConditionedForecastModule(ConditionedForecastModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        optim_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        log_wandb: bool = False,
        normalization_constants: Tuple[List, List] = None
    ):
        super().__init__(
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            log_wandb=log_wandb,
            normalization_constants=normalization_constants
        )

    def training_step(
        self,
        batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        inp, tgt, cond = batch

        # The input is [B, T, C, H, W]
        # Randomly flip along the horizontal axis of the input and target.
        if random.random() < 0.5:
            inp = torch.flip(inp, dims=[-1])
            tgt = torch.flip(tgt, dims=[-1])

        # Add gaussian noise to the input
        if random.random() < 0.4:
            scale = random.choice([0.01, 0.05, 0.1, 0.5, 1])
            inp = inp + torch.normal(0, scale, inp.shape, device=inp.device)

        pred, moe_outputs = self.model(inp, cond)

        data_loss = self.criterion(pred, tgt)
        routing_loss = sum(moe_output.load_balance_loss for moe_output in moe_outputs)
        loss = data_loss + routing_loss

        self.default_log_dict({
            "train_loss": loss,
            "train_data_loss": data_loss,
            "train_routing_loss": routing_loss,
            "learning_rate": self.get_current_lr()
        })

        return loss

    def validation_step(
        self,
        batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        inp, tgt, cond = batch
        pred, _ = self.model(inp, cond)
        loss = self.criterion(pred, tgt)
        if batch_idx == 0:
            self.validation_sample = (inp.detach(), tgt.detach(), pred.detach())

        self.default_log_dict({"val_loss": loss})
    
        return loss
