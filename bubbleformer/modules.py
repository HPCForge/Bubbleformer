import random
import time
from typing import Tuple, Optional, List

import wandb
from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingLR
from einops import rearrange
import matplotlib.pyplot as plt
import lightning as L

from bubbleformer.data.batching import CollatedBatch
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
        batch: CollatedBatch,
        batch_idx: int
    ) -> torch.Tensor:
        inp, tgt, cond = batch.input, batch.target, batch.fluid_params_tensor

        # The input is [B, T, C, H, W]
        # Randomly flip along the horizontal axis of the input and target.
        if random.random() < 0.5:
            inp = torch.fliplr(inp)
            tgt = torch.fliplr(tgt)

        # Add gaussian noise to the input
        if random.random() < 0.4:
            # Since the data is unnormalized, we can use fairly large noise scales.
            scale = random.choice([0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
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
        batch: CollatedBatch,
        batch_idx: int
    ) -> torch.Tensor:
        inp, tgt, cond = batch.input, batch.target, batch.fluid_params_tensor
        pred, _ = self.model(inp, cond)
        loss = self.criterion(pred, tgt)
        if batch_idx == 0:
            self.validation_sample = (inp.detach(), tgt.detach(), pred.detach())

        self.default_log_dict({"val_loss": loss})

        return loss


class LatentRolloutModule(MoEConditionedForecastModule):
    """
    Training module for latent rollout. Instead of decoding back to pixel space
    at every autoregressive step, the processor predicts the next latent state
    directly. Training uses both a physical reconstruction loss and a latent
    consistency loss to keep processor outputs in-distribution for multi-step
    rollout.

    Set data_cfg.time_window = num_rollout_windows * actual_window_size.
    For example, time_window=20 with num_rollout_windows=4 gives 4 windows
    of 5 frames. Window 0 is the initial input; windows 1-3 are prediction
    targets.
    """
    def __init__(
        self,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        optim_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        num_rollout_windows: int = 4,
        latent_loss_ratio: float = 0.1,
        log_wandb: bool = False,
        normalization_constants: Tuple[List, List] = None,
    ):
        # The parent sets model time_window = data_cfg.time_window,
        # but we need the model to operate on individual windows.
        # Temporarily patch data_cfg.time_window for model construction.
        total_time_window = data_cfg.time_window
        window_size = total_time_window // num_rollout_windows
        assert total_time_window % num_rollout_windows == 0, (
            f"time_window ({total_time_window}) must be divisible by "
            f"num_rollout_windows ({num_rollout_windows})"
        )
        data_cfg.time_window = window_size

        super().__init__(
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            log_wandb=log_wandb,
            normalization_constants=normalization_constants,
        )

        # Restore time_window on both the internal dict and the original
        # DictConfig so nothing downstream sees the patched value.
        data_cfg.time_window = total_time_window
        self.data_cfg["time_window"] = total_time_window
        self.num_rollout_windows = num_rollout_windows
        self.window_size = window_size
        self.latent_loss_ratio = latent_loss_ratio

        # EMA buffers for adaptive lambda
        self.register_buffer("ema_physical", torch.tensor(1.0))
        self.register_buffer("ema_latent", torch.tensor(1.0))
        self.ema_initialized = False
        self.ema_decay = 0.99

    def _encode(self, x: torch.Tensor, fluid_params: torch.Tensor) -> torch.Tensor:
        """Encode physical fields to latent space.
        Args:
            x: (B, T, C, H, W) physical fields
            fluid_params: (B, num_fluid_params)
        Returns:
            z: (B, T, H_p, W_p, D) latent representation
        """
        B, T = x.shape[:2]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.model.embed(x)
        x = rearrange(x, "(b t) c h w -> b t h w c", t=T).contiguous()
        x = self.model.film_embed(x, fluid_params)
        return x

    def _process(self, z: torch.Tensor):
        """Predict next latent state from current latent state.
        Args:
            z: (B, T, H_p, W_p, D)
        Returns:
            z: (B, T, H_p, W_p, D) predicted next latent
            moe_outputs: list of MoE routing outputs
        """
        z_in = z
        z = z + self.model.mem_block_1(z)

        moe_outputs = []
        for i in range(len(self.model.blocks) // 2):
            z, moe_out = self.model.blocks[i](z)
            moe_outputs.append(moe_out)

        z = z + self.model.mem_block_2(z)

        for i in range(len(self.model.blocks) // 2, len(self.model.blocks)):
            z, moe_out = self.model.blocks[i](z)
            moe_outputs.append(moe_out)

        # Latent residual (replaces the old embed skip connection)
        z = z + z_in
        return z, moe_outputs

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to physical fields.
        No physical-space skip connection — the processor must learn
        the full mapping for latent rollout to work.
        Args:
            z: (B, T, H_p, W_p, D)
        Returns:
            x: (B, T, C, H, W) physical fields
        """
        T = z.shape[1]
        z = rearrange(z, "b t h w c -> (b t) c h w").contiguous()
        z = self.model.debed(z)
        z = rearrange(z, "(b t) c h w -> b t c h w", t=T)
        return z

    def training_step(
        self,
        batch: CollatedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        cond = batch.fluid_params_tensor   # (B, num_fluid_params)

        K = self.num_rollout_windows

        if K == 1:
            # Single latent step: encode input, process, decode, compare
            # against batch.target (no autoregressive rollout).
            inp = batch.input              # (B, T, C, H, W)
            tgt = batch.target             # (B, T, C, H, W)

            if random.random() < 0.5:
                inp = torch.fliplr(inp)
                tgt = torch.fliplr(tgt)
            if random.random() < 0.4:
                scale = random.choice([0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
                inp = inp + torch.normal(0, scale, inp.shape, device=inp.device)

            z = self._encode(inp, cond)
            z, moe_outputs = self._process(z)
            x_pred = self._decode(z)

            physical_loss = self.criterion(x_pred, tgt)

            with torch.no_grad():
                z_target = self._encode(tgt, cond)
            latent_loss = F.mse_loss(z, z_target)

            all_moe_outputs = moe_outputs
        else:
            inp = batch.input              # (B, total_T, C, H, W)

            # Data augmentation on the full sequence so all windows are
            # augmented consistently.
            if random.random() < 0.5:
                inp = torch.fliplr(inp)
            if random.random() < 0.4:
                scale = random.choice([0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
                inp = inp + torch.normal(0, scale, inp.shape, device=inp.device)

            # Chunk into K windows, each (B, window_size, C, H, W)
            windows = list(inp.chunk(K, dim=1))

            # Encode initial window
            z = self._encode(windows[0], cond)

            physical_loss = torch.tensor(0.0, device=inp.device)
            latent_loss = torch.tensor(0.0, device=inp.device)
            all_moe_outputs = []

            for k in range(1, K):
                z, moe_outputs = self._process(z)
                all_moe_outputs.extend(moe_outputs)

                # Physical reconstruction loss
                x_pred = self._decode(z)
                physical_loss = physical_loss + self.criterion(x_pred, windows[k])

                # Latent consistency loss (stop gradient on target)
                with torch.no_grad():
                    z_target = self._encode(windows[k], cond)
                latent_loss = latent_loss + F.mse_loss(z, z_target)

            physical_loss = physical_loss / (K - 1)
            latent_loss = latent_loss / (K - 1)

        # Adaptive lambda via EMA so both loss terms are on the same scale
        with torch.no_grad():
            p = physical_loss.detach()
            l = latent_loss.detach()
            if not self.ema_initialized:
                self.ema_physical.copy_(p)
                self.ema_latent.copy_(l)
                self.ema_initialized = True
            else:
                self.ema_physical.mul_(self.ema_decay).add_(p, alpha=1 - self.ema_decay)
                self.ema_latent.mul_(self.ema_decay).add_(l, alpha=1 - self.ema_decay)
            lam = self.latent_loss_ratio * self.ema_physical / (self.ema_latent + 1e-8)

        routing_loss = sum(m.load_balance_loss for m in all_moe_outputs)
        loss = physical_loss + lam * latent_loss + routing_loss

        self.default_log_dict({
            "train_loss": loss,
            "train_physical_loss": physical_loss,
            "train_latent_loss": latent_loss,
            "train_routing_loss": routing_loss,
            "latent_lambda": lam,
            "learning_rate": self.get_current_lr(),
        })

        return loss

    def validation_step(
        self,
        batch: CollatedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        cond = batch.fluid_params_tensor

        K = self.num_rollout_windows

        if K == 1:
            inp = batch.input
            tgt = batch.target

            z = self._encode(inp, cond)
            z, _ = self._process(z)
            x_pred = self._decode(z)

            total_loss = self.criterion(x_pred, tgt)

            if batch_idx == 0:
                self.validation_sample = (
                    inp.detach(),
                    tgt.detach(),
                    x_pred.detach(),
                )
        else:
            inp = batch.input
            windows = list(inp.chunk(K, dim=1))

            z = self._encode(windows[0], cond)

            total_loss = torch.tensor(0.0, device=inp.device)
            for k in range(1, K):
                z, _ = self._process(z)
                x_pred = self._decode(z)
                total_loss = total_loss + self.criterion(x_pred, windows[k])

            total_loss = total_loss / (K - 1)

            if batch_idx == 0:
                self.validation_sample = (
                    windows[0].detach(),
                    windows[-1].detach(),
                    x_pred.detach(),
                )

        self.default_log_dict({"val_loss": total_loss})
        return total_loss


class AutoregressiveModule(MoEConditionedForecastModule):
    """
    Training module for autoregressive prediction using the model's forward pass.

    Takes a sequence of frames, splits into chunks, and autoregressively predicts
    each chunk using the previous chunk as input. Loss is computed against ground
    truth for each prediction.

    For example with num_chunks=4 and 20 total frames:
    - Chunk 0 (frames 0-4): Initial input
    - Chunk 1 (frames 5-9): Predicted from chunk 0, loss vs GT
    - Chunk 2 (frames 10-14): Predicted from chunk 1, loss vs GT
    - Chunk 3 (frames 15-19): Predicted from chunk 2, loss vs GT

    Args:
        num_chunks: Number of chunks to split the sequence into
        use_teacher_forcing: If True, use GT as input for each step during training.
                            If False, use predictions autoregressively.
    """
    def __init__(
        self,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        optim_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        num_chunks: int = 4,
        use_teacher_forcing: bool = False,
        log_wandb: bool = False,
        normalization_constants: Tuple[List, List] = None,
    ):
        # Compute per-chunk time window for model construction
        total_time_window = data_cfg.time_window
        chunk_size = total_time_window // num_chunks
        assert total_time_window % num_chunks == 0, (
            f"time_window ({total_time_window}) must be divisible by "
            f"num_chunks ({num_chunks})"
        )
        data_cfg.time_window = chunk_size

        super().__init__(
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            log_wandb=log_wandb,
            normalization_constants=normalization_constants,
        )

        # Restore time_window
        data_cfg.time_window = total_time_window
        self.data_cfg["time_window"] = total_time_window
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.use_teacher_forcing = use_teacher_forcing

    def training_step(
        self,
        batch: CollatedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        inp = batch.input  # (B, total_T, C, H, W)
        cond = batch.fluid_params_tensor  # (B, num_fluid_params)

        K = self.num_chunks

        # Data augmentation on full sequence
        if random.random() < 0.5:
            inp = torch.fliplr(inp)
        if random.random() < 0.4:
            scale = random.choice([0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
            inp = inp + torch.normal(0, scale, inp.shape, device=inp.device)

        # Split into K chunks, each (B, chunk_size, C, H, W)
        chunks = list(inp.chunk(K, dim=1))

        total_loss = torch.tensor(0.0, device=inp.device)
        routing_loss = torch.tensor(0.0, device=inp.device)

        # Start with first chunk as input
        current_input = chunks[0]

        for k in range(1, K):
            # Predict next chunk using model's forward pass
            pred, moe_outputs = self.model(current_input, cond)

            # Loss against ground truth chunk
            total_loss = total_loss + self.criterion(pred, chunks[k])
            routing_loss = routing_loss + sum(m.load_balance_loss for m in moe_outputs)

            # Prepare input for next iteration
            if self.use_teacher_forcing:
                # Use ground truth for next prediction
                current_input = chunks[k]
            else:
                # Use prediction for next prediction (true autoregressive)
                # No detach: gradients flow through full autoregressive chain (BPTT)
                current_input = pred

        total_loss = total_loss / (K - 1)
        routing_loss = routing_loss / (K - 1)
        loss = total_loss + routing_loss

        self.default_log_dict({
            "train_loss": loss,
            "train_data_loss": total_loss,
            "train_routing_loss": routing_loss,
            "learning_rate": self.get_current_lr(),
        })

        return loss

    def validation_step(
        self,
        batch: CollatedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        inp = batch.input
        cond = batch.fluid_params_tensor

        K = self.num_chunks
        chunks = list(inp.chunk(K, dim=1))

        total_loss = torch.tensor(0.0, device=inp.device)
        current_input = chunks[0]

        all_preds = []

        for k in range(1, K):
            pred, _ = self.model(current_input, cond)
            total_loss = total_loss + self.criterion(pred, chunks[k])
            all_preds.append(pred)

            # Always use predictions for validation (true autoregressive eval)
            current_input = pred

        total_loss = total_loss / (K - 1)

        if batch_idx == 0:
            # Store first input, last GT, last prediction for visualization
            self.validation_sample = (
                chunks[0].detach(),
                chunks[-1].detach(),
                all_preds[-1].detach(),
            )

        self.default_log_dict({"val_loss": total_loss})
        return total_loss


class StatefulChunkModule(MoEConditionedForecastModule):
    """
    Training module with stateful S4 memory but WITHOUT autoregressive prediction.

    Unlike StatefulAutoregressiveModule, this class always uses ground truth as
    input for each chunk (teacher forcing). The SSM state is passed between chunks
    to maintain temporal memory, but predictions are never fed back as inputs.

    For example with num_chunks=4 and 20 total frames:
    - Chunk 0 (frames 0-4): Input, state=None -> state_0, predict chunk 1
    - Chunk 1 (frames 5-9): Input + state_0 -> state_1, predict chunk 2
    - Chunk 2 (frames 10-14): Input + state_1 -> state_2, predict chunk 3
    - Chunk 3 (frames 15-19): Input + state_2 -> state_3, (no prediction target)

    Loss is computed for each prediction vs the next ground truth chunk.

    Args:
        num_chunks: Number of chunks to split the sequence into
    """
    def __init__(
        self,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        optim_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        num_chunks: int = 4,
        log_wandb: bool = False,
        normalization_constants: Tuple[List, List] = None,
    ):
        # Compute per-chunk time window for model construction
        total_time_window = data_cfg.time_window
        chunk_size = total_time_window // num_chunks
        assert total_time_window % num_chunks == 0, (
            f"time_window ({total_time_window}) must be divisible by "
            f"num_chunks ({num_chunks})"
        )
        data_cfg.time_window = chunk_size

        super().__init__(
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            log_wandb=log_wandb,
            normalization_constants=normalization_constants,
        )

        # Restore time_window
        data_cfg.time_window = total_time_window
        self.data_cfg["time_window"] = total_time_window
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size

    def training_step(
        self,
        batch: CollatedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        inp = batch.input  # (B, total_T, C, H, W)
        cond = batch.fluid_params_tensor  # (B, num_fluid_params)

        K = self.num_chunks

        # Data augmentation on full sequence
        if random.random() < 0.5:
            inp = torch.fliplr(inp)
        #if random.random() < 0.4:
            #scale = random.choice([0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
            #inp = inp + torch.normal(0, scale, inp.shape, device=inp.device)

        # Split into K chunks, each (B, chunk_size, C, H, W)
        chunks = list(inp.chunk(K, dim=1))

        total_loss = torch.tensor(0.0, device=inp.device)
        routing_loss = torch.tensor(0.0, device=inp.device)

        # Initialize SSM states as None
        states = None

        for k in range(K - 1):
            # Forward pass with state - always use ground truth chunk as input
            pred, states, moe_outputs = self.model(chunks[k], cond, states)

            # Detach states to prevent gradient accumulation across chunks
            #states = [s.detach() for s in states]

            # Loss: prediction from chunk k should match chunk k+1
            total_loss = total_loss + self.criterion(pred, chunks[k + 1])
            routing_loss = routing_loss + sum(m.load_balance_loss for m in moe_outputs)

        total_loss = total_loss / (K - 1)
        routing_loss = routing_loss / (K - 1)
        loss = total_loss + routing_loss

        self.default_log_dict({
            "train_loss": loss,
            "train_data_loss": total_loss,
            "train_routing_loss": routing_loss,
            "learning_rate": self.get_current_lr(),
        })

        return loss

    def validation_step(
        self,
        batch: CollatedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        inp = batch.input
        cond = batch.fluid_params_tensor

        K = self.num_chunks
        chunks = list(inp.chunk(K, dim=1))

        total_loss = torch.tensor(0.0, device=inp.device)
        states = None
        last_pred = None

        for k in range(K - 1):
            pred, states, _ = self.model(chunks[k], cond, states)
            total_loss = total_loss + self.criterion(pred, chunks[k + 1])
            last_pred = pred

        total_loss = total_loss / (K - 1)

        if batch_idx == 0:
            self.validation_sample = (
                chunks[0].detach(),
                chunks[-1].detach(),
                last_pred.detach() if last_pred is not None else chunks[-1].detach(),
            )

        self.default_log_dict({"val_loss": total_loss})
        return total_loss


class StatefulAutoregressiveModule(MoEConditionedForecastModule):
    """
    Training module for autoregressive prediction with stateful S4 memory.

    The SSM state is passed between chunks, allowing the memory block to
    retain information across the full sequence during training.

    For example with num_chunks=4 and 20 total frames:
    - Chunk 0 (frames 0-4): Initial input, state=None -> state_0
    - Chunk 1 (frames 5-9): Input + state_0 -> state_1, loss vs GT
    - Chunk 2 (frames 10-14): Input + state_1 -> state_2, loss vs GT
    - Chunk 3 (frames 15-19): Input + state_2 -> state_3, loss vs GT

    The SSM state compresses the history of all previous frames, so each
    chunk's output is influenced by all frames seen so far.

    Args:
        num_chunks: Number of chunks to split the sequence into
        use_teacher_forcing: If True, use GT as input for each step.
                            If False, use predictions autoregressively.
    """
    def __init__(
        self,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        optim_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        num_chunks: int = 4,
        use_teacher_forcing: bool = False,
        log_wandb: bool = False,
        normalization_constants: Tuple[List, List] = None,
    ):
        # Compute per-chunk time window for model construction
        total_time_window = data_cfg.time_window
        chunk_size = total_time_window // num_chunks
        assert total_time_window % num_chunks == 0, (
            f"time_window ({total_time_window}) must be divisible by "
            f"num_chunks ({num_chunks})"
        )
        data_cfg.time_window = chunk_size

        super().__init__(
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            log_wandb=log_wandb,
            normalization_constants=normalization_constants,
        )

        # Restore time_window
        data_cfg.time_window = total_time_window
        self.data_cfg["time_window"] = total_time_window
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.use_teacher_forcing = use_teacher_forcing

    def training_step(
        self,
        batch: CollatedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        inp = batch.input  # (B, total_T, C, H, W)
        cond = batch.fluid_params_tensor  # (B, num_fluid_params)

        K = self.num_chunks

        # Data augmentation on full sequence
        if random.random() < 0.5:
            inp = torch.fliplr(inp)
        if random.random() < 0.4:
            scale = random.choice([0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
            inp = inp + torch.normal(0, scale, inp.shape, device=inp.device)

        # Split into K chunks, each (B, chunk_size, C, H, W)
        chunks = list(inp.chunk(K, dim=1))

        total_loss = torch.tensor(0.0, device=inp.device)
        routing_loss = torch.tensor(0.0, device=inp.device)

        # Initialize SSM states as None (will be zeros on first forward)
        states = None

        # Start with first chunk as input
        current_input = chunks[0]

        for k in range(1, K):
            # Forward pass WITH state - state carries through!
            pred, states, moe_outputs = self.model(current_input, cond, states)

            # Detach states to prevent gradient accumulation across chunks
            # This prevents OOM by not building a graph spanning all K chunks
            states = [s.detach() for s in states]

            # Loss against ground truth chunk
            total_loss = total_loss + self.criterion(pred, chunks[k])
            routing_loss = routing_loss + sum(m.load_balance_loss for m in moe_outputs)

            # Prepare input for next iteration
            if self.use_teacher_forcing:
                current_input = chunks[k]
            else:
                # Use prediction autoregressively (gradients flow through)
                current_input = pred

        total_loss = total_loss / (K - 1)
        routing_loss = routing_loss / (K - 1)
        loss = total_loss + routing_loss

        self.default_log_dict({
            "train_loss": loss,
            "train_data_loss": total_loss,
            "train_routing_loss": routing_loss,
            "learning_rate": self.get_current_lr(),
        })

        return loss

    def validation_step(
        self,
        batch: CollatedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        inp = batch.input
        cond = batch.fluid_params_tensor

        K = self.num_chunks
        chunks = list(inp.chunk(K, dim=1))

        total_loss = torch.tensor(0.0, device=inp.device)

        # Initialize states
        states = None
        current_input = chunks[0]

        all_preds = []

        for k in range(1, K):
            # Forward with state
            pred, states, _ = self.model(current_input, cond, states)
            total_loss = total_loss + self.criterion(pred, chunks[k])
            all_preds.append(pred)

            # Always use predictions for validation (true autoregressive eval)
            current_input = pred

        total_loss = total_loss / (K - 1)

        if batch_idx == 0:
            self.validation_sample = (
                chunks[0].detach(),
                chunks[-1].detach(),
                all_preds[-1].detach(),
            )

        self.default_log_dict({"val_loss": total_loss})
        return total_loss


class StatefulBPTTModule(MoEConditionedForecastModule):
    """
    Training module for stateful autoregressive prediction with gradient flow
    through SSM state using truncated backpropagation through time (TBPTT).

    Key features:
    - Gradients flow through SSM state (learns what to remember)
    - Scheduled sampling (gradual transition from teacher forcing to autoregressive)
    - Truncated BPTT with configurable truncation length

    Args:
        num_chunks: Number of chunks to split the sequence into
        teacher_forcing_ratio: Initial probability of using GT input (decays over training)
        tf_decay_steps: Number of steps to decay teacher forcing to min value
        tf_min_ratio: Minimum teacher forcing ratio after decay
        tbptt_steps: Number of chunks to backprop through before truncating
                     (None = full BPTT, 2 = truncate every 2 chunks)
    """
    def __init__(
        self,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        optim_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        num_chunks: int = 4,
        teacher_forcing_ratio: float = 1.0,
        tf_decay_steps: int = 10000,
        tf_min_ratio: float = 0.0,
        tbptt_steps: Optional[int] = 2,
        log_wandb: bool = False,
        normalization_constants: Tuple[List, List] = None,
    ):
        # Compute per-chunk time window for model construction
        total_time_window = data_cfg.time_window
        chunk_size = total_time_window // num_chunks
        assert total_time_window % num_chunks == 0, (
            f"time_window ({total_time_window}) must be divisible by "
            f"num_chunks ({num_chunks})"
        )
        data_cfg.time_window = chunk_size

        super().__init__(
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            log_wandb=log_wandb,
            normalization_constants=normalization_constants,
        )

        # Restore time_window
        data_cfg.time_window = total_time_window
        self.data_cfg["time_window"] = total_time_window
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size

        # Scheduled sampling parameters
        self.initial_tf_ratio = teacher_forcing_ratio
        self.tf_decay_steps = tf_decay_steps
        self.tf_min_ratio = tf_min_ratio

        # Truncated BPTT parameters
        self.tbptt_steps = tbptt_steps

    def get_teacher_forcing_ratio(self) -> float:
        """Compute current teacher forcing ratio with linear decay."""
        step = self.global_step
        if step >= self.tf_decay_steps:
            return self.tf_min_ratio

        decay_progress = step / self.tf_decay_steps
        return self.initial_tf_ratio - (self.initial_tf_ratio - self.tf_min_ratio) * decay_progress

    def training_step(
        self,
        batch: CollatedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        inp = batch.input  # (B, total_T, C, H, W)
        cond = batch.fluid_params_tensor  # (B, num_fluid_params)

        K = self.num_chunks
        tf_ratio = self.get_teacher_forcing_ratio()

        # Data augmentation on full sequence
        if random.random() < 0.5:
            inp = torch.flip(inp, dims=[-1])  # Horizontal flip
        if random.random() < 0.4:
            scale = random.choice([0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
            inp = inp + torch.randn_like(inp) * scale

        # Split into K chunks, each (B, chunk_size, C, H, W)
        chunks = list(inp.chunk(K, dim=1))

        total_loss = torch.tensor(0.0, device=inp.device)
        routing_loss = torch.tensor(0.0, device=inp.device)

        # Initialize SSM states as None
        states = None

        # Start with first chunk as input
        current_input = chunks[0]

        # Track autoregressive vs teacher forcing for logging
        autoreg_count = 0

        # Track state statistics across chunks
        state_norms = []
        state_means = []
        state_stds = []

        for k in range(1, K):
            # Forward pass
            pred, states, moe_outputs = self.model(current_input, cond, states)

            # Track state statistics (before potential detach)
            # Note: S4 state is complex-valued, so we use .abs() for real metrics
            if states is not None and len(states) > 0:
                state = states[0]  # (B*H*W, C, N) or similar
                state_abs = state.abs() if state.is_complex() else state
                state_norms.append(state_abs.norm().item())
                state_means.append(state_abs.mean().item())
                state_stds.append(state_abs.std().item())

            # Truncated BPTT: detach states periodically to limit gradient length
            # This prevents memory explosion while still allowing some gradient flow
            #if self.tbptt_steps is not None and k % self.tbptt_steps == 0:
                #states = [s.detach() for s in states]

            # Loss against ground truth chunk
            total_loss = total_loss + self.criterion(pred, chunks[k])
            routing_loss = routing_loss + sum(m.load_balance_loss for m in moe_outputs)

            # Scheduled sampling: probabilistically choose GT or prediction
            if random.random() < tf_ratio:
                # Teacher forcing: use ground truth
                current_input = chunks[k]
            else:
                # Autoregressive: use prediction (gradients flow through)
                current_input = pred
                autoreg_count += 1

        total_loss = total_loss / (K - 1)
        routing_loss = routing_loss / (K - 1)
        loss = total_loss + routing_loss

        # Compute state statistics
        state_metrics = {}
        if len(state_norms) > 0:
            state_metrics["state/norm_mean"] = sum(state_norms) / len(state_norms)
            state_metrics["state/norm_final"] = state_norms[-1]
            state_metrics["state/mean"] = sum(state_means) / len(state_means)
            state_metrics["state/std"] = sum(state_stds) / len(state_stds)
            # Track how much state changes across chunks
            if len(state_norms) > 1:
                state_metrics["state/norm_growth"] = state_norms[-1] - state_norms[0]

        self.default_log_dict({
            "train_loss": loss,
            "train_data_loss": total_loss,
            "train_routing_loss": routing_loss,
            "teacher_forcing_ratio": tf_ratio,
            "autoreg_steps": autoreg_count,
            "learning_rate": self.get_current_lr(),
            **state_metrics,
        })

        return loss

    def validation_step(
        self,
        batch: CollatedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Validation always uses full autoregressive rollout (no teacher forcing)
        to accurately measure generalization.
        """
        inp = batch.input
        cond = batch.fluid_params_tensor

        K = self.num_chunks
        chunks = list(inp.chunk(K, dim=1))

        total_loss = torch.tensor(0.0, device=inp.device)

        states = None
        current_input = chunks[0]
        all_preds = []
        val_state_norms = []

        for k in range(1, K):
            # Forward with state (no checkpointing needed for eval)
            pred, states, _ = self.model(current_input, cond, states)
            total_loss = total_loss + self.criterion(pred, chunks[k])
            all_preds.append(pred)

            # Track validation state norms (S4 state is complex)
            if states is not None and len(states) > 0:
                state = states[0]
                state_abs = state.abs() if state.is_complex() else state
                val_state_norms.append(state_abs.norm().item())

            # Always autoregressive for validation
            current_input = pred

        total_loss = total_loss / (K - 1)

        # Log validation state metrics
        if len(val_state_norms) > 0:
            self.default_log_dict({
                "val_state/norm_mean": sum(val_state_norms) / len(val_state_norms),
                "val_state/norm_final": val_state_norms[-1],
            })

        if batch_idx == 0:
            self.validation_sample = (
                chunks[0].detach(),
                chunks[-1].detach(),
                all_preds[-1].detach(),
            )

        self.default_log_dict({"val_loss": total_loss})
        return total_loss
