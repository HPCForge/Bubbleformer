import os
import pprint
import time
import signal

import hydra
import wandb
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from lightning import seed_everything, Trainer
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelSummary, Callback
from lightning.pytorch.plugins.environments import SLURMEnvironment

from bubbleformer.data import BubbleForecast, VariableInputBubbleForecast, collate_random_variable
from bubbleformer.modules import ForecastModule, ConditionedForecastModule
from bubbleformer.models.axial_vit import SpaceTimeBlock


def checkpoint_policy(module, **kwargs):
    return isinstance(module, SpaceTimeBlock)


def is_leader_process():
    """
    Check if the current process is the leader process.
    (Only used for pretty-printing params now.)
    """
    if os.getenv("SLURM_PROCID") is None:
        if os.getenv("LOCAL_RANK") is not None:
            return int(os.getenv("LOCAL_RANK")) == 0
        else:
            return True
    else:
        return os.getenv("SLURM_PROCID") == "0"


class PreemptionCheckpointCallback(Callback):
    """
    Tries to save a checkpoint when a SIGTERM signal is received.
    Args:
        checkpoint_path: Path to save the checkpoint.
    """
    def __init__(self, checkpoint_path="preemption_checkpoint.ckpt"):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.already_handled = False

    def setup(self, trainer, pl_module, stage: str) -> None:
        self.trainer = trainer
        # Register the signal handler for SIGTERM in case of job preemption due to paid job
        signal.signal(signal.SIGTERM, self.handle_preemption)

    def handle_preemption(self, signum, frame):
        """
        Handle the SIGTERM signal.
        """
        if self.already_handled:
            return
        self.already_handled = True
        try:
            self.trainer.save_checkpoint(self.checkpoint_path)
            print(f"Due to preemption checkpoint saved to {self.checkpoint_path}.")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
        time.sleep(5)


@hydra.main(version_base=None, config_path="../bubbleformer/config", config_name="default")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    # ---- Params dict for logging / wandb config ----
    params = {}
    params["nodes"] = cfg.nodes
    params["devices"] = cfg.devices
    params["checkpoint_path"] = cfg.checkpoint_path
    params["data_cfg"] = cfg.data_cfg
    params["model_cfg"] = cfg.model_cfg
    params["optim_cfg"] = cfg.optim_cfg
    params["scheduler_cfg"] = cfg.scheduler_cfg

    # ---- Log directory / checkpoint path setup ----
    if params["checkpoint_path"] is None:
        # Starting fresh (no resume)
        log_id = (
            cfg.model_cfg.name.lower() + "_"
            + cfg.data_cfg.dataset.lower() + "_"
            + os.getenv("SLURM_JOB_ID")
        )
        params["log_dir"] = os.path.join(cfg.log_dir, log_id)
        os.makedirs(params["log_dir"], exist_ok=True)

        # First HPC checkpoint if training from scratch
        preempt_ckpt_path = os.path.join(params["log_dir"], "hpc_ckpt_1.ckpt")
    else:
        # Resuming from an existing lightning checkpoint
        # Log dir is parent path of the checkpoint
        log_id = cfg.checkpoint_path.split("/")[-2]
        params["log_dir"] = "/".join(cfg.checkpoint_path.split("/")[:-1])

        # Extract epoch number from checkpoint filename safely
        import re
        match = re.search(r"epoch=(\d+)", cfg.checkpoint_path)
        if match:
            preempt_ckpt_num = int(match.group(1)) + 1
        else:
            # fallback if lightning name didn't contain epoch=
            preempt_ckpt_num = 1

        preempt_ckpt_path = os.path.join(
            params["log_dir"], f"hpc_ckpt_{preempt_ckpt_num}.ckpt"
        )

    # ---- Logger setup (W&B or CSV) ----
    if cfg.use_wandb:
        # Optional: login via API key file, needed on some clusters
        try:
            wandb_key_path = "bubbleformer/config/wandb_api_key.txt"
            with open(wandb_key_path, "r", encoding="utf-8") as f:
                wandb_key = f.read().strip()
            wandb.login(key=wandb_key)
        except FileNotFoundError as e:
            print(e)
            print("Valid wandb API key not found at path bubbleformer/config/wandb_api_key.txt")

        logger = WandbLogger(
            project="Bubbleformer",
            name=log_id,
            save_dir=params["log_dir"],
            tags=getattr(cfg, "wandb_tags", None),
            config=params,
        )
    else:
        logger = CSVLogger(save_dir=params["log_dir"])
    
    """
    # ---- Datasets / normalization ----
    train_dataset = BubbleForecast(
        filenames=cfg.data_cfg.train_paths,
        input_fields=cfg.data_cfg.input_fields,
        output_fields=cfg.data_cfg.output_fields,
        norm=cfg.data_cfg.normalize,
        downsample_factor=cfg.data_cfg.downsample_factor,
        time_window=cfg.data_cfg.time_window,
        start_time=cfg.data_cfg.start_time,
        return_fluid_params=cfg.data_cfg.return_fluid_params,
    )
    normalization_constants = train_dataset.normalize()

    val_dataset = BubbleForecast(
        filenames=cfg.data_cfg.val_paths,
        input_fields=cfg.data_cfg.input_fields,
        output_fields=cfg.data_cfg.output_fields,
        norm=cfg.data_cfg.normalize,
        downsample_factor=cfg.data_cfg.downsample_factor,
        time_window=cfg.data_cfg.time_window,
        start_time=cfg.data_cfg.start_time,
        return_fluid_params=cfg.data_cfg.return_fluid_params,
    )
    val_dataset.normalize(*normalization_constants)

    diff_term, div_term = normalization_constants
    """
    print("making dataset")
    train_dataset = VariableInputBubbleForecast(
        filenames=cfg.data_cfg.train_paths,
        input_fields=cfg.data_cfg.input_fields,
        output_fields=cfg.data_cfg.output_fields,
        norm=cfg.data_cfg.normalize,
        downsample_factor=cfg.data_cfg.downsample_factor,
        max_input_window=cfg.data_cfg.input_window,
        pred_window=cfg.data_cfg.output_window,
        start_time=cfg.data_cfg.start_time,
        return_fluid_params=cfg.data_cfg.return_fluid_params,
    )
    
    normalization_constants = train_dataset.normalize()

    val_dataset = VariableInputBubbleForecast(
        filenames=cfg.data_cfg.val_paths,
        input_fields=cfg.data_cfg.input_fields,
        output_fields=cfg.data_cfg.output_fields,
        norm=cfg.data_cfg.normalize,
        downsample_factor=cfg.data_cfg.downsample_factor,
        max_input_window=cfg.data_cfg.input_window,
        pred_window=cfg.data_cfg.output_window,
        start_time=cfg.data_cfg.start_time,
        return_fluid_params=cfg.data_cfg.return_fluid_params,
    )

    val_dataset.normalize(*normalization_constants)

    diff_term, div_term = normalization_constants
    print("done with making dataset")

    # ---- Dataloaders ----
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_random_variable
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_random_variable
    )

    # ---- LightningModule (with or without conditioning) ----
    if cfg.data_cfg.return_fluid_params:
        train_module = ConditionedForecastModule(
            model_cfg=cfg.model_cfg,
            data_cfg=cfg.data_cfg,
            optim_cfg=cfg.optim_cfg,
            scheduler_cfg=cfg.scheduler_cfg,
            log_wandb=cfg.use_wandb,
            normalization_constants=(diff_term, div_term),
        )
    else:
        train_module = ForecastModule(
            model_cfg=cfg.model_cfg,
            data_cfg=cfg.data_cfg,
            optim_cfg=cfg.optim_cfg,
            scheduler_cfg=cfg.scheduler_cfg,
            log_wandb=cfg.use_wandb,
            normalization_constants=(diff_term, div_term),
        )

    # ---- Trainer ----
    trainer = Trainer(
        accelerator="auto",
        devices=cfg.devices,
        num_nodes=cfg.nodes,
        strategy="ddp",
        max_epochs=cfg.max_epochs,
        logger=logger,
        default_root_dir=params["log_dir"],
        # plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
        enable_model_summary=True,
        limit_train_batches=5000,
        limit_val_batches=25,
        num_sanity_val_steps=0,
        callbacks=[
            ModelSummary(max_depth=-1),
            PreemptionCheckpointCallback(preempt_ckpt_path),
        ],
    )

    # ---- Pretty-print params on leader process ----
    if is_leader_process():
        pp = pprint.PrettyPrinter(depth=4)
        pp.pprint(params)

    # ---- Train (optionally resuming from checkpoint) ----
    if cfg.checkpoint_path:
        trainer.fit(
            train_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.checkpoint_path,
        )
    else:
        trainer.fit(
            train_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()

