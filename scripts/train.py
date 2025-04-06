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
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelSummary, Callback
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.strategies import DDPStrategy

from bubbleformer.data import BubblemlForecast
from bubbleformer.modules import ForecastModule
from bubbleformer.utils.param_counter import print_model_stats

def is_leader_process():
    """
    Check if the current process is the leader process.
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
            # Save the checkpoint. Use trainer.save_checkpoint if accessible.
            # Note: You might need to call this on the main thread.
            self.trainer.save_checkpoint(self.checkpoint_path)
            print(f"Due to preemption Checkpoint saved to {self.checkpoint_path}.")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
        # Optionally, delay a bit to ensure the checkpoint save finishes.
        time.sleep(5)

@hydra.main(version_base=None, config_path="../bubbleformer/config", config_name="default")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    params = {}
    params["distributed"] = cfg.distributed
    params["nodes"] = cfg.nodes
    params["devices"] = cfg.devices
    params["checkpoint_path"] = cfg.checkpoint_path
    params["data_cfg"] = cfg.data_cfg
    params["model_cfg"] = cfg.model_cfg
    params["optim_cfg"] =  cfg.optim_cfg
    params["scheduler_cfg"] =  cfg.scheduler_cfg

    # if cfg.max_epochs<=400, set Full epoch flag to empty string, else set to "Full_epoch_"
    full_epoch_flag = "" if cfg.max_epochs <= 400 else "Full_epoch_"
    
    if params["checkpoint_path"] is None:
        log_id = (
            # "Modified_Combined_torchcompiled" +
            # "Modified_Test_" +
            # "Test5_Counter_Test_" +
            # "Saturated_MPP_baseline_" +
            full_epoch_flag +
            "RED_" + str(cfg.model_cfg.params.routed_expert_embed_dim) + "_" +
            "SED_" + str(cfg.model_cfg.params.shared_expert_embed_dim) + "_" +
            "ST_" + str(cfg.model_cfg.params.shared_expert_type) + "_" +
            "E_" + str(cfg.model_cfg.params.n_experts) + "_" +
            "S_" + str(cfg.model_cfg.params.n_shared_experts) + "_" +
            "A_" + str(cfg.model_cfg.params.top_k) + "_" +
            cfg.model_cfg.name.lower() + "_"
            + cfg.data_cfg.dataset.lower() + "_"
            + os.getenv("SLURM_JOB_ID")
        )
        params["log_dir"] = os.path.join(cfg.log_dir, log_id)
        os.makedirs(params["log_dir"], exist_ok=True)
        preempt_ckpt_path = params["log_dir"] + "/hpc_ckpt_1.ckpt"
    else:
        # Extract the log_id from the checkpoint path
        # Handle both custom checkpoints and Lightning checkpoints
        checkpoint_parts = cfg.checkpoint_path.split("/")
        if "lightning_logs" in cfg.checkpoint_path:
            # This is a Lightning checkpoint
            log_id = checkpoint_parts[-5]  # Adjust based on your path structure
            params["log_dir"] = "/".join(cfg.checkpoint_path.split("/")[:-4])
            preempt_ckpt_num = 1  # Start with a new preemption checkpoint number
        else:
            # This is a custom checkpoint (hpc_ckpt)
            log_id = checkpoint_parts[-2]
            params["log_dir"] = "/".join(cfg.checkpoint_path.split("/")[:-1])
            try:
                preempt_ckpt_num = int(cfg.checkpoint_path.split("_")[-1][:-5]) + 1
            except ValueError:
                # If we can't parse the number, just use a new number
                preempt_ckpt_num = 1
        
        preempt_ckpt_path = params["log_dir"] + "/hpc_ckpt_" + str(preempt_ckpt_num) + ".ckpt"

    logger = CSVLogger(save_dir=params["log_dir"])

    train_dataset = BubblemlForecast(
                filenames=cfg.data_cfg.train_paths,
                fields=cfg.data_cfg.fields,
                norm=cfg.data_cfg.normalize,
                time_window=cfg.data_cfg.time_window,
            )
    normalization_constants = train_dataset.normalize()
    val_dataset = BubblemlForecast(
                filenames=cfg.data_cfg.val_paths,
                fields=cfg.data_cfg.fields,
                norm=cfg.data_cfg.normalize,
                time_window=cfg.data_cfg.time_window,
            )
    val_dataset.normalize(*normalization_constants)
    diff_term = normalization_constants[0].tolist()
    div_term = normalization_constants[1].tolist()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    train_module = ForecastModule(
                model_cfg=cfg.model_cfg,
                data_cfg=cfg.data_cfg,
                optim_cfg=cfg.optim_cfg,
                scheduler_cfg=cfg.scheduler_cfg,
                log_wandb=cfg.use_wandb,
                normalization_constants=(diff_term, div_term),
            )

    # Print model parameter statistics
    print_model_stats(train_module.model)

    # trainer = Trainer(
    #     accelerator="gpu",
    #     devices=cfg.devices,
    #     num_nodes=cfg.nodes,
    #     strategy="ddp",
    #     max_epochs=cfg.max_epochs,
    #     logger=logger,
    #     default_root_dir=params["log_dir"],
    #     plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
    #     enable_model_summary=True,
    #     limit_train_batches=500,
    #     limit_val_batches=50,
    #     num_sanity_val_steps=0,
    #     callbacks=[ModelSummary(max_depth=-1), PreemptionCheckpointCallback(preempt_ckpt_path)]
    # )
    
    # for with MoE
    trainer = Trainer(
        accelerator="gpu",
        devices=cfg.devices,
        num_nodes=cfg.nodes,
        # strategy="ddp",
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=cfg.max_epochs,
        logger=logger,
        default_root_dir=params["log_dir"],
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
        enable_model_summary=True,
        limit_train_batches=50,
        limit_val_batches=50,
        num_sanity_val_steps=0,
        callbacks=[ModelSummary(max_depth=-1), PreemptionCheckpointCallback(preempt_ckpt_path)]
    )

    if is_leader_process():
        pp = pprint.PrettyPrinter(depth=4)
        pp.pprint(params)

    wandb_run = None
    if cfg.use_wandb and is_leader_process(): # Load only one wandb run
        try:
            # wandb_key_path = "bubbleformer/config/wandb_api_key.txt"
            # with open(wandb_key_path, "r", encoding="utf-8") as f:
            #     wandb_key = f.read().strip()
            # wandb.login(key=wandb_key)
            wandb_run = wandb.init(
                project="bubbleformer",
                name=log_id,
                dir=params["log_dir"],
                tags=cfg.wandb_tags,
                config=params,
                resume="auto",
            )
        except FileNotFoundError as e:
            print(e)
            print("Valid wandb API key not found at path bubbleformer/config/wandb_api_key.txt")

    if cfg.checkpoint_path:
        trainer.fit(
            train_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.checkpoint_path
        )
    else:
        trainer.fit(
            train_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
