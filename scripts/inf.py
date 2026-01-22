import os
import torch
from collections import OrderedDict
from bubbleformer.models import get_model
from bubbleformer.data import BubbleForecast
from bubbleformer.layers.moe.topk_moe import TopkMoEOutput
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import cv2
import numpy as np
from bubbleformer.utils.moe_metrics import topk_indices_to_patch_expert_counts
from bubbleformer.utils.physical_metrics import (
    physical_metrics,
    bubble_metrics
)

torch.set_float32_matmul_precision("high")

test_paths = [
    #"/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D/Twall_97.hdf5",
    "/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-R515B-2D/Twall_30.hdf5",
    #"/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-LN2-2D/Twall_-165.hdf5",
    #"/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Saturated-FC72-2D/Twall_91.hdf5",
    #"/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Saturated-R515B-2D/Twall_18.hdf5",
    #"/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Saturated-LN2-2D/Twall_-176.hdf5",
]

# TODO: This should all be written/read to/from a config file with the checkpoints
model_name = "neighbor_moe"
model_kwargs = {
    "input_fields": 4,
    "output_fields": 4,
    "time_window": 5,
    "patch_size": 4,
    "embed_dim": 384,
    "processor_blocks": 6,
    "num_heads": 6,
    "num_experts": 6,
    "topk": 2,
    "load_balance_loss_weight": 0.01,
    "num_fluid_params": 9,
}

model = get_model(model_name, **model_kwargs)
model = model.cuda()
#weights_path = "/pub/afeeney/bubbleformer_logs/filmavit_poolboiling_subcooled_47238340/checkpoints/epoch=29-step=56760.ckpt"
#weights_path = "/pub/afeeney/bubbleformer_logs/neighbor_moe_poolboiling_subcooled_47407258/checkpoints/epoch=34-step=132440.ckpt"
weights_path = "/pub/afeeney/bubbleformer_logs/neighbor_moe_poolboiling_subcooled_47512802/checkpoints/last.ckpt"
#weights_path = "/pub/afeeney/bubbleformer_logs/neighbor_moe_poolboiling_subcooled_47609426/checkpoints/last.ckpt"
model_data = torch.load(weights_path, weights_only=False)
weight_state_dict = OrderedDict()
for key, val in model_data["state_dict"].items():
    name = key[6:]
    weight_state_dict[name] = val
del model_data
model.load_state_dict(weight_state_dict)
model.eval()

print(model)

def run_test(model, test_file_path: str, max_timesteps: int):

    downsample_factor = 8
    test_dataset = BubbleForecast(
        filenames=[test_file_path],
        input_fields=["dfun", "temperature", "velx", "vely"],
        output_fields=["dfun", "temperature", "velx", "vely"],
        norm="none",    
        downsample_factor=downsample_factor,
        time_window=5,
        start_time=100,
        return_fluid_params=True,
    )

    start_time = test_dataset.start_time
    skip_itrs = test_dataset.time_window
    preds = []
    targets = []
    timesteps = []
    moe_outputs = []

    for itr in range(0, max_timesteps, skip_itrs):
        if itr % 20 == 0:
            print(f"Autoreg pred {itr}, [{start_time+itr}, {start_time+itr+skip_itrs}] to [{start_time+itr+skip_itrs}, {start_time+itr+2*skip_itrs}]")
        
        inp, tgt, fluid_params = test_dataset[itr]  
        if len(preds) > 0:
            inp = preds[-1]
            
        inp = inp.cuda().to(torch.float32).unsqueeze(0)
        fluid_params = fluid_params.cuda().to(torch.float32).unsqueeze(0)
        
        pred, moe_output = model(inp, fluid_params)
        moe_outputs.append(moe_output[0]) # NOTE: tracking first layer of MoE outputs

        pred = pred.to(torch.float32).squeeze(0)
        pred = pred.detach().cpu()
        tgt = tgt.detach().cpu()

        preds.append(pred)
        targets.append(tgt)
        timesteps.append(torch.arange(start_time+itr+skip_itrs, start_time+itr+2*skip_itrs))

    preds = torch.cat(preds, dim=0)[None, ...]         # 1, T, C, H, W
    targets = torch.cat(targets, dim=0)[None, ...]     # 1, T, C, H, W
    timesteps = torch.cat(timesteps, dim=0)             # T,

    topk_indices = [moe_output.topk_indices.squeeze(0) for moe_output in moe_outputs]
    topk_indices = torch.cat(topk_indices, dim=0) # (T, H, W, topk)
    
    sdf_target = targets[:, :, 0, :, :]
    sdf_pred = preds[:, :, 0, :, :]
    
    #norm = TwoSlopeNorm(vcenter=0)
    #plt.imshow(sdf_target[0, 0, :, :], cmap="icefire", norm=norm, origin="lower")
    #plt.savefig("sdf.png")
    #plt.close()
    
    #bubbles = find_bubbles(sdf_target)[0, 50]
    #plt.imshow(bubbles, cmap=plt.cm.nipy_spectral, origin="lower")
    #plt.savefig("bubbles.png")
    #plt.close()å
    
    print("-"*100)
    print(f"Rollout Statistics on {test_file_path}:")
    dx = 1/32 * downsample_factor
    dy = dx
    p = physical_metrics(
        preds[:, :, 0], 
        preds[:, :, 1], 
        preds[:, :, 2], 
        preds[:, :, 3],
        # TODO: get these from dataset
        heater_min=-5.25,
        heater_max=5.25,
        xcoords=torch.linspace(-8 + dx / 2, 8 - dx / 2, 512 // downsample_factor), 
        dx=dx, 
        dy=dy
    )
    b = bubble_metrics(sdf_target, preds[:, :, 2], preds[:, :, 3], dx=dx, dy=dy)
    print(p)
    print(b)
    
for test_file_path in test_paths:
    run_test(model, test_file_path, max_timesteps=100)

#save_dir = "./subcooled_fc72_97"
#print(f"saving to {save_dir}")

#os.makedirs(save_dir, exist_ok=True)
#save_path = os.path.join(save_dir, "predictions.pt")
#torch.save({"preds": model_preds, "targets": model_targets, "timesteps": timesteps}, save_path)
#plot_bubbleml(model_preds, model_targets, topk_indices, timesteps, save_dir)