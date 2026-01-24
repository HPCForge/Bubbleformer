import os
import torch
from collections import OrderedDict
from bubbleformer.models import get_model
from bubbleformer.data import BubbleForecast
from bubbleformer.utils.losses import LpLoss
from bubbleformer.layers.moe.topk_moe import TopkMoEOutput
import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import List
from bubbleformer.utils.moe_metrics import topk_indices_to_patch_expert_counts
from bubbleformer.utils.sdf_reinit import fast_marching_2d
from bubbleformer.utils.physical_metrics import eikonal

def plot_bubbleml(
        preds: torch.Tensor,
        targets: torch.Tensor,
        topk_indices: List[List[TopkMoEOutput]],
        timesteps: torch.Tensor,
        save_dir: str,
    ):
    """
    Plot the BubbleML predictions, targets and errors for each timestep
    Also plots the relative L2 error over time for each channel
    Args:
        preds: Predictions from the model for a single rollout (T, 4, H, W)
        targets: Ground truth targets for a single rollout (T, 4, H, W)
        topk_indices: MoE routing for a every step of a rollout (T, H, W, topk)
        timesteps: Timesteps for the predictions for a single rollout (T,)
        save_dir: Directory to save the plots
    """
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Compute the L2 norm across spatial dimensions (h, w)
    diff_norm = torch.norm(preds - targets, p=2, dim=(2, 3))  # Shape (t, c)
    bnorm = torch.norm(targets, p=2, dim=(2, 3))        # Shape (t, c)

    relative_l2_error = diff_norm / bnorm
    
    print("L2 error: ", diff_norm)
    print("Relative L2 error: ", relative_l2_error)

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps.numpy(), relative_l2_error[:, 0].numpy(), label="SDF")
    plt.plot(timesteps.numpy(), relative_l2_error[:, 1].numpy(), label="Temp")
    plt.plot(timesteps.numpy(), relative_l2_error[:, 2].numpy(), label="VelX")
    plt.plot(timesteps.numpy(), relative_l2_error[:, 3].numpy(), label="VelY")

    plt.xlabel("Time (timesteps)")
    plt.ylabel("Relative L2 Error")
    plt.title("Relative L2 Error over Time for Each Variable")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "relative_l2_error.png"))

    vel_mag = torch.sqrt(targets[:, 2] ** 2 + targets[:, 3] ** 2)
    vel_mean, vel_std = torch.mean(vel_mag).item(), torch.std(vel_mag).item()
    vel_min, vel_max = round(vel_mean - 3 * vel_std, 2), round(vel_mean + 3 * vel_std, 2)

    temp_mean, temp_std = torch.mean(targets[:, 1]).item(), torch.std(targets[:, 1]).item()
    temp_min, temp_max = targets[:, 1].min().item(), targets[:, 1].max().item() #-28, 30 
    print(preds[..., 1, :, :].min(), preds[..., 1, :, :].max())
    print(targets[..., 1, :, :].min(), targets[..., 1, :, :].max())
    #temp_min, temp_max = 50.0, 100.0

    sdf_mean, sdf_std = torch.mean(targets[:, 0]).item(), torch.std(targets[:, 0]).item()
    sdf_min, sdf_max = round(sdf_mean - 3 * sdf_std, 2), round(sdf_mean + 3 * sdf_std, 2)
    
    # (num_experts, T, H, W)
    patch_expert_counts = topk_indices_to_patch_expert_counts(
        topk_indices, topk_indices.max().item() + 1)
    patch_expert0_counts = patch_expert_counts[1] # (T, H, W)

    for i in range(preds.shape[0]):
        
        i_str = str(i).zfill(4)
        
        patch_expert0_count_at_i = patch_expert0_counts[i] # (H, W)
        
        # interpolate the patch expert counts to the full grid, so we can plot it
        patch_expert0_count_at_i = torch.nn.functional.interpolate(
            patch_expert0_count_at_i.unsqueeze(0).unsqueeze(0).to(torch.float32),
            size=(preds.shape[2], preds.shape[3]),
            mode="nearest"
        ).squeeze().to(torch.int32).detach().cpu().numpy()

        sdf_pred = preds[i, 0, :, :].numpy()
        dfun_pred = sdf_pred.copy()
        dfun_pred[dfun_pred>0] = 0
        dfun_pred[dfun_pred<0] = 255
        dfun_pred = dfun_pred.astype(np.uint8)
        pred_edge_map = cv2.Canny(dfun_pred, 0, 255)
        kernel = np.ones((3,3),np.uint8)
        pred_edge_map = cv2.dilate(pred_edge_map, kernel, iterations=1)
        pred_mask = np.where(pred_edge_map > 0, 0, 255)
        pred_alpha = np.where(pred_mask > 0, 0, 255)
        pred_overlay = np.dstack((pred_mask, pred_mask, pred_mask, pred_alpha))

        sdf_target = targets[i, 0, :, :].numpy()
        dfun_target = sdf_target.copy()
        dfun_target[dfun_target>0] = 0
        dfun_target[dfun_target<0] = 255
        dfun_target = dfun_target.astype(np.uint8)
        target_edge_map = cv2.Canny(dfun_target, 0, 255)
        target_edge_map = cv2.dilate(target_edge_map, kernel, iterations=1)
        target_mask = np.where(target_edge_map > 0, 0, 255)
        target_alpha = np.where(target_mask > 0, 0, 255)
        target_overlay = np.dstack((target_mask, target_mask, target_mask, target_alpha))
        dfun_err = np.abs(sdf_target - sdf_pred)/(np.abs(sdf_target) + 1.0e-8)

        temp_pred = preds[i, 1, :, :].numpy()
        temp_target = targets[i, 1, :, :].numpy()
        temp_err = np.abs(temp_target - temp_pred)/(np.abs(temp_target) + 1.0e-8)

        velx_pred = preds[i, 2, :, :].numpy()
        velx_target = targets[i, 2, :, :].numpy()
        vely_pred = preds[i, 3, :, :].numpy()
        vely_target = targets[i, 3, :, :].numpy()

        velmag_pred = np.sqrt(velx_pred**2 + vely_pred**2)
        velmag_target = np.sqrt(velx_target**2 + vely_target**2)
        velmag_err = np.abs(velmag_target - velmag_pred)/(np.abs(velmag_target) + 1.0e-8)

        # Velocity Streamlines
        x = np.arange(10, velmag_target.shape[1]-10, 1)
        y = np.arange(10, velmag_target.shape[0]-10, 1)
        X,Y = np.meshgrid(x,y)
        velx_pred[dfun_target==0] = 0
        vely_pred[dfun_target==0] = 0
        velx_target[dfun_target==0] = 0
        vely_target[dfun_target==0] = 0

        f, axarr = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
        im_00 = axarr[0][0].imshow(sdf_target, vmin=sdf_min, vmax=sdf_max, cmap="Blues", origin="lower")
        #axarr[0][0].imshow(target_overlay, alpha=1, origin="lower")
        axarr[0][0].axis("off")
        plt.colorbar(im_00, ax=axarr[0][0], fraction=0.04, pad=0.05)
        axarr[0][0].set_title(f"SDF Label {i}")

        im_01 = axarr[0][1].imshow(temp_target, vmin=temp_min, vmax=temp_max, origin="lower")
        #im_01 = axarr[0][1].imshow(temp_target, cmap="turbo", origin="lower")
        #axarr[0][1].imshow(target_overlay, alpha=1, origin="lower")
        axarr[0][1].axis("off")
        plt.colorbar(im_01, ax=axarr[0, 1], fraction=0.04, pad=0.05)
        axarr[0][1].set_title(f"Temp Label {i}")

        im_02 = axarr[0][2].imshow(np.flipud(velmag_target), vmin=vel_min, vmax=vel_max, cmap="turbo")
        #im_02 = axarr[0][2].imshow(np.flipud(velmag_target), cmap="turbo")
        #axarr[0][2].streamplot(X, Y, np.flipud(velx_target)[10:-10, 10:-10], -np.flipud(vely_target)[10:-10, 10:-10], density=0.75, color="white")
        #axarr[0][2].imshow(np.flipud(target_overlay), alpha=1)
        axarr[0][2].axis("off")
        plt.colorbar(im_02, ax=axarr[0][2], fraction=0.04, pad=0.05)
        axarr[0][2].set_title(f"Vel Label {i}")

        im_10 = axarr[1][0].imshow(sdf_pred, vmin=sdf_min, vmax=sdf_max, cmap="Blues", origin="lower")
        #axarr[1][0].imshow(pred_overlay, alpha=1, origin="lower")
        axarr[1][0].axis("off")
        plt.colorbar(im_10, ax=axarr[1][0], fraction=0.04, pad=0.05)
        axarr[1][0].set_title(f"SDF Pred {i}")

        im_11 = axarr[1][1].imshow(temp_pred, vmin=temp_min, vmax=temp_max, origin="lower")
        # NOTE: Overlay MoE expert counts on top of the temperature prediction
        #axarr[1][1].imshow(patch_expert0_count_at_i, alpha=0.3, origin="lower")

        #im_11 = axarr[1][1].imshow(temp_pred, cmap="turbo", origin="lower")
        #axarr[1][1].imshow(pred_overlay, alpha=1, origin="lower")
        axarr[1][1].axis("off")
        plt.colorbar(im_11, ax=axarr[1, 1], fraction=0.04, pad=0.05)
        axarr[1][1].set_title(f"Temp Pred {i}")

        im_12 = axarr[1][2].imshow(np.flipud(velmag_pred), vmin=vel_min, vmax=vel_max, cmap="turbo")
        #im_12 = axarr[1][2].imshow(np.flipud(velmag_pred), cmap="turbo")
        #axarr[1][2].streamplot(X, Y, np.flipud(velx_pred)[10:-10, 10:-10], -np.flipud(vely_pred)[10:-10, 10:-10], density=0.75, color="white")
        #axarr[1][2].imshow(np.flipud(pred_overlay), alpha=1)
        axarr[1][2].axis("off")
        plt.colorbar(im_12, ax=axarr[1][2], fraction=0.04, pad=0.05)
        axarr[1][2].set_title(f"Vel Pred {i}")

        plt.savefig(
            f"{str(plot_dir)}/{i_str}.png",
            bbox_inches="tight",
        )
        plt.close()
        if i % 25 == 0:
            print(f"{i}/{preds.shape[0]} files done")

torch.set_float32_matmul_precision("high")

#test_path = ["/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D/Twall_97.hdf5"]
#test_path = ["/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-R515B-2D/Twall_30.hdf5"]
test_path = ["/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-LN2-2D/Twall_-165.hdf5"]

#test_path = ["/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Saturated-FC72-2D/Twall_91.hdf5"]
#test_path = ["/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Saturated-R515B-2D/Twall_18.hdf5"]
#test_path = ["/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Saturated-LN2-2D/Twall_-176.hdf5"]

test_dataset = BubbleForecast(
    filenames=test_path,
    input_fields=["dfun", "temperature", "velx", "vely"],
    output_fields=["dfun", "temperature", "velx", "vely"],
    norm="none",    
    downsample_factor=8,
    time_window=5,
    start_time=100,
    return_fluid_params=True
)

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
    "num_fluid_params": 13,
}

model = get_model(model_name, **model_kwargs)
model = model.cuda()

#weights_path = "/pub/afeeney/bubbleformer_logs/filmavit_poolboiling_subcooled_47238340/checkpoints/epoch=29-step=56760.ckpt"
#weights_path = "/pub/afeeney/bubbleformer_logs/neighbor_moe_poolboiling_subcooled_47407258/checkpoints/epoch=34-step=132440.ckpt"
#weights_path = "/pub/afeeney/bubbleformer_logs/neighbor_moe_poolboiling_subcooled_47512802/checkpoints/last.ckpt"
#weights_path = "/pub/afeeney/bubbleformer_logs/neighbor_moe_poolboiling_subcooled_47738409/checkpoints/epoch=4-step=59125.ckpt"
weights_path = "/pub/afeeney/bubbleformer_logs/neighbor_moe_poolboiling_subcooled_47763865/checkpoints/last.ckpt"
model_data = torch.load(weights_path, weights_only=False)

diff_term = {
    "dfun": 0.0,
    "temperature": 0.0,
    "velx": 0.0,
    "vely": 0.0,
}
div_term = {
    "dfun": 1.0,
    "temperature": 1.0,
    "velx": 1.0,
    "vely": 1.0,
}

# diff_term = torch.tensor(diff_term)
# div_term = torch.tensor(div_term)
weight_state_dict = OrderedDict()
for key, val in model_data["state_dict"].items():
    name = key[6:]
    weight_state_dict[name] = val
    if torch.isnan(val).any():
        print(f"NAN found in {name}")
del model_data

print(model)

model.load_state_dict(weight_state_dict)
model.eval()

criterion = LpLoss(d=2, p=2, reduce_dims=[0,1], reductions=["mean", "mean"])
start_time = test_dataset.start_time
skip_itrs = test_dataset.time_window
model_preds = []
model_targets = []
timesteps = []

moe_outputs = []

for itr in range(0, 100, skip_itrs):
    data = test_dataset[itr]
    inp = data.input
    tgt = data.target
    fluid_params = data.fluid_params_tensor
    print(f"Autoreg pred {itr}, inp tw [{start_time+itr}, {start_time+itr+skip_itrs}], tgt tw [{start_time+itr+skip_itrs}, {start_time+itr+2*skip_itrs}]")
    if len(model_preds) > 0:
        inp = model_preds[-1] # T, C, H, W
    inp = inp.cuda().to(torch.float32).unsqueeze(0)
    fluid_params = fluid_params.cuda().to(torch.float32).unsqueeze(0)

    pred, moe_output = model(inp, fluid_params)
    moe_outputs.append(moe_output[0]) # NOTE: tracking first layer of MoE outputs

    pred = pred.to(torch.float32)
    pred = pred.squeeze(0).detach().cpu()
    tgt = tgt.detach().cpu()
    
    # reinitialize the top part of the SDF for each timestep
    # heater is at index zero, 
    for i in range(pred.shape[0]):
        pred_sdf = pred[i, 0]
        # The fast marching uses finite differences, so needs a finer grid. Fortunately, the SDF
        # is very smooth, so bicubic interpolation is good enough.
        up_pred_sdf = torch.nn.functional.interpolate(
            pred_sdf.unsqueeze(0).unsqueeze(0), scale_factor=8, mode="bicubic").squeeze()
        up_pred_sdf_corrected = torch.from_numpy(fast_marching_2d(up_pred_sdf.numpy(), dx=(1/4) / 8))
        pred_sdf_corrected = torch.nn.functional.interpolate(
            up_pred_sdf_corrected.unsqueeze(0).unsqueeze(0), scale_factor=1/8, mode="bicubic").squeeze()
        # Only reinitialize the SDF when sufficiently far from the interfaces
        # The constant -4.0 is chosen arbitrarily.
        far_mask = pred_sdf < -4.0
        pred[i, 0, far_mask] = pred_sdf_corrected[far_mask]

    model_preds.append(pred)
    model_targets.append(tgt)
    timesteps.append(torch.arange(start_time+itr+skip_itrs, start_time+itr+2*skip_itrs))

model_preds = torch.cat(model_preds, dim=0)         # T, C, H, W
model_targets = torch.cat(model_targets, dim=0)     # T, C, H, W
timesteps = torch.cat(timesteps, dim=0)             # T,
num_var = len(test_dataset.fields)                  # C

topk_indices = [moe_output.topk_indices.squeeze(0) for moe_output in moe_outputs]
topk_indices = torch.cat(topk_indices, dim=0) # (T, H, W, topk)

save_dir = "./subcooled_fc72_97"
print(f"saving to {save_dir}")

print(torch.stack([torch.isnan(p).any() for p in model.parameters()]).any())

os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "predictions.pt")
torch.save({"preds": model_preds, "targets": model_targets, "timesteps": timesteps}, save_path)
plot_bubbleml(model_preds, model_targets, topk_indices, timesteps, save_dir)