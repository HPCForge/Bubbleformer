import os
import torch
from collections import OrderedDict
from bubbleformer.models import get_model
from bubbleformer.data import TempPredict
from bubbleformer.utils.losses import LpLoss
import matplotlib.pyplot as plt
import math
import numpy as np

def plot_temp(
    preds: torch.Tensor,
    targets: torch.Tensor,
    timesteps: torch.Tensor,
    save_dir: str,
):
    """
    Plot the temperature predictions, targets and errors for each timestep
    Also plots the relative L2 error over time
    Args:
        preds: Predictions from the model for a single rollout (T, 1, H, W)
        targets: Ground truth targets for a single rollout (T, 1, H, W)
        timesteps: Timesteps for the predictions for a single rollout (T,)
        save_dir: Directory to save the plots
    """
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Compute the L2 norm across spatial dimensions (h, w)
    diff_norm = torch.norm(preds - targets, p=2, dim=(2, 3))  # Shape (t, 1)
    bnorm = torch.norm(targets, p=2, dim=(2, 3))        # Shape (t, 1)

    relative_l2_error = diff_norm / bnorm
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps.numpy(), relative_l2_error[:, 0].numpy(), label="Temperature")
    plt.xlabel("Time (timesteps)")
    plt.ylabel("Relative L2 Error")
    plt.title("Relative L2 Error Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "relative_l2_error.png"))

    temp_min, temp_max = math.floor(torch.min(targets[:, 0]).item()), math.ceil(torch.max(targets[:, 0]).item())

    for i in range(preds.shape[0]):
        i_str = str(i).zfill(4)

        temp_pred = preds[i, 0, :, :].numpy()
        temp_target = targets[i, 0, :, :].numpy()
        temp_err = np.abs(temp_target - temp_pred)/(np.abs(temp_target) + 1.0e-8)

        f, axarr = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
        im_0 = axarr[0].imshow(temp_target, cmap="turbo", vmin=temp_min, vmax=temp_max, origin="lower")
        axarr[0].axis("off")
        plt.colorbar(im_0, ax=axarr[0], fraction=0.04, pad=0.05)
        axarr[0].set_title(f"Temp Label {i}")

        im_1 = axarr[1].imshow(temp_pred, cmap="turbo", vmin=temp_min, vmax=temp_max, origin="lower")
        axarr[1].axis("off")
        plt.colorbar(im_1, ax=axarr[1], fraction=0.04, pad=0.05)
        axarr[1].set_title(f"Temp Pred {i}")

        im_2 = axarr[2].imshow(temp_err, cmap="turbo", vmin=0.0, vmax=1.0, origin="lower")
        axarr[2].axis("off")
        plt.colorbar(im_2, ax=axarr[2], fraction=0.04, pad=0.05)
        axarr[2].set_title(f"Abs Rel Error {i}")   

        plt.savefig(
            f"{str(plot_dir)}/{i_str}.png",
            bbox_inches="tight",
        )
        plt.close()
        if i % 25 == 0:
            print(f"{i} files done")

if __name__ == "__main__":
    weights_path = "/pub/sheikhh1/bubbleformer_logs/avit_bubbleml_subcooled_47041296/hpc_ckpt_1.ckpt"
    model_data = torch.load(weights_path, map_location="cuda", weights_only=False)
    model_name = model_data["hyper_parameters"]["model_cfg"]["name"]
    model_params = dict(model_data["hyper_parameters"]["model_cfg"]["params"])

    model_params["input_fields"] = len(model_data["hyper_parameters"]["data_cfg"]["input_fields"])
    model_params["output_fields"] = len(model_data["hyper_parameters"]["data_cfg"]["output_fields"])
    model_params["time_window"] = model_data["hyper_parameters"]["data_cfg"]["time_window"]
    model = get_model(model_name, **model_params)

    data_path = ["/share/crsp/lab/amowli/share/BubbleML2/PoolBoiling-SubCooled-FC72-2D-0.1/Twall-103.hdf5"]
    test_dataset = TempPredict(
        filenames=data_path,
        time_window=model_data["hyper_parameters"]["data_cfg"]["time_window"],
        start_time=model_data["hyper_parameters"]["data_cfg"]["start_time"],
    )
    diff_term, div_term = model_data['hyper_parameters']['normalization_constants']
    _, _ = test_dataset.normalize(diff_term, div_term)

    weight_state_dict = OrderedDict()
    for key, val in model_data["state_dict"].items():
        name = key[6:]
        weight_state_dict[name] = val
    del model_data
    model.load_state_dict(weight_state_dict)
    model = model.cuda()

    criterion = LpLoss(d=2, p=2, reduce_dims=[0,1], reductions=["mean", "mean"])
    model.eval()
    start_time = test_dataset.start_time
    skip_itrs = test_dataset.time_window
    model_preds = []
    model_targets = []
    timesteps = []

    for itr in range(0, 200, skip_itrs):
        inp, tgt = test_dataset[itr]
        print(f"Autoreg pred {itr}, inp tw [{start_time+itr}, {start_time+itr+skip_itrs}], tgt tw [{start_time+itr+skip_itrs}, {start_time+itr+2*skip_itrs}]")
        if len(model_preds) > 0:
            inp[:,0,:,:] = model_preds[-1].squeeze(1) # T, C, H, W
        inp = inp.cuda().float().unsqueeze(0)
        pred = model(inp)
        pred = pred.squeeze(0).detach().cpu()
        tgt = tgt.detach().cpu()

        model_preds.append(pred)
        model_targets.append(tgt)
        timesteps.append(torch.arange(start_time+itr+skip_itrs, start_time+itr+2*skip_itrs))
        print(criterion(pred, tgt))

    model_preds = torch.cat(model_preds, dim=0)         # T, C, H, W
    model_targets = torch.cat(model_targets, dim=0)     # T, C, H, W
    timesteps = torch.cat(timesteps, dim=0)             # T,

    save_dir = "/pub/sheikhh1/bubbleformer_logs/avit_bubbleml_subcooled_47041296/epoch_150"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "predictions.pt")
    torch.save({"preds": model_preds, "targets": model_targets, "timesteps": timesteps}, save_path)
    plot_temp(model_preds, model_targets, timesteps, save_dir)