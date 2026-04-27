"""
Plotting utilities for ML predictions

Author: Sheikh Md Shakeel Hassan
"""
import os
import cv2 # pylint: disable=import-error
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_bubbleml(
        preds: torch.Tensor,
        targets: torch.Tensor,
        timesteps: torch.Tensor,
        save_dir: str,
    ):
    """
    Plot the BubbleML predictions, targets and errors for each timestep
    Also plots the relative L2 error over time for each channel
    Args:
        preds: Predictions from the model for a single rollout (T, 4, H, W)
        targets: Ground truth targets for a single rollout (T, 4, H, W)
        timesteps: Timesteps for the predictions for a single rollout (T,)
        save_dir: Directory to save the plots
    """
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Compute the L2 norm across spatial dimensions (h, w)
    diff_norm = torch.norm(preds - targets, p=2, dim=(2, 3))  # Shape (t, c)
    bnorm = torch.norm(targets, p=2, dim=(2, 3))        # Shape (t, c)

    relative_l2_error = diff_norm / bnorm

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
    temp_min, temp_max = round(temp_mean - 3 * temp_std, 2), round(temp_mean + 3 * temp_std, 2)

    sdf_mean, sdf_std = torch.mean(targets[:, 0]).item(), torch.std(targets[:, 0]).item()
    sdf_min, sdf_max = round(sdf_mean - 3 * sdf_std, 2), round(sdf_mean + 3 * sdf_std, 2)

    for i in range(preds.shape[0]):
        i_str = str(i).zfill(4)

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
        x = np.arange(2, velmag_target.shape[1]-2, 1)
        y = np.arange(2, velmag_target.shape[0]-2, 1)
        X,Y = np.meshgrid(x,y)
        velx_pred[dfun_target==0] = 0
        vely_pred[dfun_target==0] = 0
        velx_target[dfun_target==0] = 0
        vely_target[dfun_target==0] = 0


        f, axarr = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
        im_00 = axarr[0][0].imshow(sdf_target, vmin=sdf_min, vmax=sdf_max, cmap="Blues", origin="lower")
        axarr[0][0].imshow(target_overlay, alpha=1, origin="lower")
        axarr[0][0].axis("off")
        plt.colorbar(im_00, ax=axarr[0][0], fraction=0.04, pad=0.05)
        axarr[0][0].set_title(f"SDF Label {i}")

        im_01 = axarr[0][1].imshow(temp_target, cmap="turbo", vmin=temp_min, vmax=temp_max, origin="lower")
        #im_01 = axarr[0][1].imshow(temp_target, cmap="turbo", origin="lower")
        #axarr[0][1].imshow(target_overlay, alpha=1, origin="lower")
        axarr[0][1].axis("off")
        plt.colorbar(im_01, ax=axarr[0, 1], fraction=0.04, pad=0.05)
        axarr[0][1].set_title(f"Temp Label {i}")

        im_02 = axarr[0][2].imshow(np.flipud(velmag_target), vmin=vel_min, vmax=vel_max, cmap="turbo")
        #im_02 = axarr[0][2].imshow(np.flipud(velmag_target), cmap="turbo")
        axarr[0][2].streamplot(X, Y, np.flipud(velx_target)[2:-2, 2:-2], -np.flipud(vely_target)[2:-2, 2:-2], density=0.75, color="white")
        #axarr[0][2].imshow(np.flipud(target_overlay), alpha=1)
        axarr[0][2].axis("off")
        plt.colorbar(im_02, ax=axarr[0][2], fraction=0.04, pad=0.05)
        axarr[0][2].set_title(f"Vel Label {i}")

        im_10 = axarr[1][0].imshow(sdf_pred, vmin=sdf_min, vmax=sdf_max, cmap="Blues", origin="lower")
        axarr[1][0].imshow(pred_overlay, alpha=1, origin="lower")
        axarr[1][0].axis("off")
        plt.colorbar(im_10, ax=axarr[1][0], fraction=0.04, pad=0.05)
        axarr[1][0].set_title(f"SDF Pred {i}")

        im_11 = axarr[1][1].imshow(temp_pred, cmap="turbo", vmin=temp_min, vmax=temp_max, origin="lower")
        #im_11 = axarr[1][1].imshow(temp_pred, cmap="turbo", origin="lower")
        #axarr[1][1].imshow(pred_overlay, alpha=1, origin="lower")
        axarr[1][1].axis("off")
        plt.colorbar(im_11, ax=axarr[1, 1], fraction=0.04, pad=0.05)
        axarr[1][1].set_title(f"Temp Pred {i}")

        im_12 = axarr[1][2].imshow(np.flipud(velmag_pred), vmin=vel_min, vmax=vel_max, cmap="turbo")
        #im_12 = axarr[1][2].imshow(np.flipud(velmag_pred), cmap="turbo")
        axarr[1][2].streamplot(X, Y, np.flipud(velx_pred)[2:-2, 2:-2], -np.flipud(vely_pred)[2:-2, 2:-2], density=0.75, color="white")
        #axarr[1][2].imshow(np.flipud(pred_overlay), alpha=1)
        axarr[1][2].axis("off")
        plt.colorbar(im_12, ax=axarr[1][2], fraction=0.04, pad=0.05)
        axarr[1][2].set_title(f"Vel Pred {i}")

        #im_20 = axarr[2][0].imshow(dfun_err, vmin=0, vmax=1, cmap="Blues", origin="lower")
        #axarr[2][0].axis("off")
        #axarr[2][0].set_title(f"Rel Dfun L1 error {i}")
        #f.colorbar(im_20, ax=axarr[2, 0], fraction=0.04, pad=0.05)

        #im_21 = axarr[2][1].imshow(temp_err, vmin=0, vmax=1, cmap="turbo", origin="lower")
        #axarr[2][1].axis("off")
        #axarr[2][1].set_title(f"Rel Temp L1 error {i}")
        #f.colorbar(im_21, ax=axarr[2, 1], fraction=0.04, pad=0.05)

        #im_22 = axarr[2][2].imshow(velmag_err, vmin=0, vmax=1, cmap="turbo", origin="lower")
        #axarr[2][2].axis("off")
        #axarr[2][2].set_title(f"Rel Vel L1 error {i}")
        #f.colorbar(im_22, ax=axarr[2][2], fraction=0.04, pad=0.05)

        plt.savefig(
            f"{str(plot_dir)}/{i_str}.png",
            bbox_inches="tight",
        )
        plt.close()
        if i % 25 == 0:
            print(f"{i} files done")

def wandb_sdf_plotter(sdf: torch.Tensor) -> plt.Figure:
    """
    Return the plot of a single T x H x W SDF tensor
    """
    fig, axes = plt.subplots(1, sdf.shape[0], figsize=(3 * sdf.shape[0], 6))
    for i, ax in enumerate(axes):
        dfun = sdf[i].cpu().numpy()
        dfun[dfun>0] = 0
        dfun[dfun<0] = 255
        dfun = dfun.astype(np.uint8)
        edge_map = cv2.Canny(dfun, 0, 255)
        kernel = np.ones((3,3),np.uint8)
        edge_map = cv2.dilate(edge_map, kernel, iterations=1)
        mask = np.where(edge_map > 0, 0, 255)
        alpha = np.where(mask > 0, 0, 255)
        overlay = np.dstack((mask, mask, mask, alpha))

        img = ax.imshow(sdf[i].cpu().numpy(), cmap="Blues", origin="lower")
        ax.imshow(overlay, alpha=1, origin="lower")
        ax.axis("off")
        ax.set_title(f"SDF {i}")
    fig.colorbar(img, fraction=0.04, pad=0.05)
    return fig

def wandb_rho_plotter(rho: torch.Tensor) -> plt.Figure:
    """
    Return the plot of a single T x H x W density tensor
    """
    fig, axes = plt.subplots(1, rho.shape[0], figsize=(3 * rho.shape[0], 6))
    for i, ax in enumerate(axes):
        img = ax.imshow(rho[i].cpu().numpy(), cmap="Spectral", origin="lower")
        ax.axis("off")
        ax.set_title(f"Rho {i}")
    fig.colorbar(img, fraction=0.04, pad=0.05)
    return fig

def wandb_temp_plotter(temp: torch.Tensor) -> plt.Figure:
    """
    Return the plot of a single T x H x W temperature tensor
    """
    fig, axes = plt.subplots(1, temp.shape[0], figsize=(3 * temp.shape[0], 6))
    for i, ax in enumerate(axes):
        img = ax.imshow(temp[i].cpu().numpy(), cmap="turbo", origin="lower")
        ax.axis("off")
        ax.set_title(f"Temp {i}")
    fig.colorbar(img, fraction=0.04, pad=0.05)
    return fig

def wandb_vel_plotter(vel: torch.Tensor) -> plt.Figure:
    """
    Return the plot of a single T x 2 x H x W velocity tensor
    """
    fig, axes = plt.subplots(1, vel.shape[0], figsize=(3 * vel.shape[0], 6))
    for i, ax in enumerate(axes):
        velx = vel[i, 0].cpu().numpy()
        vely = vel[i, 1].cpu().numpy()
        velmag = np.sqrt(velx**2 + vely**2)
        x = np.arange(0, velmag.shape[1], 1)
        y = np.arange(0, velmag.shape[0], 1)
        X,Y = np.meshgrid(x,y)
        img = ax.imshow(np.flipud(velmag), cmap="turbo")
        ax.streamplot(X, Y, np.flipud(velx), -np.flipud(vely), density=0.75, color="white")
        ax.axis("off")
        ax.set_title(f"Vel {i}")
    fig.colorbar(img, fraction=0.04, pad=0.05)
    return fig
