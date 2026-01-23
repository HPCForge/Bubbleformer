from re import M
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, SymLogNorm, LinearSegmentedColormap
import seaborn as sns
from typing import Optional

def ax_default(ax, title: Optional[str] = None):
    if title is not None:
        ax.set_title(title)
    ax.axis("off")

def sdf_cmap():
    ranges = [0.0, 0.49, 0.51, 1]
    color_codes = ["blue", "white", "red"]
    colors = list(zip(ranges, color_codes))
    cmap = LinearSegmentedColormap.from_list('temperature_colormap', colors)
    return cmap

def plot_sdf(ax, sdf: torch.Tensor, title: Optional[str] = None):
    assert sdf.dim() == 2, "SDF must be a 2D tensor (H, W)"
    sdf = sdf.detach().cpu().numpy()
    norm = TwoSlopeNorm(vcenter=0)
    im = ax.imshow(sdf, cmap="RdYlBu", norm=norm)
    ax.contour(sdf, colors="white", linewidths=0.5)
    ax.contour(sdf, levels=[0], colors="black", linewidths=0.5)
    ax_default(ax, title)
    return im

def temp_cmap():
    temp_ranges = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.134, 0.167,
                    0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    color_codes = ['#0000FF', '#0443FF', '#0E7AFF', '#16B4FF', '#1FF1FF', '#21FFD3',
                   '#22FF9B', '#22FF67', '#22FF15', '#29FF06', '#45FF07', '#6DFF08',
                   '#9EFF09', '#D4FF0A', '#FEF30A', '#FEB709', '#FD7D08', '#FC4908',
                   '#FC1407', '#FB0007']
    colors = list(zip(temp_ranges, color_codes))
    cmap = LinearSegmentedColormap.from_list('temperature_colormap', colors)
    return cmap

def plot_temp(ax, temp: torch.Tensor, bulk_temp, heater_temp, title: Optional[str] = None):
    assert temp.dim() == 2, "Temp must be a 2D tensor (H, W)"
    temp = temp.detach().cpu().numpy()
    norm = Normalize(vmin=bulk_temp, vmax=heater_temp, clip=True)
    im = ax.imshow(temp, cmap=temp_cmap(), norm=norm)
    ax_default(ax, title)
    return im

def plot_vel_mag(ax, vel_mag: torch.Tensor, title: Optional[str] = None):
    assert vel_mag.dim() == 2, "Vel mag must be a 2D tensor (H, W)"
    vel_mag = vel_mag.detach().cpu().numpy()
    im = ax.imshow(vel_mag, cmap="plasma", vmin=0)
    ax_default(ax, title)
    return im

def plot_vorticity(ax, vorticity: torch.Tensor, min_vort=None, max_vort=None, title: Optional[str] = None):
    assert vorticity.dim() == 2, "Vorticity must be a 2D tensor (H, W)"
    vorticity = vorticity.detach().cpu().numpy()
    # use a diverging colormap, centered at 0.
    norm = SymLogNorm(linthresh=0.5, vmin=min_vort, vmax=max_vort)
    im = ax.imshow(vorticity, cmap="icefire", norm=norm)
    ax_default(ax, title)
    return im

if __name__ == "__main__":
    import h5py
    import numpy as np
    from bubbleformer.utils.physical_metrics import vorticity
    with h5py.File("/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D/Twall_97.hdf5", "r") as f:
        sdf = torch.from_numpy(f["dfun"][:])
        temp = torch.from_numpy(f["temperature"][:])
        velx = torch.from_numpy(f["velx"][:])
        vely = torch.from_numpy(f["vely"][:])

    vel_mag = torch.sqrt(velx**2 + vely**2)
    v = vorticity(velx[None, ...], vely[None, ...], 1/32, 1/32)
    v = v.squeeze()

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
    t = plot_sdf(axs[0, 0], torch.flipud(sdf[400]))
    plt.colorbar(t, ax=axs[0, 0])
    t = plot_temp(axs[0, 1], torch.flipud(temp[400]), 50, 97)
    plt.colorbar(t, ax=axs[0, 1])
    plot_vel_mag(axs[1, 0], torch.flipud(vel_mag[400]))
    t = plot_vorticity(axs[1, 1], torch.flipud(v[400]))
    plt.colorbar(t, ax=axs[1, 1])
    plt.savefig("test.png")