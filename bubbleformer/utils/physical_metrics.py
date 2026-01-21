import numpy as np
import scipy
from scipy.ndimage import label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import torch
from collections import deque
import dataclasses

def vorticity(velx, vely, dx, dy):
    r"""
    This computes the vorticity (B, T, H, W) from the velocity fields.
    Args:
        velx: Velocity field in the x direction (B, T, H, W)
        vely: Velocity field in the y direction (B, T, H, W)
        dx: Spatial resolution in the x direction
        dy: Spatial resolution in the y direction
    """
    assert velx.dim() == 4 and vely.dim() == 4, "Velocity fields must be of shape (B, T, H, W)"
    assert velx.shape == vely.shape, "Velocity fields must have the same shape"
    assert dx > 0 and dy > 0, "Spatial resolution must be positive"
    dydx = torch.gradient(vely, spacing=dx, dim=-1)
    dxdy = torch.gradient(velx, spacing=dy, dim=-2)
    return dydx - dxdy

def interface_mask(sdf):
    assert sdf.dim() == 4, "SDF must be of shape (B, T, H, W)"
    interface = torch.zeros_like(sdf, dtype=torch.bool)
    [_, _, rows, cols] = sdf.shape
    for i in range(rows):
        for j in range(cols):
            # adj is a (B, T) shaped mask.
            adj = ((i < rows - 1 and sdf[:, :, i, j] * sdf[:, :, i+1, j] <= 0) or
                   (i > 0 and sdf[:, :, i, j] * sdf[:, :, i-1, j ] <= 0) or
                   (j < cols - 1 and sdf[:, :, i, j] * sdf[:, :, i, j+1] <= 0) or
                   (j > 0 and sdf[:, :, i, j] * sdf[:, :, i, j-1] <= 0))
            interface[:, :, i, j] = adj
    return interface

def interface_velocity(velx, vely, sdf):
    mask = interface_mask(sdf)
    interface_velx = velx * mask
    interface_vely = vely * mask
    return interface_velx, interface_vely

def eikonal(sdf, dx):
    r"""
    This computes ||grad(phi)|| for each timestep. It returns a tensor of shape (B, T).
    It is expected that the eikonal equation of an SDF is 1.
    Note: even the ground truth flash-x simulations do not satisfy the eikonal equation very well. 
    The simulation's SDF may experience a spike when bubbles nucleate and and occasionally gets reset.
    """
    assert sdf.dim() == 4, "SDF must be of shape (B, T, H, W)"
    grad_phi_y, grad_phi_x = torch.gradient(sdf, spacing=dx, dim=(-2, -1), edge_order=1)
    grad_mag = torch.sqrt(grad_phi_y**2 + grad_phi_x**2).sum(dim=(-2, -1)) * dx ** 2
    return grad_mag

def vapor_volume(sdf, dx):
    r"""
    This computes the vapor volume (or void fraction in domain-speak.) This is basically
    how much of the domain is in the vapor phase. It returns a tensor of shape (B, T).
    """
    assert sdf.dim() == 4, "SDF must be of shape (B, T, H, W)"
    vapor_mask = sdf > 0
    vapor_volume = torch.sum(vapor_mask, dim=(-2, -1)) * dx ** 2
    return vapor_volume

def find_bubbles_at_timestep(sdf):
    r"""
    Given an SDF, this uses a watershed algorithm to find each of the individual bubbles.
    It returns the number of bubbles, and an array of bubble labels. zero is the background.
    """
    sdf_npy = sdf.detach().cpu().numpy()
    coords = peak_local_max(sdf_npy, footprint=np.ones((3, 3)))
    mask = np.zeros_like(sdf_npy, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = label(mask)
    bubble_labels = watershed(-sdf_npy, markers, mask=sdf_npy > 0)
    return torch.from_numpy(bubble_labels).to(sdf.device)

def find_bubbles(sdf):
    assert sdf.dim() == 4, "SDF must be of shape (B, T, H, W)"
    bubble_labels = torch.zeros_like(sdf, dtype=torch.int32)
    for b in range(sdf.shape[0]):
        for t in range(sdf.shape[1]):
            bubble_labels[b, t] = find_bubbles_at_timestep(sdf[b, t])
    return bubble_labels

def bubble_count(sdf):
    bubble_labels = find_bubbles(sdf)
    return bubble_labels.max()

def bubble_volume(sdf, dx):
    bubble_labels = find_bubbles(sdf)
    bubble_mask = bubble_labels > 0
    bubble_volumes = np.sum(bubble_mask > 0, axis=(1, 2)) * dx ** 2
    return bubble_volumes