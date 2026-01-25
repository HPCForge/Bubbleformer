import numpy as np
from scipy.ndimage import label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import torch
from collections import deque
import dataclasses
from typing import List

@dataclasses.dataclass
class BubbleMetrics:
    bubble_labels: torch.Tensor # (B, T, H, W)
    bubble_count: torch.Tensor # (B, T)
    bubble_volume: List[List[List[float]]] # (B, T, num_bubbles)
    bubble_x_velocity: List[List[List[float]]] # (B, T, num_bubbles)
    bubble_y_velocity: List[List[List[float]]] # (B, T, num_bubbles)

def bubble_metrics(sdf, velx, vely, dx, dy):
    return BubbleMetrics(
        bubble_labels=find_bubbles(sdf),
        bubble_count=bubble_count(sdf),
        bubble_volume=bubble_volume(sdf, dx, dy),
        bubble_x_velocity=bubble_velocity(sdf, velx, dx, dy),
        bubble_y_velocity=bubble_velocity(sdf, vely, dx, dy)
    )

@dataclasses.dataclass
class PhysicalMetrics:
    eikonal: torch.Tensor # (B, T)
    
    heatflux: torch.Tensor # (B, T)
    heatflux_at_heater: torch.Tensor # (B, T)

    mean_liquid_temperature: torch.Tensor # (B, T)
    liquid_temperature_at_heater: torch.Tensor # (B, T, W)

    vapor_volume: torch.Tensor # (B, T)
    vapor_volume_at_height: torch.Tensor # (B, T, H)

    mean_liquid_x_velocity: torch.Tensor # (B, T)
    mean_liquid_y_velocity: torch.Tensor # (B, T)
    mean_vapor_x_velocity: torch.Tensor # (B, T)
    mean_vapor_y_velocity: torch.Tensor # (B, T)
    mean_interface_x_velocity: torch.Tensor # (B, T)
    mean_interface_y_velocity: torch.Tensor # (B, T)
    
def physical_metrics(
    sdf, 
    temperature, 
    velx,
    vely,
    heater_min,
    heater_max,
    xcoords,
    dx,
    dy=None,
):
    if dy is None:
        dy = dx
    liquid_x_velocity, liquid_y_velocity = liquid_velocity(velx, vely, sdf)
    vapor_x_velocity, vapor_y_velocity = vapor_velocity(velx, vely, sdf)
    interface_x_velocity, interface_y_velocity = interface_velocity(velx, vely, sdf)
    
    return PhysicalMetrics(
        eikonal=eikonal(sdf, dx, dy),
        heatflux=None,
        heatflux_at_heater=None,
        mean_liquid_temperature=liquid_temperature(temperature, sdf),
        liquid_temperature_at_heater=liquid_temperature_at_heater(temperature, sdf, heater_min, heater_max, xcoords),
        vapor_volume=vapor_volume(sdf, dx, dy),
        vapor_volume_at_height=vapor_volume_at_height(sdf, dx, dy),
        mean_liquid_x_velocity=liquid_x_velocity,
        mean_liquid_y_velocity=liquid_y_velocity,
        mean_vapor_x_velocity=vapor_x_velocity,
        mean_vapor_y_velocity=vapor_y_velocity,
        mean_interface_x_velocity=interface_x_velocity,
        mean_interface_y_velocity=interface_y_velocity,
    )

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
    dydx = torch.gradient(vely, spacing=dx, dim=-1)[0]
    dxdy = torch.gradient(velx, spacing=dy, dim=-2)[0]
    return dydx - dxdy

def interface_mask(sdf):
    assert sdf.dim() == 4, "SDF must be of shape (B, T, H, W)"
    interface = torch.zeros_like(sdf, dtype=torch.bool, device="cpu")
    [B, T, rows, cols] = sdf.shape
    for b in range(B):
        for t in range(T):
            for i in range(rows):
                for j in range(cols):
                    # adj is a (B, T) shaped mask.
                    adj = ((i < rows - 1 and (sdf[b, t, i, j] * sdf[b, t, i+1, j] <= 0)) or
                        (i > 0 and (sdf[b, t, i, j] * sdf[b, t, i-1, j ] <= 0)) or
                        (j < cols - 1 and (sdf[b, t, i, j] * sdf[b, t, i, j+1] <= 0)) or
                        (j > 0 and (sdf[b, t, i, j] * sdf[b, t, i, j-1] <= 0)))
                    interface[b, t, i, j] = adj
    return interface.to(sdf.device)

def interface_velocity(velx, vely, sdf):
    mask = interface_mask(sdf)
    interface_velx = (velx * mask).sum(dim=(-2, -1)) / mask.to(torch.int32).sum(dim=(-2, -1))
    interface_vely = (vely * mask).sum(dim=(-2, -1)) / mask.to(torch.int32).sum(dim=(-2, -1))
    return interface_velx, interface_vely

def liquid_velocity(velx, vely, sdf):
    mask = sdf < 0
    liquid_velx = (velx * mask).sum(dim=(-2, -1)) / mask.to(torch.int32).sum(dim=(-2, -1))
    liquid_vely = (vely * mask).sum(dim=(-2, -1)) / mask.to(torch.int32).sum(dim=(-2, -1))
    return liquid_velx, liquid_vely

def vapor_velocity(velx, vely, sdf):
    mask = sdf >= 0
    vapor_velx = (velx * mask).sum(dim=(-2, -1)) / mask.to(torch.int32).sum(dim=(-2, -1))
    vapor_vely = (vely * mask).sum(dim=(-2, -1)) / mask.to(torch.int32).sum(dim=(-2, -1))
    return vapor_velx, vapor_vely

def eikonal(sdf, dx, dy):
    r"""
    This computes ||grad(phi)|| for each timestep. It returns a tensor of shape (B, T).
    It is expected that the eikonal equation of an SDF is 1.
    Note: even the ground truth flash-x simulations do not satisfy the eikonal equation very well. 
    The simulation's SDF may experience a spike when bubbles nucleate and and occasionally gets reset.
    """
    grad_phi_y, grad_phi_x = torch.gradient(sdf, spacing=(dy, dx), dim=(-2, -1), edge_order=1)
    grad_mag = torch.sqrt(grad_phi_y**2 + grad_phi_x**2).mean(dim=(-2, -1))
    return grad_mag

def vapor_volume(sdf, dx, dy):
    r"""
    This computes the vapor volume (or void fraction in domain-speak.) This is basically
    how much of the domain is in the vapor phase. It returns a tensor of shape (B, T).
    """
    assert sdf.dim() == 4, "SDF must be of shape (B, T, H, W)"
    vapor_mask = sdf >= 0
    vapor_volume = torch.sum(vapor_mask, dim=(-2, -1)) * dx * dy
    return vapor_volume

def vapor_volume_at_height(sdf, dx, dy):
    r"""
    This checks the amount of vapor at each row of the domain.
    This returns a tensor of shape (B, T, H).
    """
    assert sdf.dim() == 4, "SDF must be of shape (B, T, H, W)"
    vapor_mask = sdf >= 0
    vapor_volume = torch.sum(vapor_mask, dim=(-1)) * dx
    return vapor_volume

def liquid_temperature(temperature, sdf):
    assert temperature.dim() == 4 and sdf.dim() == 4, "Temperature and SDF must be of shape (B, T, H, W)"
    assert temperature.shape == sdf.shape, "Temperature and SDF must have the same shape"
    liquid_mask = sdf < 0
    return (temperature * liquid_mask.to(temperature.dtype)).sum(dim=(-2, -1)) / liquid_mask.to(torch.int32).sum(dim=(-2, -1))

def liquid_temperature_at_heater(temperature, sdf, heater_min, heater_max, xcoords):
    assert temperature.dim() == 4 and sdf.dim() == 4, "Temperature and SDF must be of shape (B, T, H, W)"
    assert temperature.shape == sdf.shape, "Temperature and SDF must have the same shape"
    assert heater_min < heater_max, "Heater min must be less than heater max"
    bottom_row_liquid_mask = sdf[..., 0, :] < 0
    heater_mask = (xcoords >= heater_min) & (xcoords <= heater_max)
    liquid_at_heater_mask = (heater_mask & bottom_row_liquid_mask)
    return temperature[..., 0, :][liquid_at_heater_mask].mean(dim=-1)

def find_bubbles_at_timestep(sdf):
    r"""
    Given an SDF, this uses a watershed algorithm to find each of the individual bubbles.
    It returns an array of bubble labels [1 - num_bubbles]. Zero corresponds to the liquid phase.
    """
    sdf_npy = sdf.detach().cpu().numpy()
    coords = peak_local_max(sdf_npy)
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
    assert bubble_labels.dim() == 4, "Bubble labels must be of shape (B, T, H, W)"
    
    bubble_counts = []
    for b in range(sdf.shape[0]):
        bubble_counts_at_batch = []
        for t in range(sdf.shape[1]):
            bubble_count_at_timestep = 0
            for bubble_label in range(1, bubble_labels[b, t].max() + 1):
                bubble_mask = (bubble_labels[b, t] == bubble_label)
                if bubble_mask.sum() != 0:
                    bubble_count_at_timestep += 1
            bubble_counts_at_batch.append(bubble_count_at_timestep)
        bubble_counts.append(bubble_counts_at_batch)
    
    return torch.tensor(bubble_counts, dtype=torch.int32, device=sdf.device)

def bubble_volume(sdf, dx, dy):
    bubble_labels = find_bubbles(sdf)
    bubble_volumes = []
    for b in range(sdf.shape[0]):
        bubble_volumes_at_batch = []
        for t in range(sdf.shape[1]):
            bubbles_at_timestep = []
            for bubble_label in range(1, bubble_labels[b, t].max() + 1):
                bubble_mask = (bubble_labels[b, t] == bubble_label)
                # If the mask is empty for the current bubble, it probably
                # means that the peak-finding algorithm combined two local peaks
                # that were inside the same bubble. So some of the labels will be unused.
                if bubble_mask.sum() == 0:
                    continue 
                bubble_volume = torch.sum(bubble_mask.to(torch.float32), dim=(-2, -1)) * dx * dy
                bubbles_at_timestep.append(bubble_volume.item())
            bubble_volumes_at_batch.append(bubbles_at_timestep)
        bubble_volumes.append(bubble_volumes_at_batch)
        
    # bubble_volumes is a triply-nested list of float because
    # the number of bubbles per timestep and batch is variable.
    return bubble_volumes

def bubble_velocity(sdf, vel, dx, dy):
    bubble_labels = find_bubbles(sdf)
    bubble_velocities = []
    for b in range(sdf.shape[0]):
        bubble_velocities_at_batch = []
        for t in range(sdf.shape[1]):
            bubbles_velocities_at_timestep = []
            for bubble_label in range(1, bubble_labels[b, t].max() + 1):
                bubble_mask = (bubble_labels[b, t] == bubble_label)
                # If the mask is empty for the current bubble, it probably
                # means that the peak-finding algorithm combined two local peaks
                # that were inside the same bubble. So some of the labels will be unused.
                if bubble_mask.sum() == 0:
                    continue
                bubble_mean_velocity = (vel[b, t] * bubble_mask).to(torch.float32).sum(dim=(-2, -1)) / bubble_mask.to(torch.int32).sum(dim=(-2, -1))
                bubbles_velocities_at_timestep.append(bubble_mean_velocity.item())
            bubble_velocities_at_batch.append(bubbles_velocities_at_timestep)
        bubble_velocities.append(bubble_velocities_at_batch)
    
    # bubble_velocities is a triply-nested list of float because
    # the number of bubbles per timestep and batch is variable.
    return bubble_velocities