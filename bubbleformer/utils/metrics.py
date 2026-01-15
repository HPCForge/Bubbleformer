import torch
from collections import deque
from typing import List

def eikonal_equation(sdf):
    r"""
    This computes ||grad(phi)|| for each timestep. It returns a tensor of shape (B, T).
    It is expected that the eikonal equation of an SDF is 1.
    """
    assert sdf.dim() == 4, "SDF must be of shape (B, T, H, W)"
    H, W = sdf.shape[-2], sdf.shape[-1]
    dx = 1/32 # NOTE: This is specific to pool boiling data
    grad_phi_y, grad_phi_x = torch.gradient(sdf, spacing=dx, dim=(-2, -1), edge_order=1)
    grad_mag = torch.sqrt(grad_phi_y**2 + grad_phi_x**2).sum(dim=(-2, -1)) * dx ** 2
    return grad_mag

def vapor_volume(sdf):
    r"""
    This computes the vapor volume (or void fraction in domain-speak.) This is basically
    how much of the domain is in the vapor phase. It returns a tensor of shape (B, T).
    """
    assert sdf.dim() == 4, "SDF must be of shape (B, T, H, W)"
    dx = 1/32
    vapor_mask = sdf > 0
    vapor_volume = torch.sum(vapor_mask, dim=(-2, -1)) * dx ** 2
    return vapor_volume

def estimate_bubble_count(sdf):
    r"""
    This approximates the number of bubbles in the domain. It uses a simple "island counting"
    algorithm to count the number of disconnectred regions in vapor. It returns a tensor of shape (B, T).
    NOTE: This may miscount the number of bubbles, if two bubbles are adjacent to each other, 
    so it is just intended to provide a rough estimate. (Especially when applied to lower resolution data.)
    """
    assert sdf.dim() == 4, "SDF must be of shape (B, T, H, W)"
    B, T, H, W = sdf.shape
    
    # This uses a mask to do the "island counting." This is actually going to be less accurate
    # than directly using the SDF, since the mask loses some information--with the distance function,
    # we can determine that an interface goes between two adjacent vapor pixels, but that
    # is not possible when using a mask. (It will just count the two adjacent pixels as one bubble.)
    # This only has the advantage of being simpler.
    vapor_mask = sdf > 0
    
    # We comute the number of bubbles in each batch and timestep:
    num_bubbles: List[List[int]] = []
    for b in range(B):
        num_bubbles_in_batch: List[int] = []
        for t in range(T):
            
            # Use an island counting algorithm for the current batch and timestep:
            num_bubbles_in_timestep: int = 0
            seen_pixels = torch.zeros_like(vapor_mask[b, t], dtype=torch.bool)
            for i in range(H):
                for j in range(W):
                    if vapor_mask[b, t, i, j] and not seen_pixels[i, j]:
                        num_bubbles_in_timestep += 1
                        seen_pixels[i, j] = True
                        bfs_flood_fill(i, j, seen_pixels, vapor_mask[b, t])
            num_bubbles_in_batch.append(num_bubbles_in_timestep)
        num_bubbles.append(num_bubbles_in_batch)
        
    # We return a tensor of shape (B, T), with the number of bubbles in each batch and timestep.
    num_bubbles = torch.tensor(num_bubbles, device=sdf.device, dtype=torch.int32)
    assert num_bubbles.shape[0] == B and num_bubbles.shape[1] == T
    return num_bubbles

def vapor_neighbors(i, j, seen_pixels, vapor_mask):
    r"""
    This returns the neighbors of the pixel (i, j) that are in the vapor phase.
    """
    H, W = vapor_mask.shape[-2], vapor_mask.shape[-1]
    neighbors = []
    if i > 0 and vapor_mask[i-1, j] and not seen_pixels[i-1, j]:
        neighbors.append((i-1, j))
    if i < H-1 and vapor_mask[i+1, j] and not seen_pixels[i+1, j]:
        neighbors.append((i+1, j))
    if j > 0 and vapor_mask[i, j-1] and not seen_pixels[i, j-1]:
        neighbors.append((i, j-1))
    if j < W-1 and vapor_mask[i, j+1] and not seen_pixels[i, j+1]:
        neighbors.append((i, j+1))
    return neighbors

def bfs_flood_fill(i, j, seen_pixels, vapor_mask):
    r"""
    This is a simple breadth-first search to flood fill the bubble.
    """
    to_flood_queue = deque([(i, j)])
    while to_flood_queue:
        x, y = to_flood_queue.popleft()
        if seen_pixels[x, y]:
            continue
        seen_pixels[x, y] = True
        to_flood_queue.extend(vapor_neighbors(x, y, seen_pixels, vapor_mask))