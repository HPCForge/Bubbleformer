r"""
This implements the fast marching method for reinitializing the SDF.
https://en.wikipedia.org/wiki/Fast_marching_method
This essentially assumes that the SDf is decently accurate near the interface
and less accurate away from the interface (which seems to be true in practice!)
"""
import numpy as np
import heapq
import torch

# Status codes for the fast marching method
KNOWN = 2
TRIAL = 1
FAR = 0

def sdf_reinit(
    sdf_init: torch.Tensor, 
    dx: float,
    scale_factor: int = 8,
    far_threshold: float = -4.0
) -> torch.Tensor:
    r"""
    This applies a reinitialization of the SDF using the fast marching method. It only
    updates points that are sufficiently far from the liquid-vapor interface.
    Args:
        sdf_init: torch.Tensor, shape (T,H, W), the SDF to reinitialize.
        dx: float, the grid spacing.
        far_threshold: float, the threshold for far points.
    Returns:
        torch.Tensor, shape (T,H, W), the reinitialized SDF.
    """
    if sdf_init.dim() == 2:
        sdf_init = sdf_init.unsqueeze(0)
    assert sdf_init.dim() == 3, "SDF must be of shape (T, H, W) or (H, W)"
    
    for i in range(sdf_init.shape[0]):
        # The fast marching uses finite differences, so it works better on a finer grid. 
        # Fortunately, the SDF is very smooth, so bicubic interpolation seems good enough.
        # So, this upsamples, does fast marching, and then downsamples.
        upsample_sdf_init = torch.nn.functional.interpolate(
            sdf_init[i].unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode="bicubic").squeeze()
        upsample_sdf_corrected = torch.from_numpy(fast_marching_2d(upsample_sdf_init.numpy(), dx=dx))
        sdf_corrected = torch.nn.functional.interpolate(
            upsample_sdf_corrected.unsqueeze(0).unsqueeze(0), scale_factor=1 / scale_factor, mode="bicubic").squeeze()
        # Only reinitialize the SDF when sufficiently far from the interfaces
        far_mask = sdf_init < far_threshold
        sdf = sdf_init.clone()
        sdf[i, far_mask] = sdf_corrected[far_mask]
    return sdf.squeeze(0)

def fast_marching_2d(phi_init, dx=1.0):
    ny, nx = phi_init.shape
    phi = phi_init.copy()
    
    status = np.zeros((ny, nx), dtype=int)
    
    # Find the interface (zero crossings)
    # Mark points near interface as known
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # Check if sign changes in neighborhood
            neighbors = [phi_init[i-1,j], phi_init[i+1,j], 
                        phi_init[i,j-1], phi_init[i,j+1]]
            if any([phi_init[i,j] * n < 0 for n in neighbors]):
                # Near interface - compute distance by linear interpolation
                # phi[i,j] = compute_interface_distance(phi_init, i, j, dx)
                
                # Near interface, keep distance unchanged (not updated phi[i, j])
                status[i,j] = KNOWN
    
    # Initialize trial points (neighbors of known points)
    heap = []
    for i in range(ny):
        for j in range(nx):
            if status[i,j] == KNOWN:
                # Add neighbors to trial
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < ny and 0 <= nj < nx and status[ni,nj] == FAR:
                        status[ni,nj] = TRIAL
                        phi[ni,nj] = solve_eikonal(phi, status, ni, nj, dx)
                        heapq.heappush(heap, (abs(phi[ni,nj]), ni, nj))
    
    # Fast marching main loop
    while heap:
        _, i, j = heapq.heappop(heap)
        
        if status[i,j] == KNOWN:  # Already processed
            continue
            
        status[i,j] = KNOWN  # Mark as known
        
        # Update neighbors
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < ny and 0 <= nj < nx and status[ni,nj] != KNOWN:
                phi_new = solve_eikonal(phi, status, ni, nj, dx)
                if status[ni,nj] == 0:  # Far -> Trial
                    status[ni,nj] = 1
                    phi[ni,nj] = phi_new
                    heapq.heappush(heap, (abs(phi_new), ni, nj))
                elif abs(phi_new) < abs(phi[ni,nj]): # Already trial, update if better
                    phi[ni,nj] = phi_new
                    heapq.heappush(heap, (abs(phi_new), ni, nj))
    return phi

def compute_interface_distance(phi, i, j, dx):
    """Compute distance to interface using linear interpolation."""
    val = phi[i,j]
    min_dist = abs(val)
    
    # Check each direction
    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
        ni, nj = i+di, j+dj
        if 0 <= ni < phi.shape[0] and 0 <= nj < phi.shape[1]:
            neighbor_val = phi[ni,nj]
            if val * neighbor_val < 0:  # Sign change
                # Linear interpolation to find zero crossing
                theta = abs(val) / (abs(val) + abs(neighbor_val))
                dist = theta * dx
                min_dist = min(min_dist, dist)
    
    return min_dist if val > 0 else -min_dist

def solve_eikonal(phi, status, i, j, dx):
    """Solve the Eikonal equation |∇phi| = 1 at grid point (i,j)."""
    ny, nx = phi.shape
    sign = 1 if phi[i,j] >= 0 else -1
    
    # Get upwind values in x-direction
    phi_x = []
    if i > 0 and status[i-1,j] == KNOWN:
        phi_x.append(phi[i-1,j])
    if i < ny-1 and status[i+1,j] == KNOWN:
        phi_x.append(phi[i+1,j])
    
    # Get upwind values in y-direction
    phi_y = []
    if j > 0 and status[i,j-1] == KNOWN:
        phi_y.append(phi[i,j-1])
    if j < nx-1 and status[i,j+1] == KNOWN:
        phi_y.append(phi[i,j+1])
    
    if not phi_x and not phi_y:
        return phi[i,j]  # No known neighbors
    
    # Choose closest known value in each direction
    a = min(phi_x, key=abs) if phi_x else np.inf
    b = min(phi_y, key=abs) if phi_y else np.inf
    
    # Solve quadratic: (phi-a)^2 + (phi-b)^2 = dx^2
    if abs(a) == np.inf and abs(b) == np.inf:
        return phi[i,j]
    elif abs(a) == np.inf:
        return b + sign * dx
    elif abs(b) == np.inf:
        return a + sign * dx
    else:
        # Check if 2D solution is valid
        discriminant = 2*dx**2 - (a-b)**2
        if discriminant >= 0:
            # Full 2D solve
            phi_new = (a + b + sign * np.sqrt(discriminant)) / 2
            # Check if solution satisfies upwind condition
            if abs(phi_new) >= max(abs(a), abs(b)):
                return phi_new
        # 1D solution
        return min(a, b, key=abs) + sign * dx