from typing import List, Union
import torch
import torch.nn as nn
import math
import numpy as np

def eikonal_loss(phi):
    """
    phi = SDF torch.Tensor (B, T, H, W)
    """
    dx = 1/32
    grad_phi_y, grad_phi_x = torch.gradient(phi, spacing=dx, dim=(-2, -1), edge_order=1)
    grad_mag = torch.sqrt(grad_phi_y**2 + grad_phi_x**2)

    eikonal_mse = (grad_mag - 1.0) ** 2

    return eikonal_mse.mean()

class LpLoss(nn.Module):
    """
    Lp loss on a tensor (b, n1, n2, ..., nd)
    Args:
        d (int): Number of dimensions to flatten from right
        p (int): Power of the norm
        reduce_dims (List[int]): Dimensions to reduce
        reductions (List[str]): Reductions to apply
    """
    def __init__(
            self,
            d: int = 1,
            p: int = 2,
            reduce_dims: Union[int, List[int]] = 0,
            reductions: Union[str, List[str]] = "sum"
        ):
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == "sum" or reductions == "mean"
                self.reductions = [reductions] * len(self.reduce_dims)
            else:
                for reduction in reductions:
                    assert reduction == "sum" or reduction == "mean"
                self.reductions = reductions

    def reduce_all(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduce the tensor along the specified dimensions
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Reduced tensor
        """
        for j, reduce_dim in enumerate(self.reduce_dims):
            if self.reductions[j] == "sum":
                x = torch.sum(x, dim=reduce_dim, keepdim=True)
            else:
                x = torch.mean(x, dim=reduce_dim, keepdim=True)
        return x

    def forward(
            self,
            y_pred: torch.Tensor,
            y: torch.Tensor
        ) -> torch.Tensor:
        """
        Args:
            y_pred (torch.Tensor): Predicted tensor
            y (torch.Tensor): Target tensor
        Returns:
            torch.Tensor: Lp loss
        """
        diff = torch.norm(
            torch.flatten(y_pred, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d),
            p=self.p,
            dim=-1,
            keepdim=False,
        )
        ynorm = torch.norm(
            torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False
        )

        diff = diff / ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff


def smoothed_sdf(sdf, smear):
    """
    Args: 
        sdf (np.ndarray or torch.Tensor): Signed distance function field
        smear (Float): upper and lower bound of SDF values that should be smeared (if smear = 1, any |phi| values <= smear will be continuous, rest set to 1 or 0)
    """

    H = torch.empty_like(sdf)
    band = sdf.abs() <= smear

    sdf_band = sdf[band]
    H_band = (
        0.5
        + sdf_band / (2.0 / smear)
        + torch.sin(math.pi * sdf_band / smear) / (2.0 * math.pi)
    )
    H[band] = H_band

    H[sdf > smear] = 1.0
    H[sdf < -smear] = 0.0

    return H

def grad_finite_diff(u, dx=1.0, dy=1.0):
    
    ux = np.empty_like(u)
    uy = np.empty_like(u)

      # ----- d/dx (width dimension = last dim) -----
    ux[..., :, 1:-1] = (u[..., :, 2:] - u[..., :, :-2]) / (2 * dx)
    ux[..., :, 0]    = (u[..., :, 1]  - u[..., :, 0])   / dx
    ux[..., :, -1]   = (u[..., :, -1] - u[..., :, -2])  / dx

    # ----- d/dy (height dimension = second-last) -----
    uy[..., 1:-1, :] = (u[..., 2:, :] - u[..., :-2, :]) / (2 * dy)
    uy[..., 0, :]    = (u[..., 1, :]  - u[..., 0, :])    / dy
    uy[..., -1, :]   = (u[..., -1, :] - u[..., -2, :])   / dy

    return ux, uy

def div_u(ux, uy, dx=1.0, dy=1.0):
    # dudx from ux along x (last dim)
    dudx = np.empty_like(ux)
    dudx[..., :, 1:-1] = (ux[..., :, 2:] - ux[..., :, :-2]) / (2 * dx)
    dudx[..., :, 0]    = (ux[..., :, 1]  - ux[..., :, 0])   / dx
    dudx[..., :, -1]   = (ux[..., :, -1] - ux[..., :, -2])  / dx

    # dvdy from uy along y (second-last dim)
    dvdy = np.empty_like(uy)
    dvdy[..., 1:-1, :] = (uy[..., 2:, :] - uy[..., :-2, :]) / (2 * dy)
    dvdy[..., 0, :]    = (uy[..., 1, :]  - uy[..., 0, :])    / dy
    dvdy[..., -1, :]   = (uy[..., -1, :] - uy[..., -2, :])   / dy

    return dudx + dvdy


"""
def div_u(ux, uy, dx=1.0, dy=1.0):
    #ux: [..., H,   W+1]  (x-face velocities)
    #uy: [..., H+1, W  ]  (y-face velocities)
    #returns divergence at cell centers [..., H, W]
    # Difference on faces → cell centers
    dudx = (ux[..., :, 1:] - ux[..., :, :-1]) / dx
    dvdy = (uy[..., 1:, :] - uy[..., :-1, :]) / dy

    return dudx + dvdy
"""


