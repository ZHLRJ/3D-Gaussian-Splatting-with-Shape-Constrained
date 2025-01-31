# -*- coding: utf-8 -*-
'''
@Time    : 4/2/24 
@Author  : Zhang Haoliang
'''
import math
import numpy as np
import torch
def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.0000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= 0.2
    return p_proj, p_view, in_mask

# Compute rotation matrix from quaternion
def quaternion2rotation(r,device = "cpu"):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
def build_scaling_rotation(scale, r, device = "cpu"):
    # M = RS
    S = torch.zeros((scale.shape[0], 3, 3), dtype=torch.float, device=device)
    R = quaternion2rotation(r)

    S[:,0,0] = scale[:,0]
    S[:,1,1] = scale[:,1]
    S[:,2,2] = scale[:,2]

    M = R @ S
    return M
def computeCov3D(scale, r):
    L = build_scaling_rotation(scale, r)
    Sigma = L @ L.transpose(1, 2)
    return Sigma
# Compute 2D screen-space covariance matrix
def computeCov2D(mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y):
    # The following models the steps outlined by equations 29
    # and 31 in "EWA Splatting" (Zwicker et al., 2002).
    # Additionally considers aspect / scaling of viewport.
    # Transposes used to account for row-/column-major conventions.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)

    t = (mean3d @ viewmatrix[:3, :3]) + viewmatrix[-1:, :3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx * 1.3, max=tan_fovx * 1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy * 1.3, max=tan_fovy * 1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = focal_x / tz
    J[..., 0, 2] = -(focal_x * tx) / (tz * tz)
    J[..., 1, 1] = focal_y / tz
    J[..., 1, 2] = -(focal_y * ty) / (tz * tz)
    W = viewmatrix[:3, :3].T  # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0, 2, 1)

    # add low pass filter here according to E.q. 32 Apply low-pass filter:
    # every Gaussian should be at least, one pixel wide/high. Discard 3rd row and column.

    filter = torch.eye(2, 2).to(cov2d) * 0.3
    return cov2d[:, :2, :2] + filter[None]











