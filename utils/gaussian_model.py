# -*- coding: utf-8 -*-
'''
@Time    : 4/10/24 
@Author  : Zhang Haoliang
'''
import torch
import numpy as np
from torch import nn
import random
from scipy.spatial import KDTree
from utils.sh_utils import RGB2SH
from utils.general_utils import inverse_sigmoid
from utils.graphics_utils import BasicPointCloud

def distCUDA2(points):
    # https://github.com/graphdeco-inria/gaussian-splatting/issues/292
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)
    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)



class GaussianModel:
    def __init__(self, sh_degree: int,max_num_gaussian_points = 1000):
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.opacity_activation = torch.sigmoid

        self.max_num_gaussian_points = max_num_gaussian_points


        pass
    def create_from_pcd(self, pcd : BasicPointCloud, device = "cpu"):
        n_pcb_points = pcd.points.shape[0]
        init_points = pcd.points

        if n_pcb_points > self.max_num_gaussian_points:
            init_points_idx = random.sample([i for i in range(n_pcb_points)], self.max_num_gaussian_points)
            init_points = pcd.points[init_points_idx]
        # print(init_points.shape)

        fused_point_cloud = torch.tensor(np.asarray(init_points)).float()
        fused_color = RGB2SH(torch.tensor(np.asarray(init_points)).float())

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
                               dtype=torch.float32,device = device)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape)

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(init_points)).float()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)  # 旋转参数, 四元组
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros(self._xyz.shape[0], device=device)   # 投影到2D时, 每个2D gaussian最大的半径

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)



