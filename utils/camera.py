# -*- coding: utf-8 -*-
'''
@Time    : 4/11/24 
@Author  : Zhang Haoliang
'''
import torch
import numpy as np
from torch import nn
from utils.graphics_utils import getWorld2View2, getProjectionMatrix,fov2focal

# I following the original structure
# Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
#                   FoVx=cam_info.FovX, FoVy=cam_info.FovY,
#                   image=gt_image, gt_alpha_mask=loaded_mask,
#                   image_name=cam_info.image_name, uid=id, data_device=args.data_device)
class Camera(nn.Module):
    def __init__(self, width, height, R,T,FoVx, FoVy,znear=0.1, zfar=100.,\
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        super(Camera, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.znear = znear
        self.zfar = zfar
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height
        self.focal_x, self.focal_y = fov2focal(FoVx,width), fov2focal(FoVy,height)
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(device)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        # self.gt_image =
def parse_camera(params):
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2w = params[:, 18:34].reshape((-1, 4, 4))
    return H, W, intrinsics, c2w


def to_viewpoint_camera(camera):
    """
    Parse a camera of intrinsic and c2w into a Camera Object
    """
    device = camera.device
    Hs, Ws, intrinsics, c2ws = parse_camera(camera.unsqueeze(0))
    camera = Camera(width=int(Ws[0]), height=int(Hs[0]), intrinsic=intrinsics[0], c2w=c2ws[0])
    return camera

