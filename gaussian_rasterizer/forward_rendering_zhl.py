# -*- coding: utf-8 -*-
'''
@Time    : 4/9/24
@Author  : Zhang Haoliang
'''
import torch
import math
from utils.sh_utils import eval_sh
from utils.gaussian_model import GaussianModel
from utils.render_utils import computeCov2D,projection_ndc,computeCov3D

@torch.no_grad()
def get_radius(cov2d):
    # Compute extent in screen space (by finding eigenvalues of
    # 2D covariance matrix).
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(lambda1).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])

    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)

    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max

class GaussRender:
    """
    Render the scene.
    """
    def __init__(self, active_sh_degree=3, white_bkgd=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.active_sh_degree = active_sh_degree
        # self.debug = False
        self.white_bkgd = white_bkgd
        # self.pix_coord = torch.stack(torch.meshgrid(torch.arange(800), torch.arange(800), indexing='xy'), dim=-1).to(
        #     self.device)

    def build_color(self, means3D, shs, camera):
        rays_o = camera.camera_center
        rays_d = means3D - rays_o
        color = eval_sh(self.active_sh_degree, shs.permute(0,2,1), rays_d)
        color = (color + 0.5).clip(min=0.0)
        return color

    def rasterizer(self, camera, means2D, cov2d, color, opacity, depths, TILE_SIZE = 100,device = "cpu"):
        radii = get_radius(cov2d)
        rect = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)

        self.pix_coord = torch.stack(torch.meshgrid(torch.arange(camera.image_width), torch.arange(camera.image_height),
                                                    indexing='xy'), dim=-1).to(
            self.device)
        # print(" camera.image_height : ", camera.image_height, " camera.image_width: ", camera.image_width,"self.pix_coord.shape: ",self.pix_coord.shape)

        self.render_color = torch.ones(*self.pix_coord.shape[:2], 3).to(device)
        # print("self.render_color shape",self.render_color.shape,"self.pix_coord.shape[:2]: ",*self.pix_coord.shape[:2])
        # self.render_color = torch.ones(camera.image_height,camera.image_width, 3).to(device)


        # self.render_depth = torch.zeros(*self.pix_coord.shape[:2], 1).to(device)
        # self.render_alpha = torch.zeros(*self.pix_coord.shape[:2], 1).to(device)

        # self.render_color = torch.ones(camera.image_height,camera.image_width, 3).to(device)
        # print(self.render_color.shape,color.shape,opacity.shape)

        for h in range(0, camera.image_height, TILE_SIZE):

            for w in range(0, camera.image_width, TILE_SIZE):
                # check if the rectangle penetrate the tile
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w + TILE_SIZE - 1), rect[1][..., 1].clip(max=h + TILE_SIZE - 1)
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])  # 3D gaussian in the tile
                # print(torch.count_nonzero(in_mask))
                if not in_mask.sum() > 0:
                    continue

                P = in_mask.sum()
                tile_coord = self.pix_coord[h:h + TILE_SIZE, w:w + TILE_SIZE].flatten(0, -2)
                sorted_depths, index = torch.sort(depths[in_mask])
                sorted_means2D = means2D[in_mask][index]
                sorted_cov2d = cov2d[in_mask][index]  # P 2 2
                sorted_conic = sorted_cov2d.inverse()  # inverse of variance
                sorted_opacity = opacity[in_mask][index]
                sorted_color = color[in_mask][index]

                dx = (tile_coord[:, None, :] - sorted_means2D[None, :])  # B P 2

                gauss_weight = torch.exp(-0.5 * (
                        dx[:, :, 0] ** 2 * sorted_conic[:, 0, 0]
                        + dx[:, :, 1] ** 2 * sorted_conic[:, 1, 1]
                        + dx[:, :, 0] * dx[:, :, 1] * sorted_conic[:, 0, 1]
                        + dx[:, :, 0] * dx[:, :, 1] * sorted_conic[:, 1, 0]))

                alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99)  # B P 1
                T = torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, 1:]], dim=1).cumprod(dim=1)
                acc_alpha = (alpha * T).sum(dim=1)
                tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1 - acc_alpha) * (
                    1 if self.white_bkgd else 0)
                # print("here tile_color finish: ", h)
                tile_depth = ((T * alpha) * sorted_depths[None, :, None]).sum(dim=1)
                # print(h,w,self.render_color[h:h + TILE_SIZE, w:w + TILE_SIZE].shape,tile_color.shape)
                need_shape = self.render_color[h:h + TILE_SIZE, w:w + TILE_SIZE].shape
                self.render_color[h:h + TILE_SIZE, w:w + TILE_SIZE] = tile_color.reshape(need_shape)

                # self.render_depth[h:h + TILE_SIZE, w:w + TILE_SIZE] = tile_depth.reshape(TILE_SIZE, TILE_SIZE, -1)
                # self.render_alpha[h:h + TILE_SIZE, w:w + TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)
        self.render_color = self.render_color.clamp(0.0, 1.0)
        return {
            "render": self.render_color,
            # "depth": self.render_depth,
            # "alpha": self.render_alpha,
            "visiility_filter": radii > 0,
            "radii": radii
        }
    def render(self,viewpoint_camera, pc : GaussianModel,TILE_SIZE = 100):

        # Set up rasterization configuration
        # project gaussian center to screen
        mean_ndc, mean_view, in_mask = projection_ndc(pc._xyz,
                                                      viewmatrix=viewpoint_camera.world_view_transform,
                                                      projmatrix=viewpoint_camera.projection_matrix)
        # print("before mask: ",mean_ndc.shape,mean_view.shape,in_mask.shape)
        # mean_ndc = mean_ndc[in_mask]
        # mean_view = mean_view[in_mask]
        depths = mean_view[:, 2]

        # print("after mask: ",mean_ndc.shape, mean_view.shape, in_mask.shape)
        # 2 build color
        color = self.build_color(means3D=pc._xyz, shs=pc.get_features, camera=viewpoint_camera)
        # build cov3d
        cov3d = computeCov3D(scale = pc._scaling, r = pc._rotation)
        # Compute 2D screen-space covariance matrix
        cov2d = computeCov2D(mean3d=pc._xyz,cov3d=cov3d,
            viewmatrix=viewpoint_camera.world_view_transform,
            fov_x=viewpoint_camera.FoVx,
            fov_y=viewpoint_camera.FoVy,
            focal_x=viewpoint_camera.focal_x,
            focal_y=viewpoint_camera.focal_y)

        # ndc2Pix
        mean_coord_x = ((mean_ndc[..., 0] + 1) * viewpoint_camera.image_width - 1.0) * 0.5
        mean_coord_y = ((mean_ndc[..., 1] + 1) * viewpoint_camera.image_height - 1.0) * 0.5
        means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)  # [N, 2]
        # print("arrive last step")
        # print("means2D: " ,means2D.shape , '\n cov2d.shape : ',cov2d.shape,"\n color shape: ",
        #       color.shape,"\n opacity shape: ", pc.get_opacity.shape)
        rets = self.rasterizer(viewpoint_camera, means2D, cov2d, color, opacity = pc.get_opacity, depths = depths,
                               TILE_SIZE = TILE_SIZE )
        return rets

    # def render_test(self, viewpoint_camera, pc: GaussianModel):
    #
    #     # Set up rasterization configuration
    #     # project gaussian center to screen
    #     mean_ndc, mean_view, in_mask = projection_ndc(pc._xyz,
    #                                                   viewmatrix=viewpoint_camera.world_view_transform,
    #                                                   projmatrix=viewpoint_camera.projection_matrix)
    #     print(mean_ndc.shape, mean_view.shape, in_mask.shape)
    #
    #     mean_ndc = mean_ndc[in_mask]
    #     mean_view = mean_view[in_mask]
    #     depths = mean_view[:, 2]
    #     print(mean_ndc.shape, mean_view.shape, in_mask.shape)
    #     # return
    #     # 2 build color
    #     color = self.build_color(means3D=pc._xyz, shs=pc.get_features, camera=viewpoint_camera)
    #     # build cov3d
    #     cov3d = computeCov3D(scale=pc._scaling, r=pc._rotation)
    #     # Compute 2D screen-space covariance matrix
    #     cov2d = computeCov2D(mean3d=pc._xyz, cov3d=cov3d,
    #                          viewmatrix=viewpoint_camera.world_view_transform,
    #                          fov_x=viewpoint_camera.FoVx,
    #                          fov_y=viewpoint_camera.FoVy,
    #                          focal_x=viewpoint_camera.focal_x,
    #                          focal_y=viewpoint_camera.focal_y)
    #
    #     # ndc2Pix
    #     mean_coord_x = ((mean_ndc[..., 0] + 1) * viewpoint_camera.image_width - 1.0) * 0.5
    #     mean_coord_y = ((mean_ndc[..., 1] + 1) * viewpoint_camera.image_height - 1.0) * 0.5
    #     means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)  # [N, 2]
    #
    #     # start Render
    #     radii = get_radius(cov2d)[in_mask]
    #     print(means2D.shape, cov2d.shape,radii.shape,torch.count_nonzero(in_mask))
    #
    #     rect = get_rect(means2D, radii, width=viewpoint_camera.image_width, height=viewpoint_camera.image_height)
    #     # return radii, means2D, rect
    #     # #
    #     # render_image = torch.ones(*self.pix_coord.shape[:2],  3,device = self.device)
    #     # self.render_depth = torch.zeros(*self.pix_coord.shape[:2], 1,device = device)
    #     # self.render_alpha = torch.zeros(*self.pix_coord.shape[:2], 1,device = device)
    #     # rets = self.render(
    #     #     camera=camera,
    #     #     means2D=means2D,
    #     #     cov2d=cov2d,
    #     #     color=color,
    #     #     opacity=opacity,
    #     #     depths=depths,
    #     # )
    #
    #     # rets = self.rasterizer(viewpoint_camera, means2D, cov2d, color, opacity=pc.get_opacity, depths=depths)
    #     # return rets



