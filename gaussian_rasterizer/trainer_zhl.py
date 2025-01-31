# -*- coding: utf-8 -*-
'''
@Time    : 6/4/24 
@Author  : Zhang Haoliang

'''
import  numpy as np
import random
import utils.loss_utils as loss_utils
from torch.profiler import profile, ProfilerActivity
from gaussian_rasterizer.forward_rendering_zhl import GaussRender
from gaussian_rasterizer.load_data import readColmapSceneInfo,readNerfSyntheticInfo
from utils.gaussian_model import GaussianModel
import contextlib
import torch
from utils.camera import Camera
# Find Performance Bottlenecks
USE_PROFILE = False


class trainer_zhl():
    def __init__(self,train_scene,gaussian_points,**kwargs):
        self.train_scene = train_scene
        self.gaussian_points = gaussian_points
        self.num_iterations = kwargs.get("num_iter")
        self.lr = kwargs.get("lr")
        self.lambda_dssim = kwargs.get("lambda_dssim")
        self.tile_size = kwargs.get("tile_size")
        self.num_train_viewpoints = len(self.train_scene.train_cameras)
        self.viewpoint_stack = random.sample([i for i in range(self.num_train_viewpoints)], self.num_train_viewpoints)

        # render function
        self.gaussian_render = GaussRender()

        # self.init_num_gaussian_points = kwargs.get("init_num_gaussian_points")
        # self.max_num_gaussian_points = kwargs.get("max_num_gaussian_points")
    def one_train_step(self):
        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = random.sample([i for i in range(self.num_train_viewpoints)], self.num_train_viewpoints)
        ind = self.viewpoint_stack.pop()
        camera_viewpoint_info = self.train_scene.train_cameras[ind]
        gt_image = camera_viewpoint_info.image
        gt_image = torch.tensor(np.array(gt_image,dtype=np.float32)/255)

        # print("gt_image shape : ", np.array(gt_image).shape)
        # return gt_image
        if USE_PROFILE: prof = profile(activities=[ProfilerActivity.CUDA], with_stack=True)
        else: prof = contextlib.nullcontext()
        with prof:
            width, height, R, T, FoVx, FoVy = camera_viewpoint_info.width, camera_viewpoint_info.height, \
                                              camera_viewpoint_info.R,camera_viewpoint_info.T, \
                                              camera_viewpoint_info.FovX, camera_viewpoint_info.FovY
            # print(" camera_viewpoint_info.width : ",width," camera_viewpoint_info.height: ",height)
            camera_viewpoint = Camera(width, height, R, T, FoVx, FoVy)
            rets = self.gaussian_render.render(camera_viewpoint, self.gaussian_points,TILE_SIZE=self.tile_size)
            rendered_image = (rets["render"].detach().cpu())


        if USE_PROFILE:
            print(prof.key_averages(group_by_stack_n=True).table(sort_by='self_cuda_time_total', row_limit=20))

        l1_loss = loss_utils.l1_loss(rendered_image, gt_image)
        # depth_loss = loss_utils.l1_loss(out['depth'][..., 0][mask], depth[mask])
        ssim_loss = 1.0 - loss_utils.ssim(rendered_image, gt_image)

        total_loss = (1 - self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss
        """
        # if has depth loss
        total_loss = (1 - self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss + depth_loss * self.lambda_depth
        """

        psnr = loss_utils.img2psnr(rendered_image, gt_image)
        log_dict = {'total': total_loss, 'l1': l1_loss, 'ssim': ssim_loss, 'psnr': psnr}
        print(log_dict)
        # return total_loss, log_dict
        return rendered_image, gt_image

# trainer = trainer_zhl(**render_kwargs)
# render_kwargs.get('num_iter')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    real scene
    # scene_mip_nerf = "example_data/mip_nerf/bicycle"
    # scene_info = readColmapSceneInfo(testpath_mip_nerf)
    
    # scene_blender = "example_data/nerf_synthetic/chair"
    # scene_info = readNerfSyntheticInfo(testpath_blender)

    """
    render_kwargs = {
        'white_bkgd': True,
        'num_iterations': 2000,
        "lambda_dssim": 0.2,
        "lr": 1e-2,
        "init_num_gaussian_points": 1000,
        "max_num_gaussian_points": 2000,
        "tile_size": 100

    }
    scene_mip_nerf = "example_data/mip_nerf/bicycle"
    scene_info = readColmapSceneInfo(scene_mip_nerf)

    camera_info = scene_info.train_cameras[0]
    pc = GaussianModel(sh_degree= 4,max_num_gaussian_points=100)
    pc.create_from_pcd(scene_info.point_cloud, device)

    trainer = trainer_zhl(scene_info,pc,**render_kwargs)
    rendered_image, gt_image  = trainer.one_train_step()
    # rendered_image = rendered_image.astype()
    # print(rendered_image.shape,gt_image.shape)
    # img_arr = np.array(gt_img)

# gt_image = np.array(gt_image,dtype = np.float32)
# rendered_image = np.array(rendered_image)
#
# print(rendered_image.dtype,gt_image.dtype)
# ssim_loss = 1.0 - loss_utils.ssim(rendered_image, gt_image)
# import matplotlib.pyplot as plt
# gt_array = (255*rendered_image.numpy()).astype(int)
# plt.imshow(gt_array)
# plt.show()

