# Test code
from gaussian_rasterizer.load_data import readColmapSceneInfo,readNerfSyntheticInfo
from utils.gaussian_model import GaussianModel
import torch
import numpy as np
from utils.camera import Camera
from utils.sh_utils import eval_sh
from utils.render_utils import computeCov2D,projection_ndc,computeCov3D
import math
from gaussian_rasterizer.forward_rendering_zhl import GaussRender
# Init
testpath_mip_nerf = "example_data/mip_nerf/bicycle"
scene_info = readColmapSceneInfo(testpath_mip_nerf)
camera_info = scene_info.train_cameras[0]


# testpath_blender = "example_data/nerf_synthetic/chair"
# scene_info = readNerfSyntheticInfo(testpath_blender)
# camera_info = scene_info.train_cameras[0]
# img = np.array(camera_info.image)


pc = GaussianModel(3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pc.create_from_pcd(scene_info.point_cloud,device)


width, height, R, T, FoVx, FoVy = camera_info.width, camera_info.height, camera_info.R, camera_info.T, \
                                camera_info.FovX,camera_info.FovY
viewpoint_camera = Camera(width, height, R, T, FoVx, FoVy)

G_render = GaussRender()
rets = G_render.render(viewpoint_camera, pc,TILE_SIZE=180)


radii, means2D, rect = G_render.render_test(viewpoint_camera, pc)

im = (rets["render"].detach().cpu().numpy()*255).astype(int)

import matplotlib.pyplot as plt
plt.imshow(im)
plt.show()

plt.imshow( scene_info.train_cameras[0].image)
image = scene_info.train_cameras[0].image
plt.show()

import random
init_points = random.sample([i for i in range(10)], 5)

from PIL import Image
import numpy as np
image_path = "example_data/mip_nerf/bicycle/images/_DSC8679.JPG"
image = Image.open(image_path)
image_array = np.array(image)
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()
