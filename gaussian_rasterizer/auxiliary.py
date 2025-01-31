# -*- coding: utf-8 -*-
'''
@Time    : 4/5/24 
@Author  : Zhang Haoliang
'''
import numpy as np
from utils.graphics_utils import BasicPointCloud
class Gaussian3D:
    __slots__ = 'position_xyz', 'color_SH','opacity','rotation','scale'
    def __init__(self):
        self.position_xyz = [0,0,0]
        self.color_SH = []
        self.opacity = []
        self.rotation = []
        self.scale = []
    def create_from_pcd(self,point_cloud, cameras_extent):
        pass



class Camera:
    __slots__ = 'position_cam',
    def __init__(self):
        self.position_cam = [x,y,z]


