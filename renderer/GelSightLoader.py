'''
GelSight loader to pyrender, including the camera & gelpad

Zilin Si (zsi@andrew.cmu.edu)
Last revision: Sept 2021
'''
import os
from os import path as osp
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import pyrender
from . import config

class GelSightLoader:
    def __init__(self, gel_path):
        # load gelpad models
        gel_trimesh = trimesh.load(gel_path)
        self.gel_mesh = pyrender.Mesh.from_trimesh(gel_trimesh)
        self.gel_init_pose = np.array([
            [1.0, 0.0,  0.0, 0.0],
            [0.0, 1.0,  0.0, 0.0],
            [0.0, 0.0,  1.0, 0.0],
            [0.0, 0.0,  0.0, 1.0],
        ])
        self.gel_pose = np.array([
            [1.0, 0.0,  0.0, 0.0],
            [0.0, 1.0,  0.0, 0.0],
            [0.0, 0.0,  1.0, 0.0],
            [0.0, 0.0,  0.0, 1.0],
        ])

        # generate camera models
        self.cam = pyrender.camera.IntrinsicsCamera(fx=config.cam_fx, fy=config.cam_fy, cx=config.cam_cx, cy=config.cam_cy, znear=config.znear, zfar=config.zfar, name=None)
        self.cam_init_pose = np.array([
            [1.0, 0.0,  0.0, 0.0],
            [0.0, 1.0,  0.0, 0.0],
            [0.0, 0.0,  1.0, config.cam2gel],
            [0.0, 0.0,  0.0, 1.0],
        ])
        self.cam_pose = np.array([
            [1.0, 0.0,  0.0, 0.0],
            [0.0, 1.0,  0.0, 0.0],
            [0.0, 0.0,  1.0, config.cam2gel],
            [0.0, 0.0,  0.0, 1.0],
        ])

    def update_cam(self, new_pose, press_depth):
        # new pose 4 * 4
        # press_depth in meter

        # get into the obj
        self.cam_pose = self.cam_init_pose.copy()
        self.cam_pose[2,3] -= press_depth
        self.gel_pose = self.gel_init_pose.copy()
        self.gel_pose[2,3] -= press_depth

        # transform to the corresponding vertix on object
        self.cam_pose = np.matmul(new_pose, self.cam_pose)
        self.gel_pose = np.matmul(new_pose, self.gel_pose)

        return

if __name__ == "__main__":
    gel_path = osp.join("..", "calibs", "gel_surface_mm.obj")
    gel_loader = GelSightLoader(gel_path)
