'''
Object loader to pyrender

Zilin Si (zsi@andrew.cmu.edu)
Last revision: Sept 2021
'''

import os
from os import path as osp
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import pyrender
import open3d as o3d

class ObjectLoader:
    def __init__(self, obj_path):

        # get object's mesh, vertices, normals
        mesh = o3d.io.read_triangle_mesh(obj_path)
        obj_trimesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
        # obj_trimesh = trimesh.load(obj_path)
        self.obj_mesh = pyrender.Mesh.from_trimesh(obj_trimesh)
        self.obj_vertices = obj_trimesh.vertices
        self.obj_normals = obj_trimesh.vertex_normals
        # initial the obj pose
        self.obj_pose = np.array([
            [1.0, 0.0,  0.0, 0.0],
            [0.0, 1.0,  0.0, 0.0],
            [0.0, 0.0,  1.0, 0.0],
            [0.0, 0.0,  0.0, 1.0],
        ])

if __name__ == "__main__":
    obj_name = "005_tomato_soup_can"
    obj_path = osp.join("..", "data", obj_name, "google_512k", "nontextured.stl")
    obj_loader = ObjectLoader(obj_path)
