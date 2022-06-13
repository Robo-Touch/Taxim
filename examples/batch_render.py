'''
data generation

Zilin Si (zsi@andrew.cmu.edu)
Last revision: Sept 2021
'''
# python examples/batch_render.py 021_bleach_cleanser
import open3d as o3d
import os
from os import path as osp
import sys
import shutil
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from renderer.DepthRender import TacRender

import copy
from config import render_config

np.random.seed(0)

def batch_render(obj_name, viz=False):
    # change path to cwd
    abspath = osp.abspath(__file__)
    dname = osp.dirname(abspath)
    os.chdir(dname)

    obj_path = osp.join("..", "data", obj_name, "meshes", "model.obj")
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()

    # load gelsight
    gel_path = osp.join("..", "calibs", "gel_surface_mm.obj")
    # load calibration for taxim
    calib_path = osp.join("..", "calibs")
    # save results
    save_path = osp.join("..", "results", "batch", obj_name)
    if osp.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    os.makedirs(osp.join(save_path, "tactile_img"))
    os.makedirs(osp.join(save_path, "gt_depth"))
    os.makedirs(osp.join(save_path, "height_map"))
    os.makedirs(osp.join(save_path, "contact_mask"))

    os.makedirs(osp.join(save_path, "tactile_img_small"))
    os.makedirs(osp.join(save_path, "gt_depth_small"))
    os.makedirs(osp.join(save_path, "height_map_small"))
    os.makedirs(osp.join(save_path, "contact_mask_small"))

    tacRender = TacRender(obj_path, gel_path, calib_path)

    ### hyperparams ###
    # render_config['num_depths'] = 1
    render_config['num_total'] = render_config['num_vertices'] * render_config['num_angles'] * render_config['num_depths']
    # render_config['min_depth'] = 0.0005
    # render_config['max_depth'] = 0.0020

    # random shear direction
    theta = np.radians(render_config['shear_range'])
    orientations = []
    for i in range(render_config['num_angles']):
        z = np.random.uniform(low=np.cos(theta), high=1.0, size=(1,))[0]
        phi = np.random.uniform(low=0.0, high=2*np.pi, size=(1,))[0]
        z_axis = np.array([np.sqrt(1-z**2)*np.cos(phi), np.sqrt(1-z**2)*np.sin(phi), z])
        orientations.append(z_axis)

    # random penetration depth
    press_depths = np.random.uniform(low=render_config['min_depth'], high=render_config['max_depth'], size=(render_config['num_depths'],))
    # testing
    press_depths = [0.001]

    # random vertices on object
    total_vertices = len(tacRender.obj_loader.obj_vertices)
    print("there are total # " + str(total_vertices) + " vertices")
    vertices = np.random.randint(total_vertices, size=render_config['num_vertices'])

    tacRender.start_offline()
    start = time.time()

    poses = np.zeros([render_config['num_total'], 7]) # x, y, z, qx, qy, qz, qw
    sensorFrames = []
    c = 0
    # i_j_k.jpg = angle_depth_samplepoint
    for k in tqdm(range(render_config['num_vertices'])):
        for i in range(render_config['num_angles']):
            for j in range(render_config['num_depths']):
                z_axis = orientations[i]
                press_depth = press_depths[j]
                vertex_idx = vertices[k]
                # tacRender.update_pose(vertex_idx, press_depth, z_axis)
                pose = tacRender.update_pose(vertex_idx, press_depth, z_axis)

                depth = tacRender.render_once()
                depth, corrected_pose = tacRender.correct_contact(depth, vertex_idx, press_depth, z_axis, pose)
                # height_map_small, contact_mask_small, tactile_img_small = tacRender.taxim_small_render(depth, press_depth)
                # print(np.max(depth), np.min(depth))
                poses[c, :] = corrected_pose

                sensorFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=poses[c, 0:3])
                sensorFrameR = copy.deepcopy(sensorFrame)
                sensorR = sensorFrame.get_rotation_matrix_from_quaternion(poses[c, [6, 3, 4, 5]]) # [qw, qx, qy, qz]
                sensorFrameR.rotate(sensorR, center=poses[c, 0:3])
                sensorFrames.append(sensorFrameR)

                # height_map, contact_mask, tactile_img = tacRender.taxim_render(depth, press_depth)
                cv2.imwrite(save_path + "/gt_depth/" + str(c) + ".jpg", tacRender.correct_height_map(depth))
                # cv2.imwrite(save_path + "/height_map/" + str(c) + ".jpg", height_map)
                # cv2.imwrite(save_path + "/contact_mask/" + str(c) + ".jpg", 255*contact_mask.astype("uint8"))
                # cv2.imwrite(save_path + "/tactile_img/" + str(c) + ".jpg", tactile_img)
                # cv2.imwrite(save_path + "/gt_depth_small/" + str(c) + ".jpg", tacRender.correct_height_map_small(depth))
                # cv2.imwrite(save_path + "/height_map_small/" + str(c) + ".jpg", height_map_small)
                # cv2.imwrite(save_path + "/contact_mask_small/" + str(c) + ".jpg", 255*contact_mask_small.astype("uint8"))
                # cv2.imwrite(save_path + "/tactile_img_small/" + str(c) + ".jpg", tactile_img_small)
                c += 1
    end = time.time()
    tacRender.end_offline()
    # np.save(osp.join(save_path, "poses.npy"), poses)
    print("render time is " + str(end-start) + " s for " + str(render_config['num_angles']*render_config['num_depths']*render_config['num_vertices']) + " frames")

    if viz:
        o3d.visualization.draw_geometries([mesh] + sensorFrames)


if __name__ == "__main__":
    obj_name =  "Threshold_Porcelain_Teapot_White"
    batch_render(obj_name, viz=True)
