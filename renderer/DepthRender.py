'''
GelSight tactile/depth render with pyrender + taxim

Zilin Si (zsi@andrew.cmu.edu)
Last revision: Sept 2021
'''

import os
from os import path as osp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import torch
import pyrender
from .GelSightLoader import GelSightLoader
from .ObjectLoader import ObjectLoader
from . import config
from .utils import *

from .basics import sensorParams as psp
from .basics.CalibData import CalibData

class TacRender:

    def __init__(self, obj_path, gel_path, calib_path):
        # load object and gelsight model
        self.obj_loader = ObjectLoader(obj_path)
        self.gel_loader = GelSightLoader(gel_path)
        # create a pyrender scene
        self.scene = pyrender.Scene(ambient_light=np.array([0.2, 0.2, 0.2, 1.0]))
        # create nodes in the scene
        self.obj_node = self.scene.add(self.obj_loader.obj_mesh, pose=self.obj_loader.obj_pose)
        self.gel_node = self.scene.add(self.gel_loader.gel_mesh, pose=self.gel_loader.gel_pose)
        self.cam_node = self.scene.add(self.gel_loader.cam, pose=self.gel_loader.cam_pose)
        # taxim calibration files
        # polytable
        calib_data = osp.join(calib_path, "polycalib.npz")
        self.calib_data = CalibData(calib_data)
        # raw calibration data
        rawData = osp.join(calib_path, "dataPack.npz")
        data_file = np.load(rawData, allow_pickle=True)
        self.f0 = data_file['f0']
        self.bg_proc = processInitialFrame(self.f0)

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        ## tactile image config
        bins = psp.numBins
        [xx, yy] = np.meshgrid(range(psp.w), range(psp.h))
        xf = xx.flatten()
        yf = yy.flatten()
        self.A = np.array([xf*xf,yf*yf,xf*yf,xf,yf,np.ones(psp.h*psp.w)]).T
        binm = bins - 1
        self.x_binr = 0.5*np.pi/binm # x [0,pi/2]
        self.y_binr = 2*np.pi/binm # y [-pi, pi]

        self.A = torch.from_numpy(self.A).to(self.device)
        # self.x_binr = torch.from_numpy(self.x_binr).to(self.device)
        # self.y_binr = torch.from_numpy(self.y_binr).to(self.device)
        self.calib_data.grad_r = torch.from_numpy(self.calib_data.grad_r).to(self.device)
        self.calib_data.grad_g = torch.from_numpy(self.calib_data.grad_g).to(self.device)
        self.calib_data.grad_b = torch.from_numpy(self.calib_data.grad_b).to(self.device)

        ### first time with a new gelpad model, generate new bg depth & color
        self.raw_bg = self.render_bg()
        # self.bg_depth = self.correct_height_map(self.render_bg())
        # np.save(osp.join(calib_path, "depth_bg.npy"),self.bg_depth)
        # self.real_bg = self.taxim_render_bg(self.bg_depth)
        # np.save(osp.join(calib_path, "real_bg.npy"), self.real_bg)
        ###

        # load depth bg
        self.bg_depth = np.load(osp.join(calib_path,"depth_bg.npy"), allow_pickle=True)
        # load tactile bg
        self.real_bg = np.load(osp.join(calib_path,"real_bg.npy"), allow_pickle=True)

        # small image config
        # polytable
        calib_data_small = osp.join(calib_path, "polycalib_small.npz")
        self.calib_data_small = CalibData(calib_data_small)
        # raw calibration data
        rawData_small = osp.join(calib_path, "dataPack_small.npz")
        data_file_small = np.load(rawData_small, allow_pickle=True)
        self.f0_small = data_file_small['f0']
        self.sim_bg_small = np.load(osp.join(calib_path, 'real_bg_small.npy'))

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        ## tactile image config
        bins = psp.numBins_small
        [xx, yy] = np.meshgrid(range(psp.w_small), range(psp.h_small))
        xf = xx.flatten()
        yf = yy.flatten()
        self.A_small = np.array([xf*xf,yf*yf,xf*yf,xf,yf,np.ones(psp.h_small*psp.w_small)]).T
        binm = bins - 1
        self.x_binr_small = 0.5*np.pi/binm # x [0,pi/2]
        self.y_binr_small = 2*np.pi/binm # y [-pi, pi]

        self.A_small = torch.from_numpy(self.A_small).to(self.device)
        self.calib_data_small.grad_r = torch.from_numpy(self.calib_data_small.grad_r).to(self.device)
        self.calib_data_small.grad_g = torch.from_numpy(self.calib_data_small.grad_g).to(self.device)
        self.calib_data_small.grad_b = torch.from_numpy(self.calib_data_small.grad_b).to(self.device)

        ### first time with a new gelpad model, generate new bg depth & color
        # self.bg_depth = self.correct_height_map(self.render_bg())
        # np.save(osp.join(calib_path, "depth_bg.npy"),self.bg_depth)
        # self.real_bg = self.taxim_render_bg(self.bg_depth)
        # np.save(osp.join(calib_path, "real_bg.npy"), self.real_bg)
        ###
        # load depth bg
        # load tactile bg
        self.bg_depth_small = cv2.resize(self.bg_depth, dsize=(psp.w_small, psp.h_small), interpolation=cv2.INTER_CUBIC)
        self.real_bg_small = np.load(osp.join(calib_path,"real_bg_small.npy"), allow_pickle=True)

    def correct_height_map(self, height_map):
        # move the center of depth to the origin
        height_map = (height_map-config.cam2gel) * -1000 / psp.pixmm
        return height_map

    def correct_height_map_small(self, height_map):
        # move the center of depth to the origin
        height_map = (height_map-config.cam2gel) * -1000 / psp.pixmm_small
        return height_map

    def taxim_render_bg(self, depth):
        height_map = depth.copy()
        # get gradients
        grad_mag, grad_dir = generate_normals(height_map)
        # simulate raw image
        sim_img_r = np.zeros((psp.h,psp.w,3))
        idx_x = np.floor(grad_mag/self.x_binr).astype('int')
        idx_y = np.floor((grad_dir+np.pi)/self.y_binr).astype('int')

        params_r = self.calib_data.grad_r[idx_x,idx_y,:]
        params_r = params_r.reshape((psp.h*psp.w), params_r.shape[2])
        params_g = self.calib_data.grad_g[idx_x,idx_y,:]
        params_g = params_g.reshape((psp.h*psp.w), params_g.shape[2])
        params_b = self.calib_data.grad_b[idx_x,idx_y,:]
        params_b = params_b.reshape((psp.h*psp.w), params_b.shape[2])

        est_r = np.sum(self.A * params_r,axis = 1)
        est_g = np.sum(self.A * params_g,axis = 1)
        est_b = np.sum(self.A * params_b,axis = 1)
        sim_img_r[:,:,0] = est_r.reshape((psp.h,psp.w))
        sim_img_r[:,:,1] = est_g.reshape((psp.h,psp.w))
        sim_img_r[:,:,2] = est_b.reshape((psp.h,psp.w))
        # get the bg w/o dome shape
        real_bg = self.bg_proc - sim_img_r
        return real_bg

    def render_bg(self):
        return self.offline_render()

    def update_pose(self, idx, press_depth, z_axis):
        # idx: the idx vertice
        # get a new pose
        new_position = self.obj_loader.obj_vertices[idx].copy()
        new_orientation = self.obj_loader.obj_normals[idx].copy()

        new_pose = gen_pose(new_position, new_orientation, z_axis)

        return self.update_pose_from_pose(press_depth, new_pose)

    def update_pose_from_pose(self, press_depth, new_pose):
        # new pose: 4 x 4
        # reset the gelsight cam's and gelpad's pose
        self.gel_loader.update_cam(new_pose, press_depth)
        self.scene.set_pose(self.cam_node, pose=self.gel_loader.cam_pose)
        self.scene.set_pose(self.gel_node, pose=self.gel_loader.gel_pose)
        return gen_t_quat(self.gel_loader.gel_pose)

    def live_render(self):
        pyrender.Viewer(self.scene, use_raymond_lighting=True)

    def start_offline(self):
        self.r = pyrender.OffscreenRenderer(viewport_width=config.img_width, viewport_height=config.img_height)
    def end_offline(self):
        self.r.delete()
    def render_once(self):
        flags = pyrender.constants.RenderFlags.DEPTH_ONLY
        depth = self.r.render(self.scene, flags = flags)
        return depth

    def correct_contact(self, depth, vertex_idx, press_depth, z_axis, pose):
        diff = np.abs(depth - self.raw_bg)
        contact_mask = diff > 1e-6
        try:
            max_value = np.max(diff[contact_mask])
        except:
            print(np.sum(contact_mask*1))
        cur_press_depth = press_depth
        while max_value > press_depth:
            cur_press_depth = cur_press_depth - max_value + press_depth
            pose = self.update_pose(vertex_idx, cur_press_depth, z_axis)
            depth = self.render_once()
            diff = np.abs(depth - self.raw_bg)
            contact_mask = diff > 1e-6
            # max_value = np.max(diff[contact_mask])
            try:
                max_value = np.max(diff[contact_mask])
            except:
                print(cur_press_depth)
                print(np.sum(contact_mask*1))
        return depth, pose


    def offline_render(self):
        r = pyrender.OffscreenRenderer(viewport_width=config.img_width, viewport_height=config.img_height)
        flags = pyrender.constants.RenderFlags.DEPTH_ONLY
        # start = time.time()
        depth = r.render(self.scene, flags = flags)
        # end = time.time()
        r.delete()
        # print("render time is" + str(end-start))
        return depth

    def taxim_render(self, depth, press_depth):
        # start = time.time()
        depth = self.correct_height_map(depth)
        height_map = depth.copy()

        ## generate contact mask
        pressing_height_pix = press_depth * 1000 / psp.pixmm
        contact_mask = (height_map-(self.bg_depth)) > pressing_height_pix * 0.2

        # smooth out the soft contact
        zq_back = height_map.copy()
        kernel_size = [51,31,21,11,5]
        for k in range(len(kernel_size)):
            height_map = cv2.GaussianBlur(height_map.astype(np.float32),(kernel_size[k],kernel_size[k]),0)
            height_map[contact_mask] = zq_back[contact_mask]
        height_map = cv2.GaussianBlur(height_map.astype(np.float32),(5,5),0)

        # generate gradients
        grad_mag, grad_dir = generate_normals(height_map)
        # end = time.time()
        # print("height map process time is ", end-start)

        # simulate raw image
        # start = time.time()
        grad_mag = torch.from_numpy(grad_mag).to(self.device)
        grad_dir = torch.from_numpy(grad_dir).to(self.device)

        sim_img_r = np.zeros((psp.h,psp.w,3))
        idx_x = torch.floor(grad_mag/self.x_binr).long()
        idx_y = torch.floor((grad_dir+np.pi)/self.y_binr).long()

        params_r = self.calib_data.grad_r[idx_x,idx_y,:]
        params_r = params_r.reshape((psp.h*psp.w), params_r.shape[2])
        params_g = self.calib_data.grad_g[idx_x,idx_y,:]
        params_g = params_g.reshape((psp.h*psp.w), params_g.shape[2])
        params_b = self.calib_data.grad_b[idx_x,idx_y,:]
        params_b = params_b.reshape((psp.h*psp.w), params_b.shape[2])

        est_r = torch.sum(self.A * params_r,axis = 1)
        est_g = torch.sum(self.A * params_g,axis = 1)
        est_b = torch.sum(self.A * params_b,axis = 1)
        sim_img_r[:,:,0] = est_r.reshape((psp.h,psp.w)).numpy()
        sim_img_r[:,:,1] = est_g.reshape((psp.h,psp.w)).numpy()
        sim_img_r[:,:,2] = est_b.reshape((psp.h,psp.w)).numpy()

        # add back ground
        tactile_img = sim_img_r + self.real_bg
        # end = time.time()
        # print("simulate time is ", end-start)
        return height_map, contact_mask, tactile_img

    def taxim_small_render(self, depth, press_depth):
        # start = time.time()
        depth = cv2.resize(depth, dsize=(psp.w_small, psp.h_small), interpolation=cv2.INTER_CUBIC)
        depth = self.correct_height_map_small(depth)
        height_map = depth.copy()

        ## generate contact mask
        pressing_height_pix = press_depth * 1000 / psp.pixmm_small
        contact_mask = (height_map-(self.bg_depth_small)) > pressing_height_pix * 0.2

        # smooth out the soft contact
        zq_back = height_map.copy()
        kernel_size = [21,11,5]
        for k in range(len(kernel_size)):
            height_map = cv2.GaussianBlur(height_map.astype(np.float32),(kernel_size[k],kernel_size[k]),0)
            height_map[contact_mask] = zq_back[contact_mask]
        # height_map = cv2.GaussianBlur(height_map.astype(np.float32),(5,5),0)

        # generate gradients
        grad_mag, grad_dir = generate_normals(height_map)
        # end = time.time()
        # print("height map process time is ", end-start)

        # simulate raw image
        # start = time.time()
        grad_mag = torch.from_numpy(grad_mag).to(self.device)
        grad_dir = torch.from_numpy(grad_dir).to(self.device)

        sim_img_r = np.zeros((psp.h_small,psp.w_small,3))
        idx_x = torch.floor(grad_mag/self.x_binr_small).long()
        idx_y = torch.floor((grad_dir+np.pi)/self.y_binr_small).long()

        params_r = self.calib_data_small.grad_r[idx_x,idx_y,:]
        params_r = params_r.reshape((psp.h_small*psp.w_small), params_r.shape[2])
        params_g = self.calib_data_small.grad_g[idx_x,idx_y,:]
        params_g = params_g.reshape((psp.h_small*psp.w_small), params_g.shape[2])
        params_b = self.calib_data_small.grad_b[idx_x,idx_y,:]
        params_b = params_b.reshape((psp.h_small*psp.w_small), params_b.shape[2])

        est_r = torch.sum(self.A_small * params_r,axis = 1)
        est_g = torch.sum(self.A_small * params_g,axis = 1)
        est_b = torch.sum(self.A_small * params_b,axis = 1)
        sim_img_r[:,:,0] = est_r.reshape((psp.h_small,psp.w_small)).numpy()
        sim_img_r[:,:,1] = est_g.reshape((psp.h_small,psp.w_small)).numpy()
        sim_img_r[:,:,2] = est_b.reshape((psp.h_small,psp.w_small)).numpy()

        # add back ground
        tactile_img = sim_img_r + self.real_bg_small
        # end = time.time()
        # print("simulate time is ", end-start)
        return height_map, contact_mask, tactile_img

if __name__ == "__main__":
    # load object
    obj_name = "005_tomato_soup_can"
    obj_path = osp.join("..", "data", obj_name, "google_512k", "nontextured.stl")

    # load gelsight
    gel_path = osp.join("..", "calibs", "gel_surface_mm.obj")
    # load calibration for taxim
    calib_path = osp.join("..", "calibs")
    # save results
    save_path = osp.join("..", "results", "tests")

    # whether use live render or offline render
    # live = True
    live = False

    tacRender = TacRender(obj_path, gel_path, calib_path)
    # define pressing depth
    press_depth = 0.0020 # in meter
    vertix_idx = 10
    tacRender.update_pose(vertix_idx, press_depth)
    if live == True:
        tacRender.live_render()
    else:
        depth = tacRender.offline_render()
        height_map, contact_mask, tactile_img = tacRender.taxim_render(depth, press_depth)
        cv2.imwrite(save_path+"/height_"+str(press_depth*1000)+"_"+obj_name+".jpg", height_map)
        cv2.imwrite(save_path+"/contact_mask_"+str(press_depth*1000)+"_"+obj_name+".jpg", 255*contact_mask.astype("uint8"))
        cv2.imwrite(save_path+"/tactile_"+str(press_depth*1000)+"_"+obj_name+".jpg", tactile_img)

    # plt.figure(1)
    # plt.subplot(311)
    # plt.imshow(height_map)
    #
    # plt.subplot(312)
    # plt.imshow(contact_mask)
    #
    # plt.subplot(313)
    # plt.imshow(tactile_img/255.0)
    # plt.show()
