import matplotlib.pyplot as plt
import numpy as np
import os
from os import path as osp
import cv2
from scipy.optimize import nnls
from scipy.ndimage import correlate
import scipy.ndimage as ndimage
from scipy import interpolate

import sys
sys.path.append("..")
from compose.dataLoader import dataLoader
sys.path.append("../..")
import Basics.sensorParams as psp
import Basics.params as pr

def cropMap(deformMap):
    # from d*d to h*w
    channel = deformMap.shape[0]
    croppedMap = np.zeros((channel,psp.h,psp.w))
    for c in range(channel):
        croppedMap[c,:,:] = deformMap[c,psp.d//2-psp.h//2:psp.d//2+psp.h//2,psp.d//2-psp.w//2:psp.d//2+psp.w//2]
    return croppedMap

def fill_blank(img):
    # here we assume there are some zero value holes in the image,
    # and we hope to fill these holes with interpolation
    x = np.arange(0, img.shape[1])
    y = np.arange(0, img.shape[0])
    #mask invalid values
    array = np.ma.masked_where(img == 0, img)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = img[~array.mask]

    GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                              (xx, yy),
                                 method='linear', fill_value = 0) # cubic # nearest # linear
    return GD1


class SuperPosition:
    def __init__(self, data_folder):
        femData = np.load(data_folder, allow_pickle=True)
        self.tensorMap = femData['tensorMap']
        self.sparse_mask = femData['nodeMask']

    def correct_KeyX(self, contact_points, local_deform, uz):
        """ get the virtual loads on X-axis """
        resultMap = np.zeros((3,psp.d,psp.d))
        local_kx = psp.d//2
        local_ky = psp.d//2

        num_points = contact_points[0].shape[0]

        M = np.zeros((num_points,num_points))
        u = local_deform
        min_z = np.min(uz)
        for i in range(contact_points[0].shape[0]):
            qx = contact_points[1][i]
            qy = contact_points[0][i]
            # vectorize
            kx_list = contact_points[1]
            ky_list = contact_points[0]
            cur_x_list = qx-kx_list+local_kx
            cur_y_list = qy-ky_list+local_ky
            mask_x = (cur_x_list >= 0) & (cur_x_list < psp.d)
            mask_y = (cur_y_list >= 0) & (cur_y_list < psp.d)
            mask_valid = mask_x & mask_y
            T_list = np.zeros((num_points,3,3))
            T_list[mask_valid,:,:] = self.tensorMap[cur_y_list[mask_valid],cur_x_list[mask_valid],:,:]
            uz_valid = uz[mask_valid]

            for j, T in enumerate(T_list):
                M[i,j] = T[0,0]
        # non-negative least square
        solution = nnls(M,-1*u)
        u_corrected = solution[0]
        u_scaled = u_corrected * uz/min_z
        scale = np.mean(np.abs(np.dot(M,u_scaled)+u))
        for i in range(contact_points[0].shape[0]):
            qx = contact_points[1][i]
            qy = contact_points[0][i]
            resultMap[0,qy,qx] = -1 * u_scaled[i] * scale
        return resultMap

    def correct_KeyY(self, contact_points, local_deform, uz):
        """ get the virtual loads on Y-axis """
        resultMap = np.zeros((3,psp.d,psp.d))
        local_kx = psp.d//2
        local_ky = psp.d//2

        num_points = contact_points[0].shape[0]

        M = np.zeros((num_points,num_points))
        u = local_deform
        min_z = np.min(uz)
        for i in range(contact_points[0].shape[0]):
            qx = contact_points[1][i]
            qy = contact_points[0][i]
            # vectorize
            kx_list = contact_points[1]
            ky_list = contact_points[0]
            cur_x_list = qx-kx_list+local_kx
            cur_y_list = qy-ky_list+local_ky
            mask_x = (cur_x_list >= 0) & (cur_x_list < psp.d)
            mask_y = (cur_y_list >= 0) & (cur_y_list < psp.d)
            mask_valid = mask_x & mask_y
            T_list = np.zeros((num_points,3,3))
            T_list[mask_valid,:,:] = self.tensorMap[cur_y_list[mask_valid],cur_x_list[mask_valid],:,:]
            uz_valid = uz[mask_valid]

            for j, T in enumerate(T_list):
                M[i,j] = T[1,1]
        solution = nnls(M,-1*u)
        u_corrected = solution[0]
        u_scaled = u_corrected * uz/min_z
        scale = np.mean(np.abs(np.dot(M,u_scaled)+u))
        for i in range(contact_points[0].shape[0]):
            qx = contact_points[1][i]
            qy = contact_points[0][i]
            resultMap[1,qy,qx] = -1 * u_scaled[i] * scale
        return resultMap

    def correct_KeyZ(self, contact_points, local_deform):
        """ get the virtual loads on Z-axis """
        resultMap = np.zeros((3,psp.d,psp.d))
        local_kx = psp.d//2
        local_ky = psp.d//2

        num_points = contact_points[0].shape[0]
        M = np.zeros((num_points,num_points))
        u = local_deform

        for i in range(contact_points[0].shape[0]):
            qx = contact_points[1][i]
            qy = contact_points[0][i]
            # vectorize
            kx_list = contact_points[1]
            ky_list = contact_points[0]
            cur_x_list = qx-kx_list+local_kx
            cur_y_list = qy-ky_list+local_ky
            mask_x = (cur_x_list >= 0) & (cur_x_list < psp.d)
            mask_y = (cur_y_list >= 0) & (cur_y_list < psp.d)
            mask_valid = mask_x & mask_y
            T_list = np.zeros((num_points,3,3))
            T_list[mask_valid,:,:] = self.tensorMap[cur_y_list[mask_valid],cur_x_list[mask_valid],:,:]

            for j, T in enumerate(T_list):
                M[i,j] = T[2,2]
        solution = nnls(M,-1*u)
        u_corrected = solution[0]
        for i in range(contact_points[0].shape[0]):
            qx = contact_points[1][i]
            qy = contact_points[0][i]
            resultMap[2,qy,qx] = -1*u_corrected[i]

        return resultMap

    def compose_sparse(self, local_deform, gel_map, contact_mask):
        """
        local deform: (3,1) dx, dy, dz initial displacement
        gel map: height map
        contact_mask: indicate contact points
        return:
        resultant displacement map (3, d, d) in dx, dy, dz
        """
        resultMap = np.zeros((3,psp.d,psp.d))

        local_kx = psp.d//2
        local_ky = psp.d//2

        all_points = np.where(self.sparse_mask == 1)
        contact_points = np.where((contact_mask == 1) & (self.sparse_mask == 1))
        num_points = contact_points[0].shape[0]

        uz = -1*gel_map[(contact_mask == 1) & (self.sparse_mask == 1)]
        min_z = -1*np.min(uz)

        # correct the motion within contact area
        activeMapZ = self.correct_KeyZ(contact_points, uz)
        activeMap = activeMapZ
        if local_deform[0] != 0.0:
            ux = np.zeros((num_points))
            factor = 1
            if local_deform[0] > 0:
                factor = -1
            ux[:] = factor*local_deform[0]/psp.pixmm
            activeMapX = self.correct_KeyX(contact_points, ux, uz)
            activeMapX *= -1*factor
            activeMap += activeMapX

        if local_deform[1] != 0.0:
            uy = np.zeros((num_points))
            factor = 1
            if local_deform[1] > 0:
                factor = -1
            uy[:] = factor*local_deform[1]/psp.pixmm
            activeMapY = self.correct_KeyY(contact_points, uy, uz)
            activeMapY *= -1*factor
            activeMap += activeMapY

        # superposition principle to get all node's motion
        for i in range(all_points[0].shape[0]):
            qx = all_points[1][i]
            qy = all_points[0][i]
            total_deform = np.array([0.0,0.0,0.0])
            # vectorize
            kx_list = contact_points[1]
            ky_list = contact_points[0]

            # get relative position
            cur_x_list = qx-kx_list+local_kx
            cur_y_list = qy-ky_list+local_ky
            mask_x = (cur_x_list >= 0) & (cur_x_list < psp.d)
            mask_y = (cur_y_list >= 0) & (cur_y_list < psp.d)
            mask_valid = mask_x & mask_y

            # retrieve the mutual tensors
            T_list = self.tensorMap[cur_y_list[mask_valid],cur_x_list[mask_valid],:,:]
            # fraction factors
            T_list[:,0:2,2] *= pr.normal_friction
            T_list[:,0:2,0:2] *= pr.shear_friction

            # get the corrected motions for active nodes
            corrected_deform = activeMap[:,ky_list[mask_valid],kx_list[mask_valid]]
            cur_deform = np.matmul(T_list, corrected_deform.T[..., np.newaxis])
            total_deform = np.sum(cur_deform,axis=0)
            resultMap[:,qy,qx] = total_deform.squeeze()
        saveMap = cropMap(resultMap)
        return saveMap
