import matplotlib.pyplot as plt
import numpy as np
import os
from os import path as osp
import cv2
from scipy import interpolate

import sys
sys.path.append("../..")
import Basics.sensorParams as psp

def fill_blank(img):
    """
    fill the zero value holes with interpolation
    """
    if np.max(img) == np.min(img):
        return img
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

class dataLoader:

    def __init__(self, x_path, y_path, z_path):
        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.dx_list = []
        self.dy_list = []
        self.dz_list = []
        with open(z_path) as f:
            lines = f.readlines()
        num_lines = len(lines)
        for i in range(1,num_lines):
            ll = lines[i].split()
            # meter -> millimeter
            self.x_list.append(float(ll[1])*1000.0)
            self.y_list.append(float(ll[2])*1000.0)
            self.z_list.append(float(ll[3])*1000.0)
            self.dz_list.append(float(ll[4])*1000.0)

        with open(x_path) as f:
            lines = f.readlines()
        num_lines = len(lines)
        for i in range(1,num_lines):
            ll = lines[i].split()
            self.dx_list.append(float(ll[4])*1000.0)

        with open(y_path) as f:
            lines = f.readlines()
        num_lines = len(lines)
        for i in range(1,num_lines):
            ll = lines[i].split()
            self.dy_list.append(float(ll[4])*1000.0)
        self.num_points = num_lines-1
        # centralize the points
        self.x_list = self.x_list - np.mean(self.x_list)
        self.y_list = self.y_list - np.mean(self.y_list)
        self.z_list = self.z_list - np.min(self.z_list)

    def generateDeformMap(self, dx=0, dy=0):
        """
        from the raw txt data to the deformatiom map
        return:
        deformMap (4, d, d): height map, dx, dy, dz
        mask: sparse nodes
        """
        # origin height, dx, dy, dz
        deformMap = np.zeros((4, psp.d, psp.d)) # here 640 * 640, easy to rotate
        mask = np.zeros((psp.d, psp.d))
        y_shift = psp.d//2
        x_shift = psp.d//2

        for i in range(self.num_points):
            x_pix = int(self.x_list[i]/psp.pixmm)
            y_pix = int(self.y_list[i]/psp.pixmm)

            # XY coordinate in pixel
            x_local = int(-1*x_pix + x_shift +dx/psp.pixmm)
            y_local = int(-1*y_pix + y_shift +dy/psp.pixmm)
            # check boundary
            if x_local < 0 or x_local >= psp.d:
                continue
            if y_local < 0 or y_local >= psp.d:
                continue
            z_pix = self.z_list[i]/psp.pixmm
            deformMap[0, y_local,x_local] = z_pix

            dx_pix = self.dx_list[i]/psp.pixmm
            dy_pix = self.dy_list[i]/psp.pixmm
            dz_pix = self.dz_list[i]/psp.pixmm
            deformMap[1, y_local,x_local] = -1*dx_pix
            deformMap[2, y_local,x_local] = -1*dy_pix
            deformMap[3, y_local,x_local] = dz_pix
            mask[y_local,x_local] = 1
        return deformMap, mask

    def correctSym_dz(self, deformMap):
        # this is used to force symmetric
        z = fill_blank(deformMap[0,:,:])
        dx = fill_blank(deformMap[1,:,:])
        dy = fill_blank(deformMap[2,:,:])
        dz = fill_blank(deformMap[3,:,:])
        idx_x = np.arange(psp.d//2)
        idx_vx = psp.d-1 - idx_x
        idx_y = np.arange(psp.d//2)
        idx_vy = psp.d-1 - idx_y
        error_x = dx[:,idx_x] + dx[:,idx_vx]
        error_y = dy[idx_y,:] + dy[idx_vy,:]
        # correct the error
        dx[:,idx_vx] -= error_x/2.0
        dx[:,idx_x] -= error_x/2.0
        dy[idx_vy,:] -= error_y/2.0
        dy[idx_y,:] -= error_y/2.0
        #
        error_y = dy[:,idx_x] - dy[:,idx_vx]
        error_x = dx[idx_y,:] - dx[idx_vy,:]

        #
        dy[:,idx_x] -= error_y/2.0
        dy[:,idx_vx] += error_y/2.0
        dx[idx_y,:] -= error_x/2.0
        dx[idx_vy,:] += error_x/2.0

        # z
        error_zy = dz[:,idx_x] - dz[:,idx_vx]
        dz[:,idx_x] -= error_zy/2.0
        dz[:,idx_vx] += error_zy/2.0
        error_zx = dz[idx_y,:] - dz[idx_vy,:]
        dz[idx_y,:] -= error_zx/2.0
        dz[idx_vy,:] += error_zx/2.0

        filledMap = np.zeros((4, psp.d, psp.d))
        filledMap[0,:,:] = z
        filledMap[1,:,:] = dx
        filledMap[2,:,:] = dy
        filledMap[3,:,:] = dz

        return filledMap

    def correctSym_dxdz(self, deformMap):
        # input a [dx, 0, dz]
        # output 4 symmetric deformMap
        # [dx, 0, dz], [0, dy, dz], [-dx, 0, dz], [0, -dy, dz]
        z = fill_blank(deformMap[0,:,:])
        dx = fill_blank(deformMap[1,:,:])
        dy = fill_blank(deformMap[2,:,:])
        dz = fill_blank(deformMap[3,:,:])
        idx_x = np.arange(psp.d//2)
        idx_vx = psp.d-1 - idx_x
        idx_y = np.arange(psp.d//2)
        idx_vy = psp.d-1 - idx_y
        error_x = dx[:,idx_x] + dx[:,idx_vx]
        error_y = dy[idx_y,:] + dy[idx_vy,:]

        # correct the error
        dy[idx_vy,:] -= error_y/2.0
        dy[idx_y,:] -= error_y/2.0

        #
        error_y = dy[:,idx_x] - dy[:,idx_vx]
        error_x = dx[idx_y,:] - dx[idx_vy,:]

        #
        dx[idx_y,:] -= error_x/2.0
        dx[idx_vy,:] += error_x/2.0

        # z
        error_zx = dz[idx_y,:] - dz[idx_vy,:]
        dz[idx_y,:] -= error_zx/2.0
        dz[idx_vy,:] += error_zx/2.0

        filledMap = np.zeros((4, psp.d, psp.d))
        filledMap[0,:,:] = z
        filledMap[1,:,:] = dx
        filledMap[2,:,:] = dy
        filledMap[3,:,:] = dz

        deform_list = []
        deform_list.append(filledMap)

        fullMap = np.zeros((4, psp.d, psp.d))
        fullMap[0,:,:] = z
        fullMap[1,:,:] = dx
        fullMap[2,:,:] = dy
        fullMap[3,:,:] = dz

        # rotate 3 times
        filledMap = np.zeros((4, psp.d, psp.d))
        filledMap[0,:,:] = np.rot90(fullMap[0,:,:])
        filledMap[2,:,:] = -1*np.rot90(fullMap[1,:,:])
        filledMap[1,:,:] = np.rot90(fullMap[2,:,:])
        filledMap[3,:,:] = np.rot90(fullMap[3,:,:])
        deform_list.append(filledMap)

        filledMap = np.zeros((4, psp.d, psp.d))
        filledMap[0,:,:] = np.rot90(fullMap[0,:,:],2)
        filledMap[1,:,:] = -1*np.rot90(fullMap[1,:,:],2)
        filledMap[2,:,:] = -1*np.rot90(fullMap[2,:,:],2)
        filledMap[3,:,:] = np.rot90(fullMap[3,:,:],2)
        deform_list.append(filledMap)

        filledMap = np.zeros((4, psp.d, psp.d))
        filledMap[0,:,:] = np.rot90(fullMap[0,:,:],3)
        filledMap[2,:,:] = np.rot90(fullMap[1,:,:],3)
        filledMap[1,:,:] = -1*np.rot90(fullMap[2,:,:],3)
        filledMap[3,:,:] = np.rot90(fullMap[3,:,:],3)
        deform_list.append(filledMap)

        return deform_list
