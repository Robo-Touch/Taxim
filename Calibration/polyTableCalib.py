from os import path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.ndimage import gaussian_filter
from scipy import interpolate
import argparse

import sys
sys.path.append("..")
import Basics.params as pr
import Basics.sensorParams as psp
from Basics.Geometry import Circle

parser = argparse.ArgumentParser()
parser.add_argument("-data_path", nargs='?', default='../data/calib_ball/',
                    help="Path to the folder with data pack.")
args = parser.parse_args()


class PolyTable:
    """ each list contains the N*N tables for one (x,y,v) pair"""
    value_list = []
    locx_list = []
    locy_list = []

class Grads:
  """each grad contains the N*N*params table for one (normal_mag, normal_dir) pair"""
  grad_r = None
  grad_g = None
  grad_b = None
  countmap = None

class polyCalibration:
    """
    Calibrate the polynomial table from the data pack
    """
    def __init__(self,fn):
        self.fn = osp.join(fn, "dataPack.npz")
        data_file = np.load(self.fn,allow_pickle=True)

        self.f0 = data_file['f0']
        self.BallRad = psp.ball_radius
        self.Pixmm = psp.pixmm
        self.imgs = data_file['imgs']
        self.radius_record = data_file['touch_radius']
        self.touchCenter_record = data_file['touch_center']

        self.bg_proc = self.processInitialFrame()
        self.grads = Grads()
        self.poly_table = PolyTable()

        self.img_data_dir = fn

    def processInitialFrame(self):
        # gaussian filtering with square kernel with
        # filterSize : kscale*2+1
        # sigma      : kscale
        kscale = pr.kscale

        img_d = self.f0.astype('float')
        convEachDim = lambda in_img :  gaussian_filter(in_img, kscale)

        f0 = self.f0.copy()
        for ch in range(img_d.shape[2]):
            f0[:,:, ch] = convEachDim(img_d[:,:,ch])

        frame_ = img_d

        # Checking the difference between original and filtered image
        diff_threshold = pr.diffThreshold
        dI = np.mean(f0-frame_, axis=2)
        idx =  np.nonzero(dI<diff_threshold)

        # Mixing image based on the difference between original and filtered image
        frame_mixing_per = pr.frameMixingPercentage
        h,w,ch = f0.shape
        pixcount = h*w

        for ch in range(f0.shape[2]):
            f0[:,:,ch][idx] = frame_mixing_per*f0[:,:,ch][idx] + (1-frame_mixing_per)*frame_[:,:,ch][idx]

        return f0
    def calibrate_all(self):
        num_img = np.shape(self.imgs)[0]
        # loop through all the data points
        for idx in range(num_img):
            print("# iter " + str(idx))
            self.calibrate_single(idx)
        # final interpolation
        grad_r, grad_g, grad_b = self.lookuptable_smooth()
        # save the calibrated file
        out_fn_path = osp.join(self.img_data_dir, "polycalib.npz")
        np.savez(out_fn_path,\
            bins=psp.numBins,
            grad_r = grad_r,
            grad_g = grad_g,
            grad_b = grad_b)
        print("Saved!")

    def calibrate_single(self,idx):
        # keep adding items
        frame = self.imgs[idx,:,:,:]
        # remove background
        dI = frame.astype("float") - self.bg_proc
        circle = Circle(int(self.touchCenter_record[idx,0]), int(self.touchCenter_record[idx,1]),int(self.radius_record[idx]))

        bins = psp.numBins
        ball_radius_pix = psp.ball_radius/psp.pixmm

        center = circle.center
        radius = circle.radius

        sizey, sizex = dI.shape[:2]
        [xqq, yqq] = np.meshgrid(range(sizex), range(sizey))
        xq = xqq - center[0]
        yq = yqq - center[1]

        rsqcoord = xq*xq + yq*yq
        rad_sq = radius*radius
        # get the contact area
        valid_rad = min(rad_sq, int(ball_radius_pix*ball_radius_pix))
        valid_mask = rsqcoord < (valid_rad)

        validId = np.nonzero(valid_mask)
        xvalid = xq[validId]; yvalid = yq[validId]
        rvalid = np.sqrt( xvalid*xvalid + yvalid*yvalid)
        # get gradients
        gradxseq = np.arcsin(rvalid/ball_radius_pix)
        gradyseq = np.arctan2(-yvalid, -xvalid)
        binm = bins - 1

        x_binr = 0.5*np.pi/binm # x [0,pi/2]
        y_binr = 2*np.pi/binm # y [-pi, pi]
        # discritize the gradients
        idx_x = np.floor(gradxseq/x_binr).astype('int')
        idx_y = np.floor((gradyseq+np.pi)/y_binr).astype('int')

        # r channel
        value_map = np.zeros((bins,bins,3))
        loc_x_map = np.zeros((bins,bins))
        loc_y_map = np.zeros((bins,bins))

        valid_r = dI[:,:,0][validId]
        valid_x = xqq[validId]
        valid_y = yqq[validId]
        value_map[idx_x, idx_y, 0] += valid_r

        # g channel
        valid_g = dI[:,:,1][validId]
        value_map[idx_x, idx_y, 1] += valid_g

        # b channel
        valid_b = dI[:,:,2][validId]
        value_map[idx_x, idx_y, 2] += valid_b

        loc_x_map[idx_x, idx_y] += valid_x
        loc_y_map[idx_x, idx_y] += valid_y
        loc_x_map = self.interpolate(loc_x_map)
        loc_y_map = self.interpolate(loc_y_map)

        value_map[:,:,0] = self.interpolate(value_map[:,:,0])
        value_map[:,:,1] = self.interpolate(value_map[:,:,1])
        value_map[:,:,2] = self.interpolate(value_map[:,:,2])

        self.poly_table.value_list.append(value_map)
        self.poly_table.locx_list.append(loc_x_map)
        self.poly_table.locy_list.append(loc_y_map)


    def interpolate(self,img):
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
                                     method='nearest', fill_value = 0) # cubic # nearest
        return GD1

    def lookuptable_smooth(self):
        # final refine
        [h,w,c] = self.bg_proc.shape
        xx,yy = np.meshgrid(np.arange(w),np.arange(h))
        xf = xx.flatten()
        yf = yy.flatten()
        A = np.array([xf*xf,yf*yf,xf*yf,xf,yf,np.ones(h*w)]).T

        table_v = np.array(self.poly_table.value_list)
        table_x = np.array(self.poly_table.locx_list)
        table_y = np.array(self.poly_table.locy_list)

        bins = psp.numBins
        self.grads.grad_r = np.zeros((bins,bins,6))
        self.grads.grad_g = np.zeros((bins,bins,6))
        self.grads.grad_b = np.zeros((bins,bins,6))

        for i in range(table_v.shape[1]):
            for j in range(table_v.shape[2]):
                params_r = self.fitPolyParams(table_x[:,i,j],table_y[:,i,j],table_v[:,i,j,0])
                params_g = self.fitPolyParams(table_x[:,i,j],table_y[:,i,j],table_v[:,i,j,1])
                params_b = self.fitPolyParams(table_x[:,i,j],table_y[:,i,j],table_v[:,i,j,2])
                self.grads.grad_r[i,j,:] = params_r
                self.grads.grad_g[i,j,:] = params_g
                self.grads.grad_b[i,j,:] = params_b

        return self.grads.grad_r, self.grads.grad_g, self.grads.grad_b

    def fitPolyParams(self,xf,yf,b):
        A = np.array([xf*xf,yf*yf,xf*yf,xf,yf,np.ones(xf.shape)]).T
        params, res, rnk, s = lstsq(A, b)
        return params

if __name__ == "__main__":
    polyCalib = polyCalibration(args.data_path)
    polyCalib.calibrate_all()
