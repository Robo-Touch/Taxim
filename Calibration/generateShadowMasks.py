import gc
from glob import glob
from os import path as osp
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import argparse

import sys
sys.path.append("..")
from Basics.Geometry import Circle
from Basics.RawData import RawData
import Basics.params as pr

parser = argparse.ArgumentParser()
parser.add_argument("-data_path", nargs='?', default='../data/calib_pin/',
                    help="Path to the folder with raw tactile data.")
args = parser.parse_args()

class ShadowExtraction:
    """ extract the shadow list from the data pack """
    def __init__(self,fn):
        self.raw_data = RawData(osp.join(fn, "dataPack.npz"))
        self.f0 = self.raw_data.f0
        self.bg_proc = self.processInitialFrame()

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

    def extract(self, i, c, max_rad):
        """ extract the shadow patch for a single channel in a single frame """
        frame = self.raw_data.imgs[i,:,:]
        circle = Circle(int(self.raw_data.touch_center[i,0]), int(self.raw_data.touch_center[i,1]),int(self.raw_data.touch_radius[i]))
        dI = frame.astype("float") - self.bg_proc
        dI_c = dI[:,:,c]
        h, w = dI_c.shape
        center = circle.center
        radius = circle.radius
        if (circle.center[1] - max_rad < 0 or circle.center[1] + max_rad > h or circle.center[0] - max_rad < 0 or circle.center[0] + max_rad > w):
            return None

        xcoord, ycoord = np.meshgrid(range(w), range(h))
        xcoord = xcoord - center[0]
        ycoord = ycoord - center[1]
        rsqcoord = xcoord*xcoord + ycoord*ycoord

        rad_sq = radius*radius
        contact_mask = rsqcoord < (rad_sq)

        thresh_mask = dI_c > 0
        inv_mask = contact_mask
        patch = dI_c[circle.center[1]-max_rad:circle.center[1]+max_rad,circle.center[0]-max_rad:circle.center[0]+max_rad]
        p_mask = inv_mask[circle.center[1]-max_rad:circle.center[1]+max_rad,circle.center[0]-max_rad:circle.center[0]+max_rad]
        patch = gaussian_filter(patch, sigma=(3, 3), order=0)
        patch[p_mask == 1] = 0

        return patch

    def extractAll(self):
        """ extract and average the shadow patchs under a certain depth """
        num_img = np.shape(self.raw_data.imgs)[0]
        print("There are total #" + str(num_img) + "frames.")
        shadowMask = np.zeros((pr.max_rad*2, pr.max_rad*2,3))
        for c in range(3):
            num_invalid = 0
            for i in range(num_img):
                patch = self.extract(i, c, pr.max_rad)
                if (patch is None):
                    num_invalid += 1
                    continue
                shadowMask[:,:,c] += patch
            shadowMask[:,:,c] /= (num_img-num_invalid)
        h, w = self.bg_proc.shape[:2]
        cy = h//2
        cx = w//2
        bg_patch = self.bg_proc[cy-pr.max_rad:cy+pr.max_rad,cx-pr.max_rad:cx+pr.max_rad,:]
        recover_patch = shadowMask + bg_patch
        return shadowMask

    def extractList(self):
        """ extract the shadow patchs for a list of different penatration depths """
        num_img = np.shape(self.raw_data.imgs)[0]
        print("There are total #" + str(num_img) + "frames.")
        shadowList = []
        for i in range(num_img):
            shadowMask = np.zeros((pr.max_rad*2, pr.max_rad*2,3))
            for c in range(3):
                patch = self.extract(i, c, pr.max_rad)
                if (patch is None):
                    continue
                shadowMask[:,:,c] = patch
            shadowList.append(shadowMask)
        h, w = self.bg_proc.shape[:2]
        cy = h//2
        cx = w//2
        bg_patch = self.bg_proc[cy-pr.max_rad:cy+pr.max_rad,cx-pr.max_rad:cx+pr.max_rad,:]
        recover_patch = shadowMask + bg_patch
        return shadowList
    def generateShadowTable(self, shadowList):
        """
        Generate shadow table.
        Return values
        1) thetas: (N,1) discritized directions in 2D
        2) c_values: (3, # direcitions, # depths) shadow table
        """
        # load in shadow masks
        print("length of the shadow list: " + str(len(shadowList)))

        # get the center coordinate and the size of the shadow mask
        scx = shadowList[0].shape[1]//2
        scy = shadowList[0].shape[0]//2
        sh = shadowList[0].shape[0]
        sw = shadowList[0].shape[1]

        # discritize the angle in the range (0, 2pi)
        thetas = np.arange(0, 2*np.pi, pr.discritize_precision)
        st = np.sin(thetas)
        ct = np.cos(thetas)
        direction = np.stack((ct,st))

        # build up the shadow table
        c_values = []
        # loop through the r,g,b channel
        for c in range(3):
            d_values = []
            # loop through all discritized directions
            for dd in range(direction.shape[1]):
                direct = direction[:,dd]
                s_values = []
                # loop through the shadow masks with different depths
                for shadowMask in shadowList:
                    shadow_sample = shadowMask[:,:,c]
                    values = []
                    # step from the center towards the boundary
                    for i in range(1, pr.num_step):
                        cur_x = int(scx + pr.shadow_step * i * direct[0])
                        cur_y = int(scy + pr.shadow_step * i * direct[1])
                        # check image boundary
                        if (cur_x < 0 or cur_x >= sw-1 or cur_y < 0 or cur_y >= sh-1):
                            break
                        val = shadow_sample[cur_y,cur_x]
                        # check if the pixel is outside the shadow area
                        if (val != 0.0 and val > pr.shadow_threshold):
                            break
                        # check if the the pixel is within the contact area
                        if (val != 0.0):
                            values.append(shadow_sample[cur_y,cur_x])
                    s_values.append(values)
                d_values.append(s_values)
            c_values.append(d_values)
        c_values = np.array(c_values,dtype=object)
        return thetas, c_values

if __name__ == "__main__":
    shadow = ShadowExtraction(args.data_path)
    shadow_list = shadow.extractList()
    shadow_directions, shadow_table = shadow.generateShadowTable(shadow_list)
    save_path = args.data_path + "shadowTable.npz"
    np.savez(save_path, shadowDirections=shadow_directions, shadowTable = shadow_table)
    print("Saved!")
