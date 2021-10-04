from os import path as osp
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import correlate
import scipy.ndimage as ndimage
from scipy import interpolate
import cv2
import argparse

import sys
sys.path.append("..")
from Basics.RawData import RawData
from Basics.CalibData import CalibData
import Basics.params as pr
import Basics.sensorParams as psp

parser = argparse.ArgumentParser()
parser.add_argument("-obj", nargs='?', default='square',
                    help="Name of Object to be tested, supported_objects_list = [square, cylinder6]")
parser.add_argument('-depth', default = 1.0, type=float, help='Indetation depth into the gelpad.')
args = parser.parse_args()

class simulator(object):
    def __init__(self, data_folder, filePath, obj):
        """
        Initialize the simulator.
        1) load the object,
        2) load the calibration files,
        3) generate shadow table from shadow masks
        """
        # read in object's ply file
        # object facing positive direction of z axis
        objPath = osp.join(filePath,obj)
        self.obj_name = obj.split('.')[0]
        print("load object: " + self.obj_name)
        f = open(objPath)
        lines = f.readlines()
        self.verts_num = int(lines[3].split(' ')[-1])
        verts_lines = lines[10:10 + self.verts_num]
        self.vertices = np.array([list(map(float, l.strip().split(' '))) for l in verts_lines])

        # polytable
        calib_data = osp.join(data_folder, "polycalib.npz")
        self.calib_data = CalibData(calib_data)

        # raw calibration data
        rawData = osp.join(data_folder, "dataPack.npz")
        data_file = np.load(rawData,allow_pickle=True)
        self.f0 = data_file['f0']
        self.bg_proc = self.processInitialFrame()

        #shadow calibration
        self.shadow_depth = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
        shadowData = np.load(osp.join(data_folder, "shadowTable.npz"),allow_pickle=True)
        self.direction = shadowData['shadowDirections']
        self.shadowTable = shadowData['shadowTable']

    def processInitialFrame(self):
        """
        Smooth the initial frame
        """
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

    def simulating(self, heightMap, contact_mask, contact_height, shadow=False):
        """
        Simulate the tactile image from the height map
        heightMap: heightMap of the contact
        contact_mask: indicate the contact area
        contact_height: the height of each pix
        shadow: whether add the shadow

        return:
        sim_img: simulated tactile image w/o shadow
        shadow_sim_img: simluated tactile image w/ shadow
        """

        # generate gradients of the height map
        grad_mag, grad_dir = self.generate_normals(heightMap)

        # generate raw simulated image without background
        sim_img_r = np.zeros((psp.h,psp.w,3))
        bins = psp.numBins

        [xx, yy] = np.meshgrid(range(psp.w), range(psp.h))
        xf = xx.flatten()
        yf = yy.flatten()
        A = np.array([xf*xf,yf*yf,xf*yf,xf,yf,np.ones(psp.h*psp.w)]).T
        binm = bins - 1

        # discritize grids
        x_binr = 0.5*np.pi/binm # x [0,pi/2]
        y_binr = 2*np.pi/binm # y [-pi, pi]

        idx_x = np.floor(grad_mag/x_binr).astype('int')
        idx_y = np.floor((grad_dir+np.pi)/y_binr).astype('int')

        # look up polynomial table and assign intensity
        params_r = self.calib_data.grad_r[idx_x,idx_y,:]
        params_r = params_r.reshape((psp.h*psp.w), params_r.shape[2])
        params_g = self.calib_data.grad_g[idx_x,idx_y,:]
        params_g = params_g.reshape((psp.h*psp.w), params_g.shape[2])
        params_b = self.calib_data.grad_b[idx_x,idx_y,:]
        params_b = params_b.reshape((psp.h*psp.w), params_b.shape[2])

        est_r = np.sum(A * params_r,axis = 1)
        est_g = np.sum(A * params_g,axis = 1)
        est_b = np.sum(A * params_b,axis = 1)

        sim_img_r[:,:,0] = est_r.reshape((psp.h,psp.w))
        sim_img_r[:,:,1] = est_g.reshape((psp.h,psp.w))
        sim_img_r[:,:,2] = est_b.reshape((psp.h,psp.w))

        # attach background to simulated image
        sim_img = sim_img_r + self.bg_proc

        if not shadow:
            return sim_img, sim_img

        # add shadow
        cx = psp.w//2
        cy = psp.h//2

        # find shadow attachment area
        kernel = np.ones((5, 5), np.uint8)
        dialate_mask = cv2.dilate(np.float32(contact_mask),kernel,iterations = 2)
        enlarged_mask = dialate_mask == 1
        boundary_contact_mask = 1*enlarged_mask - 1*contact_mask
        contact_mask = boundary_contact_mask == 1

        # (x,y) coordinates of all pixels to attach shadow
        x_coord = xx[contact_mask]
        y_coord = yy[contact_mask]

        # get normal index to shadow table
        normMap = grad_dir[contact_mask] + np.pi
        norm_idx = normMap // pr.discritize_precision
        # get height index to shadow table
        contact_map = contact_height[contact_mask]
        height_idx = (contact_map * psp.pixmm - self.shadow_depth[0]) // pr.height_precision
        height_idx_max = int(np.max(height_idx))
        total_height_idx = self.shadowTable.shape[2]

        shadowSim = np.zeros((psp.h,psp.w,3))

        # all 3 channels
        for c in range(3):
            frame = sim_img_r[:,:,c].copy()
            frame_back = sim_img_r[:,:,c].copy()
            for i in range(len(x_coord)):
                # get the coordinates (x,y) of a certain pixel
                cy_origin = y_coord[i]
                cx_origin = x_coord[i]
                # get the normal of the pixel
                n = int(norm_idx[i])
                # get height of the pixel
                h = int(height_idx[i]) + 6
                if h < 0 or h >= total_height_idx:
                    continue
                # get the shadow list for the pixel
                v = self.shadowTable[c,n,h]

                # number of steps
                num_step = len(v)

                # get the shadow direction
                theta = self.direction[n]
                d_theta = theta
                ct = np.cos(d_theta)
                st = np.sin(d_theta)
                # use a fan of angles around the direction
                theta_list = np.arange(d_theta-pr.fan_angle, d_theta+pr.fan_angle, pr.fan_precision)
                ct_list = np.cos(theta_list)
                st_list = np.sin(theta_list)
                for theta_idx in range(len(theta_list)):
                    ct = ct_list[theta_idx]
                    st = st_list[theta_idx]

                    for s in range(1,num_step):
                        cur_x = int(cx_origin + pr.shadow_step * s * ct)
                        cur_y = int(cy_origin + pr.shadow_step * s * st)
                        # check boundary of the image and height's difference
                        if cur_x >= 0 and cur_x < psp.w and cur_y >= 0 and cur_y < psp.h and heightMap[cy_origin,cx_origin] > heightMap[cur_y,cur_x]:
                            frame[cur_y,cur_x] = np.minimum(frame[cur_y,cur_x],v[s])

            shadowSim[:,:,c] = frame
            shadowSim[:,:,c] = ndimage.gaussian_filter(shadowSim[:,:,c], sigma=(pr.sigma, pr.sigma), order=0)

        shadow_sim_img = shadowSim+ self.bg_proc
        shadow_sim_img = cv2.GaussianBlur(shadow_sim_img.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
        return sim_img, shadow_sim_img

    def generateHeightMap(self, gelpad_model_path, pressing_height_mm, dx, dy):
        """
        Generate the height map by interacting the object with the gelpad model.
        pressing_height_mm: pressing depth in millimeter
        dx, dy: shift of the object
        return:
        zq: the interacted height map
        gel_map: gelpad height map
        contact_mask: indicate contact area
        """
        assert(self.vertices.shape[1] == 3)
        # load dome-shape gelpad model
        gel_map = np.load(gelpad_model_path)
        gel_map = cv2.GaussianBlur(gel_map.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
        heightMap = np.zeros((psp.h,psp.w))

        # centralize the points
        cx = np.mean(self.vertices[:,0])
        cy = np.mean(self.vertices[:,1])
        # add the shifting and change to the pix coordinate
        uu = ((self.vertices[:,0] - cx)/psp.pixmm + psp.w//2+dx).astype(int)
        vv = ((self.vertices[:,1] - cy)/psp.pixmm + psp.h//2+dy).astype(int)
        # check boundary of the image
        mask_u = np.logical_and(uu > 0, uu < psp.w)
        mask_v = np.logical_and(vv > 0, vv < psp.h)
        # check the depth
        mask_z = self.vertices[:,2] > 0.2
        mask_map = mask_u & mask_v & mask_z
        heightMap[vv[mask_map],uu[mask_map]] = self.vertices[mask_map][:,2]/psp.pixmm

        max_g = np.max(gel_map)
        min_g = np.min(gel_map)
        max_o = np.max(heightMap)
        # pressing depth in pixel
        pressing_height_pix = pressing_height_mm/psp.pixmm

        # shift the gelpad to interact with the object
        gel_map = -1 * gel_map + (max_g+max_o-pressing_height_pix)

        # get the contact area
        contact_mask = heightMap > gel_map

        # combine contact area of object shape with non contact area of gelpad shape
        zq = np.zeros((psp.h,psp.w))

        zq[contact_mask]  = heightMap[contact_mask]
        zq[~contact_mask] = gel_map[~contact_mask]
        return zq, gel_map, contact_mask

    def deformApprox(self, pressing_height_mm, height_map, gel_map, contact_mask):
        zq = height_map.copy()
        zq_back = zq.copy()
        pressing_height_pix = pressing_height_mm/psp.pixmm
        # contact mask which is a little smaller than the real contact mask
        mask = (zq-(gel_map)) > pressing_height_pix * pr.contact_scale
        mask = mask & contact_mask

        # approximate soft body deformation with pyramid gaussian_filter
        for i in range(len(pr.pyramid_kernel_size)):
            zq = cv2.GaussianBlur(zq.astype(np.float32),(pr.pyramid_kernel_size[i],pr.pyramid_kernel_size[i]),0)
            zq[mask] = zq_back[mask]
        zq = cv2.GaussianBlur(zq.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)

        contact_height = zq - gel_map

        return zq, mask, contact_height

    def interpolate(self,img):
        """
        fill the zero value holes with interpolation
        """
        x = np.arange(0, img.shape[1])
        y = np.arange(0, img.shape[0])
        # mask invalid values
        array = np.ma.masked_where(img == 0, img)
        xx, yy = np.meshgrid(x, y)
        # get the valid values
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = img[~array.mask]

        GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                  (xx, yy),
                                     method='linear', fill_value = 0) # cubic # nearest # linear
        return GD1

    def generate_normals(self,height_map):
        """
        get the gradient (magnitude & direction) map from the height map
        """
        [h,w] = height_map.shape
        top = height_map[0:h-2,1:w-1] # z(x-1,y)
        bot = height_map[2:h,1:w-1] # z(x+1,y)
        left = height_map[1:h-1,0:w-2] # z(x,y-1)
        right = height_map[1:h-1,2:w] # z(x,y+1)
        dzdx = (bot-top)/2.0
        dzdy = (right-left)/2.0

        mag_tan = np.sqrt(dzdx**2 + dzdy**2)
        grad_mag = np.arctan(mag_tan)
        invalid_mask = mag_tan == 0
        valid_mask = ~invalid_mask
        grad_dir = np.zeros((h-2,w-2))
        grad_dir[valid_mask] = np.arctan2(dzdx[valid_mask]/mag_tan[valid_mask], dzdy[valid_mask]/mag_tan[valid_mask])

        grad_mag = self.padding(grad_mag)
        grad_dir = self.padding(grad_dir)
        return grad_mag, grad_dir

    def padding(self,img):
        """ pad one row & one col on each side """
        return np.pad(img, ((1, 1), (1, 1)), 'symmetric')

if __name__ == "__main__":
    data_folder = osp.join(osp.join( "..", "calibs"))
    filePath = osp.join('..', 'data', 'objects')
    gelpad_model_path = osp.join( '..', 'calibs', 'gelmap5.npy')
    obj = args.obj + '.ply'
    sim = simulator(data_folder, filePath, obj)
    press_depth = args.depth
    dx = 0
    dy = 0

    # generate height map
    height_map, gel_map, contact_mask = sim.generateHeightMap(gelpad_model_path, press_depth, dx, dy)
    # approximate the soft deformation
    heightMap, contact_mask, contact_height = sim.deformApprox(press_depth, height_map, gel_map, contact_mask)
    # simulate tactile images
    sim_img, shadow_sim_img = sim.simulating(heightMap, contact_mask, contact_height, shadow=True)
    img_savePath = osp.join('..', 'results', obj[:-4]+'_sim.jpg')
    shadow_savePath = osp.join('..', 'results', obj[:-4]+'_shadow.jpg')
    height_savePath = osp.join('..', 'results', obj[:-4]+'_height.npy')
    cv2.imwrite(img_savePath, sim_img)
    cv2.imwrite(shadow_savePath, shadow_sim_img)
    np.save(height_savePath, heightMap)
