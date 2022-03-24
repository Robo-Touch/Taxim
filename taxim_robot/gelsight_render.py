import cv2
import os
import numpy as np

gelsight_path = os.path.dirname(os.path.realpath(__file__))
conf = {
    'background_path': os.path.join(gelsight_path, "gelsight_data/bg.jpg"),
    'calib_path': os.path.join(gelsight_path, "gelsight_data/polycalib.npz"),
    'real_bg': os.path.join(gelsight_path, "gelsight_data/real_bg.npy"),
    'h': 480,
    'w': 640,
    'numBins': 120,
}

h = conf['h']
w = conf['w']


def padding(img):
    # pad one row & one col on each side
    if len(img.shape) == 2:
        return np.pad(img, ((1, 1), (1, 1)), 'symmetric')
    elif len(img.shape) == 3:
        return np.pad(img, ((1, 1), (1, 1), (0, 0)), 'symmetric')


def generate_normals(height_map):
    [h, w] = height_map.shape
    center = height_map[1:h - 1, 1:w - 1]  # z(x,y)
    top = height_map[0:h - 2, 1:w - 1]  # z(x-1,y)
    bot = height_map[2:h, 1:w - 1]  # z(x+1,y)
    left = height_map[1:h - 1, 0:w - 2]  # z(x,y-1)
    right = height_map[1:h - 1, 2:w]  # z(x,y+1)
    dzdx = (bot - top) / 2.0
    dzdy = (right - left) / 2.0
    direction = -np.ones((h - 2, w - 2, 3))
    direction[:, :, 0] = dzdx
    direction[:, :, 1] = dzdy

    mag_tan = np.sqrt(dzdx ** 2 + dzdy ** 2)
    grad_mag = np.arctan(mag_tan)
    invalid_mask = mag_tan == 0
    valid_mask = ~invalid_mask
    grad_dir = np.zeros((h - 2, w - 2))
    grad_dir[valid_mask] = np.arctan2(dzdx[valid_mask] / mag_tan[valid_mask], dzdy[valid_mask] / mag_tan[valid_mask])

    magnitude = np.sqrt(direction[:, :, 0] ** 2 + direction[:, :, 1] ** 2 + direction[:, :, 2] ** 2)
    normal = direction / magnitude[:, :, np.newaxis]  # unit norm

    normal = padding(normal)
    grad_mag = padding(grad_mag)
    grad_dir = padding(grad_dir)
    return grad_mag, grad_dir, normal


class CalibData:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        data = np.load(dataPath)

        self.numBins = data['bins']
        self.grad_r = data['grad_r']
        self.grad_g = data['grad_g']
        self.grad_b = data['grad_b']


class gelsightRender:

    def __init__(self):
        self.background = cv2.imread(conf['background_path'])
        self.calib_data = CalibData(conf['calib_path'])
        self.real_bg = np.load(conf['real_bg'])
        print("Gelsight initialization done!")
        bins = conf['numBins']
        [xx, yy] = np.meshgrid(range(w), range(h))
        xf = xx.flatten()
        yf = yy.flatten()
        self.A = np.array([xf * xf, yf * yf, xf * yf, xf, yf, np.ones(h * w)]).T

        binm = bins - 1
        self.x_binr = 0.5 * np.pi / binm  # x [0,pi/2]
        self.y_binr = 2 * np.pi / binm  # y [-pi, pi]
        self.bg_depth = None

    def smooth_heightMap(self, height_map):
        if self.bg_depth is None:
            print("reset depth bg map")
            self.bg_depth = height_map.copy()
            return height_map
        diff_depth = np.abs(height_map - self.bg_depth)
        contact_mask = diff_depth > (np.max(diff_depth) * 0.4)
        zq_back = height_map.copy()

        # kernel_size = [51, 21, 11, 5]
        kernel_size = [101, 51, 21, 11, 5]
        # kernel_size = [201, 101, 51, 21, 11, 5]
        for i in range(len(kernel_size)):
            height_map = cv2.GaussianBlur(height_map.astype(np.float32), (kernel_size[i], kernel_size[i]), 0)
            if i < 6:
                height_map[contact_mask] = zq_back[contact_mask]
        height_map = cv2.GaussianBlur(height_map.astype(np.float32), (5, 5), 0)
        return height_map

    def render(self, heightMap):
        # print("gelsight render")
        heightMap *= -1000.0
        heightMap /= 0.0295
        heightMap = self.smooth_heightMap(heightMap)
        grad_mag, grad_dir, _ = generate_normals(heightMap)
        sim_img_r = np.zeros((h, w, 3))
        idx_x = np.floor(grad_mag / self.x_binr).astype('int')
        idx_y = np.floor((grad_dir + np.pi) / self.y_binr).astype('int')

        params_r = self.calib_data.grad_r[idx_x, idx_y, :]
        params_r = params_r.reshape((h * w), params_r.shape[2])
        params_g = self.calib_data.grad_g[idx_x, idx_y, :]
        params_g = params_g.reshape((h * w), params_g.shape[2])
        params_b = self.calib_data.grad_b[idx_x, idx_y, :]
        params_b = params_b.reshape((h * w), params_b.shape[2])

        est_r = np.sum(self.A * params_r, axis=1)
        est_g = np.sum(self.A * params_g, axis=1)
        est_b = np.sum(self.A * params_b, axis=1)
        sim_img_r[:, :, 0] = est_r.reshape((h, w))
        sim_img_r[:, :, 1] = est_g.reshape((h, w))
        sim_img_r[:, :, 2] = est_b.reshape((h, w))

        # write tactile image
        sim_img = (sim_img_r + self.real_bg)  # /255.0
        return sim_img
