import numpy as np
import scipy
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R

def gen_t_quat(pose):
    # convert 4 x 4 transformation matrix to [x, y, z, qx, qy, qz, qw]
    r = R.from_matrix(pose[0:3,0:3])
    q = r.as_quat() # qx, qy, qz, qw
    t = pose[0:3,3].T
    return np.concatenate((t, q), axis=0)

def skewMat(v):
    # vector to its skew matrix
    mat = np.zeros((3,3))
    mat[0,1] = -1*v[2]
    mat[0,2] = v[1]

    mat[1,0] = v[2]
    mat[1,2] = -1*v[0]

    mat[2,0] = -1*v[1]
    mat[2,1] = v[0]
    return mat

def normalize(v):
    # normalize the vector
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def padding(img):
    # pad one row & one col on each side
    if len(img.shape) == 2:
        return np.pad(img, ((1, 1), (1, 1)), 'edge')
    elif len(img.shape) == 3:
        return np.pad(img, ((1, 1), (1, 1), (0, 0)), 'edge')

def gen_pose(vertex, normal, z_axis):
    # given the position & orientation
    # output the 4 x 4 transformation matrix

    # z_axis = np.array([0,0,1])
    T_wp = np.zeros((4,4)) # transform from point coord to world coord
    T_wp[3,3] = 1
    T_wp[0:3,3] = vertex # t

    # if (z_axis==normal).all():
    #     T_wp[0:3,0:3] = np.identity(3)
    # else:
    ### make x pointing up
    Rot = np.zeros((3,3))
    Rot[:,2] = normal
    Rot[:,1] = np.cross(z_axis, normal)
    Rot[:,0] = np.cross(Rot[:,1], normal)
    # # generate rotation for sensor
    # v = np.cross(z_axis,normal)
    # s = np.linalg.norm(v)
    # c = np.dot(z_axis,normal)
    # Rot = np.identity(3) + skewMat(v) + np.linalg.matrix_power(skewMat(v),2) * (1-c)/(s**2) # rodrigues
    T_wp[0:3,0:3] = Rot
    det = np.linalg.det(Rot)
    if det == 0:
        print("Singularity!!!")
        print(T_wp)
        T_wp[0:3,0:3] = np.identity(3)
    # try:
    #     T_pw = np.linalg.inv(T_wp) # transform from world coord to point coord
    # except:
    #     print("Singularity!!!")
    #     print(T_wp)
    #     T_wp[0:3,0:3] = np.identity(3)
    return T_wp

def generate_normals(height_map):
    # from height map to gradient magnitude & directions

    [h,w] = height_map.shape
    center = height_map[1:h-1,1:w-1] # z(x,y)
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

    grad_mag = padding(grad_mag)
    grad_dir = padding(grad_dir)
    return grad_mag, grad_dir

def processInitialFrame(f_init):
    # gaussian filtering with square kernel with
    # filterSize : kscale*2+1
    # sigma      : kscale
    kscale = 50

    img_d = f_init.astype('float')
    convEachDim = lambda in_img :  gaussian_filter(in_img, kscale)

    f0 = f_init.copy()
    for ch in range(img_d.shape[2]):
        f0[:,:, ch] = convEachDim(img_d[:,:,ch])

    frame_ = img_d

    # Checking the difference between original and filtered image
    diff_threshold = 5
    dI = np.mean(f0-frame_, axis=2)
    idx =  np.nonzero(dI<diff_threshold)

    # Mixing image based on the difference between original and filtered image
    frame_mixing_per = 0.15
    h,w,ch = f0.shape
    pixcount = h*w

    for ch in range(f0.shape[2]):
        f0[:,:,ch][idx] = frame_mixing_per*f0[:,:,ch][idx] + (1-frame_mixing_per)*frame_[:,:,ch][idx]

    return f0
