import matplotlib.pyplot as plt
import numpy as np
import os
from os import path as osp
import argparse
import sys
sys.path.append("..")
import Basics.sensorParams as psp
from compose.dataLoader import dataLoader
from compose.superposition import SuperPosition, fill_blank, cropMap

parser = argparse.ArgumentParser()
parser.add_argument("-obj", nargs='?', default='square',
                    help="Name of Object to be tested, supported_objects_list = [square, cylinder6]")
parser.add_argument('-dx', default = 0.0, type=float, help='Shear load on X axis.')
parser.add_argument('-dy', default = 0.0, type=float, help='Shear load on Y axis.')
parser.add_argument('-dz', default = 1.0, type=float, help='Shear load on Z axis.')
args = parser.parse_args()

def getDomeHeightMap(filePath, obj, press_depth, domeMap):
    """
    get the height map & contact mask from the object and gelpad model
    obj: object's point cloud
    press_depth: in millimeter
    domeMap: gelpad model
    return:
    zq: height map with contact
    contact_mask
    """
    # read in the object's model
    objPath = osp.join(filePath,obj)
    f = open(objPath)
    lines = f.readlines()
    verts_num = int(lines[3].split(' ')[-1])
    verts_lines = lines[10:10 + verts_num]
    vertices = np.array([list(map(float, l.strip().split(' '))) for l in verts_lines])
    heightMap = np.zeros((psp.d,psp.d))

    cx = np.mean(vertices[:,0])
    cy = np.mean(vertices[:,1])
    uu = ((vertices[:,0] - cx)/psp.pixmm + psp.d//2).astype(int)
    vv = ((vertices[:,1] - cy)/psp.pixmm + psp.d//2).astype(int)

    mask_u = np.logical_and(uu > 0, uu < psp.d)
    mask_v = np.logical_and(vv > 0, vv < psp.d)
    mask_z = vertices[:,2] > 0.2
    mask_map = mask_u & mask_v & mask_z
    heightMap[vv[mask_map],uu[mask_map]] = vertices[mask_map][:,2]/psp.pixmm

    max_o = np.max(heightMap)
    heightMap -= max_o
    pressing_height_pix = press_depth/psp.pixmm

    gel_map = heightMap+pressing_height_pix

    contact_mask = (gel_map > domeMap)
    zq = np.zeros((psp.d,psp.d))
    zq[contact_mask] = gel_map[contact_mask] - domeMap[contact_mask]
    return zq, contact_mask


if __name__ == "__main__":
    # calibration file
    data_folder = osp.join("..", "calibs", "femCalib.npz")
    super = SuperPosition(data_folder)

    # compose
    filePath = osp.join('..', 'data', 'objects')
    obj = args.obj+'.ply'
    local_deform = np.array([args.dx, args.dy, args.dz])
    press_depth = local_deform[2]

    domeMap = np.load(osp.join('..', 'calibs', 'dome_gel.npy'))
    gel_map, contact_mask = getDomeHeightMap(filePath, obj, press_depth, domeMap)
    resultMap = super.compose_sparse(local_deform, gel_map, contact_mask)

    #### for visualization/saving the results ###
    compose_savePath = osp.join('..', 'results', args.obj+'_compose.jpg')

    plt.figure(1)
    plt.subplot(311)
    fig = plt.imshow(fill_blank(resultMap[0,:,:]), cmap='RdBu')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.subplot(312)
    fig = plt.imshow(fill_blank(resultMap[1,:,:]), cmap='RdBu')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.subplot(313)
    fig = plt.imshow(fill_blank(resultMap[2,:,:]), cmap='RdBu')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    # plt.show()
    plt.savefig(compose_savePath)
