import glob
import os
import numpy as np
import xml.etree.ElementTree as ET
from stl import mesh

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
objBaseDir = os.path.join(BASE_DIR, "objects")
obj_list = []
for root, dirs, files in os.walk(objBaseDir, topdown=False):
    for name in dirs:
        obj_list.append(name)

def _get_height(obj_path):
    lcut = hcut = None
    with open(obj_path) as file:
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                z_cor = float(strs[3])
                if lcut is None or z_cor < lcut:
                    lcut = z_cor
                if hcut is None or z_cor > hcut:
                    hcut = z_cor
            if strs[0] == "vt":
                break
    return hcut - lcut


def getObjInfo(objName):
    if objName == "Cube":
        urdf_path = os.path.join(objBaseDir, objName, "cube_small.urdf")
        force_range, deformation = FT_data_dict["RubiksCube"]
        return urdf_path, 1., 0.05, force_range, deformation
    assert objName in obj_list
    urdf_path = os.path.join(objBaseDir, objName, "model.urdf")
    tree = ET.parse(urdf_path)
    mass_node = next(tree.iter('mass'))
    if mass_node is None:
        raise KeyError("No mass in the urdf file.")
    mass = float(mass_node.attrib["value"])

    friction_node = next(tree.iter('lateral_friction'))
    if friction_node is None:
        raise KeyError("No friction in the urdf file.")
    friction = float(friction_node.attrib["value"])

    obj_mesh_list = glob.glob(os.path.join(objBaseDir, objName, "*.obj"))
    if len(obj_mesh_list) > 0:
        obj_path = obj_mesh_list[0]
        height = _get_height(obj_path)
    else:
        mesh_file_name = os.path.join(objBaseDir, objName, "*.stl")
        mesh_file_list = glob.glob(mesh_file_name)
        stl_path = mesh_file_list[0]
        stl_file = mesh.Mesh.from_file(stl_path)
        height = np.max(stl_file.z) - np.min(stl_file.z)

    force_range = np.array(
        [0.013903, 0.08583400000000001, 0.18635599999999997, 0.301228, 0.44313, 0.6062639999999999,
         0.7980979999999996,
         1.006655, 1.255222, 1.498395, 1.791708, 2.10153, 2.4639089999999997, 2.873739, 3.3301070000000004,
         3.8420690000000004, 4.392766999999999, 4.958345, 5.535276, 6.171562, 6.803239000000002,
         7.445841000000001,
         8.154558, 8.841395, 9.530221, 10.181313, 10.924032999999998, 11.770725, 12.293285000000001,
         13.091981999999998])
    deformation = np.arange(0.0001, 0.0031, 0.0001)

    return urdf_path, mass, height, force_range, deformation, friction


if __name__ == "__main__":
    heights = []
    for obe in obj_list:
        heights.append(getObjInfo(obe)[2])
    print(obj_list)
    print(max(heights))
