import os
from collections import defaultdict

import cv2
import deepdish as dd
import numpy as np
import copy
import math
from PIL import Image
from collections.abc import Iterable

ee_gap = 0.1465


def heightReal2Sim(height):
    """
    convert real height to simulation ee height
    :param height: height in the real world
    :return: height of ee in the simulation
    """
    if isinstance(height, Iterable):
        height = np.array(height)
    height = height / 1000 + ee_gap
    return height


def heightSim2Real(height):
    """
    convert simulation ee height to real height
    :param height: height of ee in the simulation
    :return: height in the real world
    """
    if isinstance(height, Iterable):
        height = np.array(height)
    height = (height - ee_gap) * 1000
    return height


def save_gif(fname, frames, duration=500):
    frames = list(map(Image.fromarray, frames))
    frames[0].save(fname, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)

class video_recorder:
    """
    The class is desgined to capture both side camera images and tactile images as a single video
    """

    def __init__(self, vision_size, tactile_size, path, fps):
        self.img_size = [max(vision_size[0], tactile_size[0]), max(vision_size[1], tactile_size[1])]
        # print(self.img_size, vision_size, tactile_size)
        dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(dir, exist_ok=True)
        if path.endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif path.endswith('.mp4'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            raise ValueError("Unsupported video format")
        self.rec = cv2.VideoWriter(path, fourcc, fps, (self.img_size[1] * 2, self.img_size[0]), True)
        self.path = path

    def capture(self, vision_image, color_image):
        img_cap = self._align_image(vision_image, color_image)
        self.rec.write(img_cap)

    def release(self, new_path=None, delete=False):
        self.rec.release()
        if delete:
            os.remove(self.path)
        if new_path is not None:
            os.rename(self.path, new_path)

    def _align_image(self, img1, img2):
        assert img1.dtype == np.uint8 and img2.dtype == np.uint8
        assert max(img1.shape[0], img2.shape[0]) == self.img_size[0]
        assert max(img1.shape[1], img2.shape[1]) == self.img_size[1]

        new_img = np.zeros([self.img_size[0], self.img_size[1] * 2, 3], dtype=np.uint8)
        new_img[:img1.shape[0], :img1.shape[1]] = (img1[..., :3])[..., ::-1]
        new_img[:img2.shape[0], self.img_size[1]:self.img_size[1] + img2.shape[1]] = img2[..., :3]
        return new_img


class Camera:
    def __init__(self, pb, cameraResolution=[320, 240]):
        self.cameraResolution = cameraResolution

        camTargetPos = [0.581, 0.002, 0.2]
        camDistance = 0.6
        upAxisIndex = 2

        yaw = 50
        pitch = 0
        roll = 0
        fov = 75
        nearPlane = 0.1
        farPlane = 2

        self.viewMatrix = pb.computeViewMatrixFromYawPitchRoll(
            camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex
        )

        aspect = cameraResolution[0] / cameraResolution[1]

        self.projectionMatrix = pb.computeProjectionMatrixFOV(
            fov, aspect, nearPlane, farPlane
        )
        self.pb = pb

    def get_image(self):
        img_arr = self.pb.getCameraImage(
            self.cameraResolution[0],
            self.cameraResolution[1],
            self.viewMatrix,
            self.projectionMatrix,
            shadow=1,
            lightDirection=[1, 1, 1],
            renderer=self.pb.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb = img_arr[2]  # color data RGB
        dep = img_arr[3]  # depth data
        return rgb, dep


def get_forces(pb, bodyA=None, bodyB=None, linkIndexA=None, linkIndexB=None):
    """
    get contact forces

    :return: normal force, lateral force
    """
    kwargs = {
        "bodyA": bodyA,
        "bodyB": bodyB,
        "linkIndexA": linkIndexA,
        "linkIndexB": linkIndexB,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    pts = pb.getContactPoints(**kwargs)

    totalNormalForce = 0
    totalLateralFrictionForce = [0, 0, 0]

    for pt in pts:
        totalNormalForce += pt[9]

        totalLateralFrictionForce[0] += pt[11][0] * pt[10] + pt[13][0] * pt[12]
        totalLateralFrictionForce[1] += pt[11][1] * pt[10] + pt[13][1] * pt[12]
        totalLateralFrictionForce[2] += pt[11][2] * pt[10] + pt[13][2] * pt[12]
        # print(
        #     f"lateralFriction1 {pt[10]}, lateralFrictionDir1{pt[11]},\nlateralFriction2 {pt[12]}, lateralFrictionDir2{pt[13]}")

    return totalNormalForce, totalLateralFrictionForce


class Log:
    def __init__(self, dirName, start=0):
        self.dirName = dirName
        self.id = 0
        self.start = start
        self.dataList = []
        self.batch_size = 100
        os.makedirs(dirName, exist_ok=True)

    def save(self, data):
        self.dataList.append(data.copy())

        if len(self.dataList) >= self.batch_size:
            id_str = "{:07d}".format(self.id + self.start)
            # os.makedirs(outputDir, exist_ok=True)
            outputDir = os.path.join(self.dirName, id_str)
            os.makedirs(outputDir, exist_ok=True)

            # print(newData["tactileColorL"][0].shape)
            newData = {k: [] for k in data.keys()}
            for d in self.dataList:
                for k in data.keys():
                    newData[k].append(d[k])

            for k in data.keys():
                fn_k = "{}_{}.h5".format(id_str, k)
                outputFn = os.path.join(outputDir, fn_k)
                dd.io.save(outputFn, newData[k])

            self.dataList = []
            self.id += 1


def get_object_pose(pb, objID):
    res = pb.getBasePositionAndOrientation(objID)

    world_positions = res[0]
    world_orientations = res[1]

    world_positions = np.array(world_positions)
    world_orientations = np.array(world_orientations)

    obj_pose_in_world = pose2mat((world_positions, world_orientations))
    # return obj_pose_in_world

    return world_positions, world_orientations, obj_pose_in_world

def print_all_links(pb, robotID):
    _link_name_to_index = {pb.getBodyInfo(robotID)[0].decode('UTF-8'):-1,}

    for _id in range(pb.getNumJoints(robotID)):
        _name = pb.getJointInfo(robotID, _id)[12].decode('UTF-8')
        _link_name_to_index[_name] = _id
        print(_name, _id)

def quat2mat(quaternion):
    """
    Converts given quaternion (x, y, z, w) to matrix.

    Args:
        quaternion: vec4 float angles

    Returns:
        3x3 rotation matrix
    """
    q = np.array(quaternion, dtype=np.float32, copy=True)[[3, 0, 1, 2]]
    n = np.dot(q, q)
    EPS = np.finfo(float).eps * 4.
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
        ]
    )

def pose2mat(pose):
    """
    Converts pose to homogeneous matrix.

    Args:
        pose: a (pos, orn) tuple where pos is vec3 float cartesian, and
            orn is vec4 float quaternion.

    Returns:
        4x4 homogeneous matrix
    """
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = quat2mat(pose[1])
    homo_pose_mat[:3, 3] = np.array(pose[0], dtype=np.float32)
    homo_pose_mat[3, 3] = 1.
    return homo_pose_mat

def get_gelsight_pose(pb, robotID):
    eef_pos_in_world = np.array(pb.getLinkState(robotID, 10)[0])
    eef_orn_in_world = np.array(pb.getLinkState(robotID, 10)[1])
    eef_pose_in_world = pose2mat((eef_pos_in_world, eef_orn_in_world))
    return eef_pose_in_world
