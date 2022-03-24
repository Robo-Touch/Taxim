# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import logging
import os
import time

import numpy as np
import pybullet as pb
import pybullet_data

import taxim_robot
import utils
from robot import Robot
from setup import getObjInfo

from collections import defaultdict

import cv2

logger = logging.getLogger(__name__)
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
os.makedirs("logs/data_collect", exist_ok=True)
logging.basicConfig(filename="logs/data_collect/logs_{}.log".format(current_time), level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("-s", '--start', default=0, type=int, help="start of id in log")
parser.add_argument('-dxy', action='store_true', help='whether use dxy')
parser.add_argument("-obj", nargs='?', default='TomatoSoupCan',
                    help="Name of Object to be tested, supported_objects_list = [TomatoSoupCan, RubiksCube, MustardBottle]")
parser.add_argument('-data_path', nargs='?', default='data/grasp3', help='Data Path.')
parser.add_argument('-gui', action='store_true', help='whether use GUI')
args = parser.parse_args()


def convertTime(seconds):
    seconds = round(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


def _align_image(img1, img2):
    img_size = [480, 640]
    new_img = np.zeros([img_size[0], img_size[1] * 2, 3], dtype=np.uint8)
    new_img[:img1.shape[0], :img1.shape[1]] = img2[..., :3]
    new_img[:img2.shape[0], img_size[1]:img_size[1] + img2.shape[1], :] = (img1[..., :3])[..., ::-1]
    return new_img

force_range_list = {
    "TomatoSoupCan": [8],
    "RubiksCube": [10],
    "MustardBottle": [10]
}


height_range_list = {
    "TomatoSoupCan": utils.heightReal2Sim(np.array([74])),
    "MustardBottle": utils.heightReal2Sim(np.array([142])),
    "RubiksCube": utils.heightReal2Sim(np.array([45])),
}

if args.dxy:
    dx_range_list = defaultdict(lambda: np.linspace(-0.005, 0.005, 5).tolist())
else:
    dx_range_list = defaultdict(lambda: [0.00])


if __name__ == "__main__":
    log = utils.Log(args.data_path, args.start)

    save_dir = os.path.join('data', 'seq', args.obj)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize World
    logging.info("Initializing world")
    if args.gui:
        physicsClient = pb.connect(pb.GUI)
    else:
        physicsClient = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    pb.setGravity(0, 0, -9.81)  # Major Tom to planet Earth

    # Initialize digits
    gelsight = taxim_robot.Sensor(width=640, height=480, visualize_gui=args.gui)

    pb.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=15, cameraPitch=-15,
                                  cameraTargetPosition=[0.5, 0, 0.08])

    planeId = pb.loadURDF("plane.urdf")  # Create plane

    robotURDF = "setup/robots/ur5e_wsg50_simplified.urdf"
    robotID = pb.loadURDF(robotURDF, useFixedBase=True)
    rob = Robot(robotID)

    cam = utils.Camera(pb, [640, 480])
    rob.go(rob.pos, wait=True)

    sensorLinks = rob.get_id_by_name(["guide_joint_finger_left"])  # [21, 24]
    gelsight.add_camera(robotID, sensorLinks)

    nbJoint = pb.getNumJoints(robotID)

    # Add object to pybullet and tacto simulator
    urdfObj, obj_mass, obj_height, force_range, deformation, _ = getObjInfo(args.obj)
    finger_height = 0.17
    ori = [0, np.pi / 2, 0]
    heigth_before_grasp = max(0.32, obj_height + 0.26)

    objStartPos = [0.5, 0, obj_height / 2 + 0.01]
    objStartOrientation = pb.getQuaternionFromEuler([0, 0, np.pi / 2])

    objID = pb.loadURDF(urdfObj, objStartPos, objStartOrientation)
    try:
        visual_file = urdfObj.replace("model.urdf", "visual.urdf")
        gelsight.add_object(visual_file, objID, force_range=force_range, deformation=deformation)
    except:
        gelsight.add_object(urdfObj, objID, force_range=force_range, deformation=deformation)
    sensorID = rob.get_id_by_name(["guide_joint_finger_left"])
    if args.gui:
        color, depth = gelsight.render()
        gelsight.updateGUI(color, depth)

    dz = 0.003
    height_list = height_range_list[args.obj]
    gripForce_list = force_range_list[args.obj]
    dx_list = dx_range_list[args.obj]
    print(f"{args.obj} height: {obj_height}")
    print(f"height_list: {height_list}")
    print(f"gripForce_list: {gripForce_list}")
    print(f"dx_list: {dx_list}")

    ## generate config list
    config_list = []
    total_data = 0
    for i, height in enumerate(height_list):
        for j, force in enumerate(gripForce_list):
            for k, dx in enumerate(dx_list):
                config_list.append((height, force, dx))
                total_data += 1

    pb.resetBasePositionAndOrientation(objID, objStartPos, objStartOrientation)
    objPosLast, objOriLast, _ = utils.get_object_pose(pb, objID)
    while True:
        for _ in range(20):
            pb.stepSimulation()
        objPosTrue, objOriTrue, _ = utils.get_object_pose(pb, objID)
        err = np.sum(np.abs(objPosTrue[:2] - objPosLast[:2]))
        if err < 0.001:
            break
        objPosLast, objOriLast = objPosTrue, objOriTrue
    objPos0, objOri0, _ = utils.get_object_pose(pb, objID)
    objStartPos, objStartOrientation = [0.5, 0, objPos0[2]], objOri0

    print("\n")

    start_time = time.time()
    t = num_pos = num_data = 0
    while True:
        # pick_and_place()
        if t == 0:
            rob.reset_robot()
            # print("\nInitializing robot")
            for i in range(10):
                rob.go([0.5, 0, heigth_before_grasp], ori=ori, width=0.13)
                pb.stepSimulation()

            pb.resetBasePositionAndOrientation(objID, objStartPos, objStartOrientation)
            objPosLast, objOriLast, _ = utils.get_object_pose(pb, objID)
            while True:
                for _ in range(20):
                    pb.stepSimulation()
                objPosTrue, objOriTrue, _ = utils.get_object_pose(pb, objID)
                err = np.sum(np.abs(objPosTrue[:2] - objPosLast[:2]))
                if err < 0.001:
                    break
                objPosLast, objOriLast = objPosTrue, objOriTrue

            rot = -1.57
            bias_lth = 0.012
            x_bias = bias_lth * np.cos([rot])[0]
            y_bias = bias_lth * np.sin([rot])[0]
            dx = dy = 0
            config = config_list[num_data]
            height_grasp = config[0]
            gripForce = config[1]
            dx = config[2]

            pos = [objPosTrue[0] + x_bias + dx, objPosTrue[1] + y_bias + dy, height_grasp]
            for i in range(10):
                rob.go([pos[0], pos[1], heigth_before_grasp], ori=ori, width=0.13)
                pb.stepSimulation()
            visualize_data = []
            tactileColor_tmp, _ = gelsight.render()
            visionColor_tmp, _ = cam.get_image()
            visualize_data.append(_align_image(tactileColor_tmp[0], visionColor_tmp))

            ## for seq data
            vision_size, tactile_size = visionColor_tmp.shape, tactileColor_tmp[0].shape
            video_path = os.path.join(save_dir, "demo.mp4")
            rec = utils.video_recorder(vision_size, tactile_size, path=video_path, fps=30)

            cam_poses = []
            gel_poses = []
            obj_poses = []

            obj_world_pose = []
            gel_world_pose = []


        elif t <= 50:
            # Rotating
            rob.operate([pos[0], pos[1], heigth_before_grasp], rot=rot, width=0.13)
        elif t <= 100:
            # Dropping
            rob.go(pos, width=0.13)
            if t == 100:
                tactileColor_tmp, _ = gelsight.render()
                visionColor_tmp, _ = cam.get_image()
                visualize_data.append(_align_image(tactileColor_tmp[0], visionColor_tmp))
        elif t < 150:
            # Grasping
            rob.go(pos, width=0.03, gripForce=gripForce)
        elif t == 150:
            rob.go(pos, width=0.03, gripForce=gripForce)
            # Record sensor states
            data_dict = {}
            normalForce0, lateralForce0 = utils.get_forces(pb, robotID, objID, sensorID[0], -1)
            tactileColor, tactileDepth = gelsight.render()
            data_dict["tactileColorL"], data_dict["tactileDepthL"] = tactileColor[0], tactileDepth[0]
            data_dict["visionColor"], data_dict["visionDepth"] = cam.get_image()
            normalForce = [normalForce0]
            data_dict["normalForce"] = normalForce
            data_dict["height"], data_dict["gripForce"], data_dict["rot"] = utils.heightSim2Real(
                pos[-1]), gripForce, rot
            objPos0, objOri0, _ = utils.get_object_pose(pb, objID)
            visualize_data.append(_align_image(data_dict["tactileColorL"], data_dict["visionColor"]))
            if args.gui:
                gelsight.updateGUI(tactileColor, tactileDepth)
            pos_copy = pos.copy()
        elif t > 150 and t < 210:
            # Lift
            pos_copy[-1] += dz
            rob.go(pos_copy, width=0.03, gripForce=gripForce)
        elif t == 210:
            pos_copy[-1] += dz
            rob.go(pos_copy, width=0.03, gripForce=gripForce)
            tactileColor_tmp, depth = gelsight.render()
            visionColor_tmp, _ = cam.get_image()
            visualize_data.append(_align_image(tactileColor_tmp[0], visionColor_tmp))

            if args.gui:
                gelsight.updateGUI(tactileColor_tmp, depth)
        elif t > 300:
            # Save the data
            tactileColor_tmp, depth = gelsight.render()
            visionColor_tmp, _ = cam.get_image()
            visualize_data.append(_align_image(tactileColor_tmp[0], visionColor_tmp))
            if args.gui:
                gelsight.updateGUI(tactileColor_tmp, depth)
            objPos, objOri, _ = utils.get_object_pose(pb, objID)
            label = 0 if objPos[2] - objPos0[2] < 60 * dz * 0.8 else 1
            data_dict["label"] = label
            data_dict["visual"] = visualize_data

            if log.id < 1:
                gripForce = round(gripForce, 2)
                height_real = round(utils.heightSim2Real(height_grasp))
                dx_real = round(1000 * dx)
                dy_real = round(1000 * dy)

                rec.release()
                cam_poses = np.array(cam_poses)
                gel_poses = np.array(gel_poses)
                obj_poses = np.array(obj_poses)

                gel_world_pose = np.array(gel_world_pose)
                obj_world_pose = np.array(obj_world_pose)

                np.savez(os.path.join(save_dir, "render_poses.npz"), cam_poses=cam_poses, gel_poses=gel_poses, obj_poses=obj_poses)
                np.savez(os.path.join(save_dir, "world_poses.npz"), gel_poses=gel_world_pose, obj_poses=obj_world_pose)

            num_pos += label
            num_data += 1
            log.save(data_dict)
            config_str = "h{:.3f}-rot{:.2f}-f{:.1f}-l{}".format(pos[-1], rot, gripForce, label)
            static_str = "{}:{}:{} is taken in total. {} positive samples".format(
                *convertTime(time.time() - start_time), num_pos)
            print_str = "\rsample {} is collected. {}. {}.".format(
                num_data, config_str, static_str)
            print(print_str, end="")

            if num_data >= total_data:
                break
            # Reset
            t = 0
            continue

        ### seq data
        if t % 3 == 0:
            camera_pose, gel_pose, obj_pose = gelsight.renderer.print_all_pos()
            cam_poses.append(camera_pose)
            gel_poses.append(gel_pose)
            obj_poses.append(obj_pose)

            _, _, obj_pose_in_world = utils.get_object_pose(pb, objID)
            gel_pose_in_world = utils.get_gelsight_pose(pb, robotID)
            obj_world_pose.append(obj_pose_in_world)
            gel_world_pose.append(gel_pose_in_world)

            tactileColor_tmp, depth = gelsight.render()
            visionColor, visionDepth = cam.get_image()
            rec.capture(visionColor.copy(), tactileColor_tmp[0].copy())

        pb.stepSimulation()
        gelsight.update()
        t += 1
        if args.gui:
            time.sleep(0.01)
    print("\nFinished!")
    pb.disconnect()  # Close PyBullet
