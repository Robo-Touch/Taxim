# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Set backend platform for OpenGL render (pyrender.OffscreenRenderer)
- Pyglet, the same engine that runs the pyrender viewer. This requires an active
  display manager, so you can’t run it on a headless server. This is the default option.
- OSMesa, a software renderer. require extra install OSMesa.
  (https://pyrender.readthedocs.io/en/latest/install/index.html#installing-osmesa)
- EGL, which allows for GPU-accelerated rendering without a display manager.
  Requires NVIDIA’s drivers.

The handle for EGL is egl (preferred, require NVIDIA driver),
The handle for OSMesa is osmesa.
Default is pyglet, which requires active window
"""

# import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"

import logging
import os

import cv2
import numpy as np
import pybullet as p
import pyrender
import trimesh
from omegaconf import OmegaConf
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from .gelsight_render import gelsightRender

logger = logging.getLogger(__name__)


def euler2matrix(angles=(0, 0, 0), translation=(0, 0, 0)):
    q = p.getQuaternionFromEuler(angles)
    r = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
    pose = np.eye(4)
    pose[:3, 3] = translation
    pose[:3, :3] = r
    return pose

class Renderer:
    def __init__(self, width, height, background, config_path):
        """

        :param width: scalar
        :param height: scalar
        :param background: image
        :param config_path:
        """
        self._width = width
        self._height = height

        self.gelsight_render = gelsightRender()

        self.use_gelsight = ("gelsight" in config_path)

        if background is not None:
            self.set_background(background)
        else:
            self._background_real = None

        logger.info("Loading configuration from: %s" % config_path)
        self.conf = OmegaConf.load(config_path)

        self.force_enabled = (
                self.conf.sensor.force is not None and self.conf.sensor.force.enable
        )

        if self.force_enabled:
            if len(self.conf.sensor.force.range_force) == 2:
                self.get_offset = interp1d(self.conf.sensor.force.range_force,
                                           [0, self.conf.sensor.force.max_deformation],
                                           bounds_error=False,
                                           fill_value=(0, self.conf.sensor.force.max_deformation))
            else:
                self.get_offset = interp1d(self.conf.sensor.force.range_force,
                                           self.conf.sensor.force.max_deformation,
                                           bounds_error=False,
                                           fill_value=(0, self.conf.sensor.force.max_deformation[-1]))

        self._init_pyrender()

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def background(self):
        return self._background_real

    def _init_pyrender(self):
        """
        Initialize pyrender
        """
        # Create scene for pybullet sync
        self.scene = pyrender.Scene()

        # Create scene for rendering given depth image
        self.scene_depth = pyrender.Scene()

        """
        objects format:
            {obj_name: pyrender node}
        """
        self.object_nodes = {}
        self.object_depth_nodes = {}
        self.current_object_nodes = {}
        self.object_trimeshes = {}

        self.current_light_nodes = []
        self.cam_light_ids = []

        self._init_gel()
        self._init_camera()
        self._init_light()

        self.r = pyrender.OffscreenRenderer(self.width, self.height)
        self.r_depth = pyrender.OffscreenRenderer(self.width, self.height)
        colors, depths = self.render(object_poses=None, noise=False, calibration=False)
        # self.show_scene()
        self.depth0 = depths
        self._background_sim = colors

    def show_scene(self):
        scene_visual = pyrender.Scene()

        # add object nodes
        for objname, objnode in self.current_object_nodes.items():
            objTrimesh = self.object_trimeshes[objname]
            pose = objnode.matrix
            mesh = pyrender.Mesh.from_trimesh(objTrimesh)
            obj_node_new = pyrender.Node(mesh=mesh, matrix=pose)
            scene_visual.add_node(obj_node_new)

        # add gel node
        mesh_gel = pyrender.Mesh.from_trimesh(self.gel_trimesh, smooth=False)
        gel_pose = self.gel_node.matrix
        gel_node_new = pyrender.Node(mesh=mesh_gel, matrix=gel_pose)
        scene_visual.add_node(gel_node_new)

        # add light
        for i, light_node in enumerate(self.light_nodes):
            color = self.light_colors[i]
            intensity = self.light_intensities[i]
            light_new = pyrender.PointLight(color=color, intensity=intensity)
            light_pose = light_node.matrix
            light_node_new = pyrender.Node(light=light_new, matrix=light_pose)
            scene_visual.add_node(light_node_new)

        # add camera
        for i, camera_node in enumerate(self.camera_nodes):
            cami = self.conf_cam[i]
            camera = pyrender.camera.IntrinsicsCamera(fx=cami.fx, fy=cami.fy, cx=cami.cx, cy=cami.cy, znear=cami.znear)
            pose = camera_node.matrix
            camera_node = pyrender.Node(camera=camera, matrix=pose)
            scene_visual.add_node(camera_node)

        pyrender.Viewer(scene_visual)

    def show_scene_depth(self):
        print("call show_scene_depth")
        self._print_all_pos_depth()
        scene_visual = pyrender.Scene()
        for objname, objnode in self.object_depth_nodes.items():
            objTrimesh = self.object_trimeshes[objname]
            pose = objnode.matrix
            mesh = pyrender.Mesh.from_trimesh(objTrimesh)
            obj_node_new = pyrender.Node(mesh=mesh, matrix=pose)
            scene_visual.add_node(obj_node_new)
        # add light
        for i, light_node in enumerate(self.light_nodes):
            color = self.light_colors[i]
            intensity = self.light_intensities[i]
            light_new = pyrender.PointLight(color=color, intensity=intensity)
            light_pose = light_node.matrix
            light_node_new = pyrender.Node(light=light_new, matrix=light_pose)
            scene_visual.add_node(light_node_new)

        # add camera
        for i, camera_node in enumerate(self.camera_nodes):
            cami = self.conf_cam[i]
            camera = pyrender.camera.IntrinsicsCamera(fx=cami.fx, fy=cami.fy, cx=cami.cx, cy=cami.cy, znear=cami.znear)
            pose = camera_node.matrix
            camera_node = pyrender.Node(camera=camera, matrix=pose)
            scene_visual.add_node(camera_node)

        pyrender.Viewer(scene_visual)

    def _init_gel(self):
        """
        Add gel surface in the scene
        """
        # Create gel surface (flat/curve surface based on config file)
        self.gel_trimesh = self._generate_gel_trimesh()
        mesh_gel = pyrender.Mesh.from_trimesh(self.gel_trimesh, smooth=False)
        self.gel_pose0 = np.eye(4)
        self.gel_node = pyrender.Node(mesh=mesh_gel, matrix=self.gel_pose0)
        self.scene.add_node(self.gel_node)

    def _generate_gel_trimesh(self):

        # Load config
        g = self.conf.sensor.gel

        if hasattr(g, "mesh") and g.mesh is not None:
            mesh_dir = os.path.dirname(os.path.realpath(__file__))
            mesh_path = os.path.join(mesh_dir, g.mesh)
            gel_trimesh = trimesh.load(mesh_path)

        elif not g.curvature:
            # Flat gel surface
            origin = g.origin

            X0, Y0, Z0 = origin[0], origin[1], origin[2]
            W, H = g.width, g.height
            gel_trimesh = trimesh.Trimesh(
                vertices=[
                    [X0, Y0 + W / 2, Z0 + H / 2],
                    [X0, Y0 + W / 2, Z0 - H / 2],
                    [X0, Y0 - W / 2, Z0 - H / 2],
                    [X0, Y0 - W / 2, Z0 + H / 2],
                ],
                faces=[[0, 1, 2], [2, 3, 0]],
            )
        else:
            origin = g.origin
            X0, Y0, Z0 = origin[0], origin[1], origin[2]
            W, H = g.width, g.height
            # Curved gel surface
            N = g.countW
            M = int(N * H / W)
            R = g.R
            zrange = g.curvatureMax

            y = np.linspace(Y0 - W / 2, Y0 + W / 2, N)
            z = np.linspace(Z0 - H / 2, Z0 + H / 2, M)
            yy, zz = np.meshgrid(y, z)

            h = R - np.maximum(0, R ** 2 - (yy - Y0) ** 2 - (zz - Z0) ** 2) ** 0.5
            xx = X0 - zrange * h / h.max()

            gel_trimesh = self._generate_trimesh_from_depth(xx)
        # from datetime import datetime
        # stl_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gelsight_data", "mesh_tmp_{}.stl".format(datetime.now().strftime("%Y%m%d%H%M%S")))
        # gel_trimesh.export(stl_filename)
        print("gel mesh bounds={}".format(gel_trimesh.bounds))
        return gel_trimesh

    def _generate_trimesh_from_depth(self, depth):
        # Load config
        g = self.conf.sensor.gel
        origin = g.origin

        _, Y0, Z0 = origin[0], origin[1], origin[2]
        W, H = g.width, g.height

        N = depth.shape[1]
        M = depth.shape[0]

        # Create grid mesh
        y = np.linspace(Y0 - W / 2, Y0 + W / 2, N)
        z = np.linspace(Z0 - H / 2, Z0 + H / 2, M)
        yy, zz = np.meshgrid(y, z)

        # Vertex format: [x, y, z]
        vertices = np.zeros([N * M, 3])

        # Add x, y, z position to vertex
        vertices[:, 0] = depth.reshape([-1])
        vertices[:, 1] = yy.reshape([-1])
        vertices[:, 2] = zz.reshape([-1])

        # Create faces

        faces = np.zeros([(N - 1) * (M - 1) * 6], dtype=np.uint)

        # calculate id for each vertex: (i, j) => i * m + j
        xid = np.arange(N)
        yid = np.arange(M)
        yyid, xxid = np.meshgrid(xid, yid)
        ids = yyid[:-1, :-1].reshape([-1]) + xxid[:-1, :-1].reshape([-1]) * N

        # create upper triangle
        faces[::6] = ids  # (i, j)
        faces[1::6] = ids + N  # (i+1, j)
        faces[2::6] = ids + 1  # (i, j+1)

        # create lower triangle
        faces[3::6] = ids + 1  # (i, j+1)
        faces[4::6] = ids + N  # (i+1, j)
        faces[5::6] = ids + N + 1  # (i+1, j+1)

        faces = faces.reshape([-1, 3])
        gel_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        return gel_trimesh

    def _init_camera(self):
        """
        Set up camera
        """

        self.camera_nodes = []
        self.camera_zero_poses = []
        self.camera_depth_nodes = []

        self.conf_cam = self.conf.sensor.camera
        self.nb_cam = len(self.conf_cam)

        for i in range(self.nb_cam):
            cami = self.conf_cam[i]
            camera = pyrender.camera.IntrinsicsCamera(fx=cami.fx, fy=cami.fy, cx=cami.cx, cy=cami.cy, znear=cami.znear)
            camera_zero_pose = euler2matrix(
                angles=np.deg2rad(cami.orientation), translation=cami.position,
            )
            self.camera_zero_poses.append(camera_zero_pose)

            # Add camera node into scene
            camera_node = pyrender.Node(camera=camera, matrix=camera_zero_pose)
            self.scene.add_node(camera_node)
            self.camera_nodes.append(camera_node)

            # Add extra camera node into scene_depth
            camera_node_depth = pyrender.Node(camera=camera, matrix=camera_zero_pose)
            self.scene_depth.add_node(camera_node_depth)
            self.camera_depth_nodes.append(camera_node_depth)

            # Add corresponding light for rendering the camera
            self.cam_light_ids.append(list(cami.lightIDList))

    def _init_light(self):
        """
        Set up light
        """

        # Load light from config file
        light = self.conf.sensor.lights

        origin = np.array(light.origin)

        xyz = []
        if light.polar:
            # Apply polar coordinates
            thetas = light.xrtheta.thetas
            rs = light.xrtheta.rs
            xs = light.xrtheta.xs
            for i in range(len(thetas)):
                theta = np.pi / 180 * thetas[i]
                xyz.append([xs[i], rs[i] * np.cos(theta), rs[i] * np.sin(theta)])
        else:
            # Apply cartesian coordinates
            xyz = np.array(light.xyz.coords)

        self.light_colors = np.array(light.colors)
        self.light_intensities = light.intensities

        # Save light nodes
        self.light_nodes = []
        self.light_poses0 = []
        self.light_depth_nodes = []

        for i in range(len(self.light_colors)):
            color = self.light_colors[i]
            intensity = self.light_intensities[i]
            light_pose_0 = euler2matrix(angles=[0, 0, 0], translation=xyz[i] + origin)

            light = pyrender.PointLight(color=color, intensity=intensity)
            light_node = pyrender.Node(light=light, matrix=light_pose_0)

            self.scene.add_node(light_node)
            self.light_nodes.append(light_node)
            self.light_poses0.append(light_pose_0)
            self.current_light_nodes.append(light_node)

            # Add extra light node into scene_depth
            light_node_depth = pyrender.Node(light=light, matrix=light_pose_0)
            self.scene_depth.add_node(light_node_depth)
            self.light_depth_nodes.append(light_node_depth)

    def add_object(self, objTrimesh, obj_name, position=(0, 0, 0), orientation=(0, 0, 0), force_range=None,
                   deformation=None):
        """
        Add object into the scene
        """

        mesh = pyrender.Mesh.from_trimesh(objTrimesh)
        pose = euler2matrix(angles=orientation, translation=position)

        obj_node = pyrender.Node(mesh=mesh, matrix=pose)
        self.scene.add_node(obj_node)
        self.object_nodes[obj_name] = obj_node
        self.current_object_nodes[obj_name] = obj_node
        self.object_trimeshes[obj_name] = objTrimesh

        # add object in depth scene
        obj_depth_node = pyrender.Node(mesh=mesh, matrix=pose)
        self.scene_depth.add_node(obj_depth_node)
        self.object_depth_nodes[obj_name] = obj_depth_node

        if force_range is not None and deformation is not None:
            self.get_offset = interp1d(force_range, deformation, bounds_error=False, fill_value=(0, deformation[-1]))

    def update_camera_pose(self, position, orientation):
        """
        Update sensor pose (including camera, lighting, and gel surface)
        ### important ###
        call self.update_camera_pose before self.render
        """

        pose = euler2matrix(angles=orientation, translation=position)

        # Update camera
        for i in range(self.nb_cam):
            camera_pose = pose.dot(self.camera_zero_poses[i])
            self.camera_nodes[i].matrix = camera_pose
            # update depth camera
            self.camera_depth_nodes[i].matrix = camera_pose

        # Update gel
        gel_pose = pose.dot(self.gel_pose0)
        self.gel_node.matrix = gel_pose

        # Update light
        for i in range(len(self.light_nodes)):
            light_pose = pose.dot(self.light_poses0[i])
            self.light_nodes[i].matrix = light_pose
            # update depth light
            self.light_depth_nodes[i].matrix = light_pose

    def update_object_pose(self, obj_name, position, orientation):
        """
        orientation: euler angles
        """
        node = self.object_nodes[obj_name]
        pose = euler2matrix(angles=orientation, translation=position)
        self.scene.set_pose(node, pose=pose)

    def update_light(self, lightIDList):
        """
        Update the light node based on lightIDList, remove the previous light
        """
        # Remove previous light nodes
        for node in self.current_light_nodes:
            self.scene.remove_node(node)

        # Add light nodes
        self.current_light_nodes = []
        for i in lightIDList:
            light_node = self.light_nodes[i]
            self.scene.add_node(light_node)
            self.current_light_nodes.append(light_node)

    def _add_noise(self, color):
        """
        Add Gaussian noise to the RGB image
        :param color:
        :return:
        """
        # Add noise to the RGB image
        mean = self.conf.sensor.noise.color.mean
        std = self.conf.sensor.noise.color.std

        if mean != 0 or std != 0:
            noise = np.random.normal(mean, std, color.shape)  # Gaussian noise
            color = np.clip(color + noise, 0, 255).astype(
                np.uint8
            )  # Add noise and clip

        return color

    def _calibrate(self, color, camera_index):
        """
        Calibrate simulation wrt real sensor by adding background
        :param color:
        :return:
        """

        if self._background_real is not None:
            # Simulated difference image, with scaling factor 0.5
            diff = (color.astype(np.float) - self._background_sim[camera_index]) * 0.5

            # Add low-pass filter to match real readings
            diff = cv2.GaussianBlur(diff, (7, 7), 0)

            # Combine the simulated difference image with real background image
            color = np.clip((diff[:, :, :3] + self._background_real), 0, 255).astype(
                np.uint8
            )

        return color

    def set_background(self, background):
        self._background_real = cv2.resize(background, (self._width, self._height))
        self._background_real = self._background_real[:, :, ::-1]
        return 0

    def get_bias(self, camera_pos, camera_ori, normal_forces, object_poses):
        for obj_name in normal_forces:
            if obj_name not in object_poses:
                continue
            obj_pos, objOri = object_poses[obj_name]
            node = self.object_depth_nodes[obj_name]
            pose = euler2matrix(angles=objOri, translation=obj_pos)
            self.scene_depth.set_pose(node, pose=pose)

        # self.show_scene_depth()
        color, depth = self.r.render(self.scene_depth)
        depth[depth < 1e-8] = 1
        depth_gap = depth - self.depth0[0]
        bias = np.min(depth_gap)
        if np.abs(bias) > 0.01:
            bias = 0
        return bias

    def get_volume(self):
        # self.show_scene_depth()
        _, depth = self.r.render(self.scene)
        depth_gap = np.clip(self.depth0[0] - depth, 0, 1)
        volume = np.sum(depth_gap)
        return volume

    def adjust_with_force(
            self, camera_pos, camera_ori, normal_forces, object_poses,
    ):
        """
        Adjust object pose with normal force feedback
        The larger the normal force, the larger indentation
        Currently linear adjustment from force to shift distance
        It can be replaced by non-linear adjustment with calibration from real sensor
        """
        existing_obj_names = list(self.current_object_nodes.keys())
        for obj_name in existing_obj_names:
            # Remove object from scene if not in contact
            if obj_name not in normal_forces:
                self.scene.remove_node(self.current_object_nodes[obj_name])
                self.current_object_nodes.pop(obj_name)

        # Add/Update the objects' poses the scene if in contact

        def get_pos(camera_pos, camera_ori, obj_pos, offset):
            camera_pos = np.array(camera_pos)
            obj_pos = np.array(obj_pos)

            direction_vector = camera_ori.apply(np.array([0, 0, 1]))
            direction = camera_pos - obj_pos
            direction = direction / (np.sum(direction ** 2) ** 0.5 + 1e-8)
            # print("direction_veactor:", direction_vector, 'direction:',  direction)
            obj_pos_new = obj_pos + offset * direction_vector
            return obj_pos_new

        for obj_name in normal_forces:
            if obj_name not in object_poses:
                continue
            obj_pos, objOri = object_poses[obj_name]

            # Add the object node to the scene
            if obj_name not in self.current_object_nodes:
                node = self.object_nodes[obj_name]
                self.scene.add_node(node)
                self.current_object_nodes[obj_name] = node

            if self.force_enabled:
                offset = -1.0
                if obj_name in normal_forces:
                    offset = self.get_offset([normal_forces[obj_name]])[0]

                    gap_bias = self.get_bias(camera_pos, camera_ori, normal_forces, object_poses)
                    offset += gap_bias
                    obj_pos_tmp = get_pos(camera_pos, camera_ori, obj_pos, offset)
                    self.update_object_pose(obj_name, obj_pos_tmp, objOri)
                    pixmm = 0.0295

                    cur_volume = self.get_volume() / pixmm * 1000
                    # binary_search for the best pressing_depth
                    slope = 200039
                    est_volume = normal_forces[obj_name] * slope
                    start_point = offset
                    maximum_depth_mm = 8.2 + gap_bias * 1000 # 3.2
                    minimum_depth_mm = gap_bias * 1000
                    cur_depth_pix = offset * 1000 / pixmm  # starting point
                    max_depth_pix = maximum_depth_mm / pixmm
                    min_depth_pix = minimum_depth_mm / pixmm
                    max_threshold = (0.003 + gap_bias) * 1000 / pixmm

                    err_threshold = 1000
                    recurring_times = 0
                    while np.abs(cur_volume - est_volume) > err_threshold:
                        recurring_times += 1
                        if recurring_times > 20:
                            # exceed the maximum iteration
                            cur_depth_pix = offset * 1000 / pixmm
                            break
                        if cur_volume < est_volume:
                            # increase the pressing depth
                            min_depth_pix = cur_depth_pix
                            cur_depth_pix = cur_depth_pix + (max_depth_pix - cur_depth_pix) / 2.0
                        elif cur_volume > est_volume:
                            # decrease the pressing depth
                            max_depth_pix = cur_depth_pix
                            cur_depth_pix = cur_depth_pix - (cur_depth_pix - min_depth_pix) / 2.0
                        else:
                            break

                        new_offset = cur_depth_pix * pixmm / 1000
                        obj_pos_tmp = get_pos(camera_pos, camera_ori, obj_pos, new_offset)
                        self.update_object_pose(obj_name, obj_pos_tmp, objOri)
                        cur_volume = self.get_volume() / pixmm * 1000

                    offset = min(cur_depth_pix * pixmm / 1000, 0.0082 + gap_bias)
                    self.vol = cur_volume
                # Calculate pose changes based on normal force
                obj_pos = get_pos(camera_pos, camera_ori, obj_pos, offset)

            self.update_object_pose(obj_name, obj_pos, objOri)

    def _post_process(self, color, depth, camera_index, noise=True, calibration=True):
        if calibration:
            color = self._calibrate(color, camera_index)
        if noise:
            color = self._add_noise(color)
        return color, depth

    def render(self, object_poses=None, normal_forces=None, noise=False, calibration=True,
               shear_forces=None, camera_pos_old=None, camera_ori_old=None):
        """
        :param object_poses:
        :param normal_forces:
        :param noise:
        :return:
        """
        # print("Begin Rendering")
        if camera_pos_old is not None and camera_ori_old is not None:
            self.update_camera_pose(camera_pos_old, camera_ori_old)

        colors, depths = [], []

        for i in range(self.nb_cam):
            # Set the main camera node for rendering
            self.scene.main_camera_node = self.camera_nodes[i]

            # Set up corresponding lights (max: 8)
            self.update_light(self.cam_light_ids[i])

            # Adjust contact based on force
            if object_poses is not None and normal_forces is not None:
                # Get camera pose for adjusting object pose

                camera_pose = self.camera_nodes[i].matrix
                camera_pos = camera_pose[:3, 3].T
                camera_ori = R.from_matrix(camera_pose[:3, :3])

                # self.show_scene()
                self.adjust_with_force(
                    camera_pos, camera_ori, normal_forces, object_poses,
                )

                # add shear discplacement
                # shear_slope = 20
                # camera_offset = - np.array(shear_forces['2_-1']) * shear_slope / 1000 * 0.0295
                # print("shear displacement", np.linalg.norm(camera_offset), camera_offset)

            color, depth = self.r.render(self.scene)
            color, depth = self._post_process(color, depth, i, noise, calibration)

            # render color from gelsight
            if self.use_gelsight:
                color_gel = self.gelsight_render.render(depth.copy())
                color = np.clip(color_gel, 0, 255, out=color_gel).astype(np.uint8)

            colors.append(color)
            depths.append(depth)
        return colors, depths

    def print_all_pos(self):
        camera_pose = self.camera_nodes[0].matrix
        camera_pos = camera_pose[:3, 3].T
        camera_ori = R.from_matrix(camera_pose[:3, :3]).as_quat()
        # print("camera pos and ori in pyrender=", (camera_pos, camera_ori))

        gel_pose = self.gel_node.matrix
        gel_pos = gel_pose[:3, 3].T
        gel_ori = R.from_matrix(gel_pose[:3, :3]).as_quat()
        # print("Gel pos and ori in pyrender=", (gel_pos, gel_ori))

        obj_pose = self.object_nodes["2_-1"].matrix
        obj_pos = obj_pose[:3, 3].T
        obj_ori = R.from_matrix(obj_pose[:3, :3]).as_quat()
        return camera_pose, gel_pose, obj_pose

    def _print_all_pos_depth(self):
        camera_pose = self.camera_depth_nodes[0].matrix
        camera_pos = camera_pose[:3, 3].T
        camera_ori = R.from_matrix(camera_pose[:3, :3]).as_quat()
        print("depth camera pos and ori in pyrender=", (camera_pos, camera_ori))

        obj_pose = self.object_depth_nodes["2_-1"].matrix
        obj_pos = obj_pose[:3, 3].T
        obj_ori = R.from_matrix(obj_pose[:3, :3]).as_quat()
        print("depth obj pos and ori in pyrender=", (obj_pos, obj_ori))
