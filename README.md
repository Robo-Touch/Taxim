# Taxim-Robot: An extension of Taxim simulation model 

[![License: MIT](https://img.shields.io/github/license/facebookresearch/tacto)](LICENSE)

This package is an integrated robot simulation framework with [Taxim](https://github.com/CMURoboTouch/Taxim). We show the application of this framework with grasping demos. It is adapted from Facebook's [tacto](https://github.com/facebookresearch/tacto) repo.

![demo_screwdriver](/media/demo_screwdriver.gif)

![demo_tomatosoupcan](/media/demo_tomatosoupcan.gif)


## Installation

You can manually clone the repository and install the package using:

```bash
git clone -b taxim-robot https://github.com/CMURoboTouch/Taxim
cd Taxim
pip install -e .
```

To install dependencies:

```bash
pip install -r requirements.txt
```

## Content

This package contain several components:

1) A renderer to simulate tactile readings from a GelSight.
2) A contact model maps normal forces to indentation depths and shear forces to shear displacements.
2) Mesh models and urdf files for GelSight, UR5e robot, WSG50 gripper and objects.
3) Examples of grasping.

## Usage

### experiments

```bash
python grasp_data_demo.py -obj TomatoSoupCan
```

```bash
python grasp_air_demo.py -obj 044_flat_screwdriver
```

`grasp_data_demo.py` is used to demonstrate grasping on objects that can stand on the table. `grasp_air_demo.py` is used to demonstrate grasping on elongated objects which will be initialized in the air.

## Operating System

We recommend to conduct experiments on **Ubuntu**.

For **macOS**, there exists some visualization problem between pybullet.GUI and pyrender as we know of. Please let us
know if it can be resolved, and we will share the information at the repo!

## License
This project is licensed under MIT license, as found in the [LICENSE](LICENSE) file.


