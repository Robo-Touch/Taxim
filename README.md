# project_object_folder
this is pyrender+taxim optical simulation for large-scale data collection.

## Setup environment:
`pip install -r requirements.txt`
`pip install -e .`


## Usage: how to generate data
- `example/single_render.py` runs a single case and save the generated contact mask/height map/tactile image. Here are several parameters to adjust:
    - `obj_name`: object to load. You may also want to change the path to the data folder
    - `press_depth`: pressing depth in meter unit
    - `shear_range`: the range of orientation (a cone) along the normal direction
    - `vertex_idx`: vertex index on the object
    - `save_path`: save generated data
    - `live`: online/offline rendering. Online rendering can be used for debugging. Offline for data generation.

- `example/batch_render.py` runs a batch of rendering and save the generated depth map/contact mask/height map/tactile images. Here are several parameters to adjust:
    - `obj_name`: object to load. You may also want to change the path to the data folder
    - `max_depth`: maximum pressing depth in meter unit
    - `min_depth`: minimum pressing depth in meter unit
    - `num_depths`: number of random pressing depths
    - `shear_range`: the range of orientation (a cone) along the normal direction
    - `num_angles`: number of random angles
    - `num_vertices`: number of random vertices
    - `save_path`: save generated data

All the calibration files are under `calibs`.

## License
This project is licensed under MIT license, as found in the [LICENSE](LICENSE) file.
