
# Directed Distance Function

This repository contains all the code needed to train a network to learn the directed distance to an object surface. This includes code for data generation, model creation, training, and visualization.

# Requirements

- trimesh
- tqdm
- scikit-learn
- matplotlib
- pytorch
- open3d
- pickle (part of standard Python)
- palettable
- rtree
- igl

More requirements for Srinath's v2:

- beacon (`pip install git+https://github.com/brown-ivl/beacon.git`)
- tk3dv (`pip install git+https://github.com/drsrinathsridhar/tk3dv.git`)
- [ Required on some machines to avoid an OpenMP issue ] `conda install -c conda-forge nomkl`


## Data Generation / Sampling Rays

The data generation code offers a few methods for sampling rays with the idea that some sampling techniques produce harder to learn rays (edge cases). To visualize the different sampling methods, use

`python sampling.py -v --mesh_file <path to .obj --use_4d>`

You can also see how fast each sampling method is by running

`python sampling.py -s`

## Training

To train, test, and save a network, run

`python train4D.py -Tts -n mynetwork --mesh_file <path to .obj> --save_dir <dir results can be written in> --intersect_limit <# of intersections>`

The flags `-d`, `-v`, `-p`, and `-m` can be used to create depth images, depth video, point clouds, and meshes respectively. The video will be saved to `<save_dir>/depth_videos` while the rest of the visualizations will be displayed on screen.

To see all flags use

`python train4D.py --help`

## Codebase Overview

* Training/Testing - `train4D.py`
* Data Generation - `data.py`, `sampling.py`
* Network - `model.py`
* Utility Functions - `utils.py`, `rasterization.py`
* Visualization - `camera.py`, `visualization.py`
