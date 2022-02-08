from turtle import position
import torch
from beacon import utils as butils
import argparse
import math

from PyQt5.QtWidgets import QApplication
import numpy as np
from tk3dv.pyEasel import *
from Easel import Easel
from tqdm import tqdm
import multiprocessing as mp
from skimage.measure import marching_cubes



FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from odf_dataset import ODFDatasetLiveVisualizer, ODFDatasetVisualizer
# from pc_sampler import PC_SAMPLER_RADIUS
from depth_sampler_5d import DEPTH_SAMPLER_RADIUS
from single_losses import SingleDepthBCELoss, SINGLE_MASK_THRESH
from single_models import ODFSingleV3
# from pc_odf_dataset import PCODFDatasetLoader as PCDL
from depth_odf_dataset_5d import DepthODFDatasetLoader as DDL
from odf_dataset import ODFDatasetLoader as ODL
import odf_v2_utils as o2utils

def generate_marching_cubes_coordinates(resolution=256, direction=[0.,1.,0.], radius=1.25):
    direction = torch.tensor(direction).float()
    direction /= torch.linalg.norm(direction)
    lin_coords = torch.linspace(-1.,1.,resolution)
    x_coords, y_coords, z_coords = torch.meshgrid([lin_coords, lin_coords, lin_coords])
    positional_coords = torch.stack([x_coords, y_coords, z_coords], dim=-1).reshape((-1, 3))
    valid_positions = torch.linalg.norm(positional_coords, dim=-1) < 1.
    repeat_directions = torch.stack([direction]*positional_coords.shape[0], dim=0)
    coordinates = torch.cat([positional_coords, repeat_directions], dim=1)
    return coordinates, valid_positions

def show_mesh(verts, faces, normals):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    # o3d.visualization.draw_geometries([mesh])

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(mesh)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()

def run_inference(Network, Device, coordinates):
    all_depths = []
    all_masks = []

    batch_size = 10000
    for i in range(0, coordinates.shape[0], batch_size):
        DataTD = butils.sendToDevice(coordinates[i:i+batch_size, ...], Device)

        masks, depths = Network([DataTD], {})[0]
        depths = depths.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
        
        all_depths.append(depths)
        all_masks.append(masks)
    depths = np.concatenate(all_depths, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    depths = depths.flatten()
    masks = masks.flatten()
    return depths, masks
    

def extract_mesh(Network, Device, resolution=256):

    Network.eval()  # switch to evaluation mode
    Network.to(Device)
    
    coordinates, valid_mask = generate_marching_cubes_coordinates(resolution=resolution)
    coordinates = coordinates[valid_mask]
    valid_mask = np.array(valid_mask)
    final_depths = np.ones(valid_mask.shape, dtype=float)

    # DataPosEnc = coordinates.copy()
    # if UsePosEnc:
    #     for Idx, BD in enumerate(DataPosEnc):
    #         DataPosEnc[Idx] = torch.from_numpy(o2utils.get_positional_enc(BD.numpy())).to(torch.float32)

    depths, masks = run_inference(Network, Device, coordinates)
    # depths = depths.reshape((resolution, resolution, resolution))
    # masks = masks.reshape((resolution, resolution, resolution))

    masks = masks > 0.5
    depths[np.logical_not(masks)] = 1.0
    final_depths[valid_mask] = depths
    final_depths = final_depths.reshape((resolution, resolution, resolution))
    spacing = 2.0/(resolution-1)
    verts, faces, normals, _ = marching_cubes(final_depths, level=0.0, spacing=(spacing, spacing, spacing))
    
    return verts, faces, normals

def extract_mesh_multiple_directions(Network, Device, resolution=256):
    Network.eval()  # switch to evaluation mode
    Network.to(Device)

    directions = [[1.,0.,0.],
                  [-1.,0.,0.],
                  [0.,1.,0.],
                  [0.,-1.,0.],
                  [0.,0.,1.],
                  [0.,0.,-1.]]

    directions = [[1.,0.,0.]]

    all_depths = []
    all_masks = []

    for dir in tqdm(directions):
        coordinates, valid_mask = generate_marching_cubes_coordinates(resolution=resolution, direction=dir)
        depths, masks = run_inference(Network, Device, coordinates)
        print(depths.shape)
        all_depths.append(depths)
        all_masks.append(masks)

    threshold = len(directions)/2.
    depths = np.stack(all_depths, axis=-1)
    print(depths.shape)
    masks = np.stack(all_masks, axis=-1)
    masks = masks > 0.5
    depths[np.logical_not(masks)] = 1.0
    interior_count = np.sum(depths < 0., axis=1)
    print(f"interior count: {np.sum(interior_count)}")
    interior = interior_count >= threshold
    print(f"interior suM: {np.sum(interior)}")
    implicit_values = np.ones_like(interior)
    implicit_values[interior] = -1.
    print(f"iv sum: {np.sum(implicit_values)}")
    implicit_values = implicit_values.reshape((resolution, resolution, resolution))
    print(implicit_values.shape)

    spacing = 2.0/(resolution-1)
    verts, faces, normals, _ = marching_cubes(implicit_values, level=0.0, spacing=(spacing, spacing, spacing))
    
    return verts, faces, normals






Parser = argparse.ArgumentParser(description='Inference code for NeuralODFs.')
Parser.add_argument('--arch', help='Architecture to use.', choices=['standard'], default='standard')
Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('--no-posenc', help='Choose not to use positional encoding.', action='store_true', required=False)
Parser.add_argument('--resolution', help='Resolution of the mesh to extract', type=int, default=256)
Parser.set_defaults(no_posenc=False)

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    butils.seedRandom(Args.seed)
    nCores = 0#mp.cpu_count()

    usePosEnc = not Args.no_posenc
    print('[ INFO ]: Using positional encoding:', usePosEnc)
    if Args.arch == 'standard':
        NeuralODF = ODFSingleV3(input_size=(120 if usePosEnc else 6), radius=DEPTH_SAMPLER_RADIUS, coord_type=Args.coord_type, pos_enc=usePosEnc, n_layers=10)

    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Device = torch.device("cpu")
    NeuralODF.setupCheckpoint(Device)

    # verts, faces, normals = extract_mesh_multiple_directions(NeuralODF, Device, resolution=Args.resolution)
    verts, faces, normals = extract_mesh(NeuralODF, Device, resolution=Args.resolution)
    show_mesh(verts, faces, normals)
