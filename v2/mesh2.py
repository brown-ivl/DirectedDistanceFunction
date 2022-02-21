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
import trimesh
import matplotlib.pyplot as plt
from torch.nn.functional import pad
import time

# MASK_THRESH = 0.995
# MASK_THRESH = 0.80
MASK_THRESH = 0.5 #0.995

MESH_RADIUS = 1.0

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from odf_dataset import ODFDatasetLiveVisualizer, ODFDatasetVisualizer
# from pc_sampler import PC_SAMPLER_RADIUS
from depth_sampler_5d import DEPTH_SAMPLER_RADIUS
from single_losses import SingleDepthBCELoss, SINGLE_MASK_THRESH
from single_models import ODFSingleV3, ODFSingleV3SH, ODFSingleV3Constant, ODFSingleV3ConstantSH
# from pc_odf_dataset import PCODFDatasetLoader as PCDL
from depth_odf_dataset_5d import DepthODFDatasetLoader as DDL
from odf_dataset import ODFDatasetLoader as ODL
import odf_v2_utils as o2utils
from spherical_harmonics import SHV2, fibonnacci_sphere_sampling

# RADIUS = 1.25

# def mesh_normalize(verts):
#     '''
#     Translates and rescales mesh vertices so that they are tightly bounded within the unit sphere
#     '''
#     translation = (np.max(verts, axis=0) + np.min(verts, axis=0))/2.
#     verts = verts - translation
#     scale = np.max(np.linalg.norm(verts, axis=1))
#     verts = verts / scale
#     return verts

def load_object(obj_name, data_path):
    obj_file = os.path.join(data_path, f"{obj_name}.obj")

    obj_mesh = trimesh.load(obj_file)
    # obj_mesh.show()

    ## deepsdf normalization
    mesh_vertices = obj_mesh.vertices
    mesh_faces = obj_mesh.faces
    center = (mesh_vertices.max(axis=0) + mesh_vertices.min(axis=0))/2.0
    max_dist = np.linalg.norm(mesh_vertices - center, axis=1).max()
    max_dist = max_dist * 1.03
    mesh_vertices = (mesh_vertices - center) / max_dist
    obj_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
    
    return mesh_vertices, mesh_faces, obj_mesh


def generate_marching_cubes_coordinates(resolution=256, direction=[0.,1.,0.]):
    direction = torch.tensor(direction).float()
    direction /= torch.linalg.norm(direction)
    lin_coords = torch.linspace(-MESH_RADIUS,MESH_RADIUS,resolution)
    x_coords, y_coords, z_coords = torch.meshgrid([lin_coords, lin_coords, lin_coords])
    positional_coords = torch.stack([x_coords, y_coords, z_coords], dim=-1).reshape((-1, 3))
    repeat_directions = torch.stack([direction]*positional_coords.shape[0], dim=0)
    coordinates = torch.cat([positional_coords, repeat_directions], dim=1)
    return coordinates


def generate_marching_cubes_coordinates_sh(resolution=256):
    lin_coords = torch.linspace(-MESH_RADIUS,MESH_RADIUS,resolution)
    x_coords, y_coords, z_coords = torch.meshgrid([lin_coords, lin_coords, lin_coords])
    positional_coords = torch.stack([x_coords, y_coords, z_coords], dim=-1).reshape((-1, 3))
    valid_positions = torch.linalg.norm(positional_coords, dim=-1) < 1.
    return positional_coords, valid_positions


def show_mesh(verts, faces, ground_truth=None):
    import open3d as o3d
    gt_translation = [2.5,0.,0.]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    # mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.compute_vertex_normals()

    gt_mesh = None

    if ground_truth is not None:
        gt_mesh = o3d.geometry.TriangleMesh()
        gt_mesh.vertices = o3d.utility.Vector3dVector(ground_truth.vertices)
        gt_mesh.triangles = o3d.utility.Vector3iVector(ground_truth.faces)
        gt_mesh.compute_vertex_normals()
        gt_mesh.translate(gt_translation)
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(mesh)
    if gt_mesh is not None:
        viewer.add_geometry(gt_mesh)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()


def run_inference(Network, Device, coordinates):
    all_depths = []
    all_masks = []

    sigmoid = torch.nn.Sigmoid()

    batch_size = 5000
    for i in range(0, coordinates.shape[0], batch_size):
        DataTD = butils.sendToDevice(coordinates[i:i+batch_size, ...], Device)

        output = Network([DataTD], {})[0]

        if len(output) == 2:
            PredMaskConf, PredDepth = output
        else:
            PredMaskConf, PredDepth, PredMaskConst, PredConst = output
            PredDepth += sigmoid(PredMaskConst)*PredConst

        depths = PredDepth.detach().cpu().numpy()
        masks = sigmoid(PredMaskConf)
        masks = masks.detach().cpu().numpy()
        # masks, depths = Network([DataTD], {})[0]
        # depths = depths.detach().cpu().numpy()
        # masks = masks.detach().cpu().numpy()
        
        all_depths.append(depths)
        all_masks.append(masks)
    depths = np.concatenate(all_depths, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    depths = depths.flatten()
    masks = masks.flatten()
    return depths, masks

def run_inference_v2(Network, Device, coordinates):
    all_depths = []
    all_masks = []

    sigmoid = torch.nn.Sigmoid()

    batch_size = 5000
    for i in range(0, coordinates.shape[0], batch_size):
        DataTD = butils.sendToDevice(coordinates[i:i+batch_size, ...], Device)

        output = Network([DataTD], {})[0]

        if len(output) == 2:
            PredMaskConf, PredDepth = output
        else:
            PredMaskConf, PredDepth, PredMaskConst, PredConst = output
            PredDepth += sigmoid(PredMaskConst)*PredConst

        depths = PredDepth#.detach().cpu().numpy()
        masks = sigmoid(PredMaskConf)
        masks = masks#.detach().cpu().numpy()
        # masks, depths = Network([DataTD], {})[0]
        # depths = depths.detach().cpu().numpy()
        # masks = masks.detach().cpu().numpy()
        
        all_depths.append(depths)
        all_masks.append(masks)
    depths = torch.cat(all_depths, axis=0)
    masks = torch.cat(all_masks, axis=0)
    depths = torch.flatten(depths)
    masks = torch.flatten(masks)
    return depths, masks

# def run_inference(Network, Device, coordinates):
#     all_depths = []
#     all_masks = []

#     sigmoid = torch.nn.Sigmoid()

#     batch_size = 10000
#     for i in range(0, coordinates.shape[0], batch_size):
#         DataTD = butils.sendToDevice(coordinates[i:i+batch_size, ...], Device)

#         masks, depths = Network([DataTD], {})[0]
#         masks = sigmoid(masks)
#         depths = depths.detach().cpu().numpy()
#         masks = masks.detach().cpu().numpy()


        
#         all_depths.append(depths)
#         all_masks.append(masks)
#     depths = np.concatenate(all_depths, axis=0)
#     masks = np.concatenate(all_masks, axis=0)
#     depths = depths.flatten()
#     masks = masks.flatten()
#     return depths, masks

def show_mask_threshold_curve(depths, bounds_masks, intersection_masks, resolution, ground_truth):
    thresholds = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
    # thresholds = [0.1,0.5,0.9]
    pred_to_gt_dists = []
    gt_to_pred_dists = []

    for thresh in tqdm(thresholds):
        all_final_depths = []
        for i in range(len(depths)):
            final_depths = np.ones(bounds_masks[i].shape, dtype=float)
            curr_intersection_mask = intersection_masks[i] > thresh
            curr_depths = depths[i]
            curr_depths[np.logical_not(curr_intersection_mask)] = 1.0
            final_depths[bounds_masks[i]] = curr_depths
            final_depths = final_depths.reshape((resolution, resolution, resolution))
            all_final_depths.append(final_depths)
    
        threshold = len(depths)/2.
        stacked_depths = np.stack(all_final_depths, axis=-1)
        interior_count = np.sum(stacked_depths < 0., axis=-1)
        interior = interior_count > threshold
        implicit_values = np.ones(interior.shape)
        implicit_values[interior] = -1.
        
        # run marching cubes, compute mesh accuracy

        spacing = 2.*MESH_RADIUS/(resolution-1)
        verts, faces, _, _ = marching_cubes(implicit_values, level=0.0, spacing=(spacing, spacing, spacing))
        verts = verts - 1.0
        predicted_mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # print(f"Showing mesh with threshold: {thresh}")
        # show_mesh(verts, faces)

        pred_to_gt_dists.append(np.mean(np.abs(trimesh.proximity.signed_distance(ground_truth, predicted_mesh.vertices))))
        gt_to_pred_dists.append(np.mean(np.abs(trimesh.proximity.signed_distance(predicted_mesh, ground_truth.vertices))))




                # print(f"applying mask: {MASK_THRESH}")
                # masks = masks > MASK_THRESH
                # depths[np.logical_not(masks)] = 1.0
                # final_depths[valid_mask] = depths
                # final_depths = final_depths.reshape((resolution, resolution, resolution))
    f, ax = plt.subplots()
    ax.plot(thresholds, pred_to_gt_dists, label="Predicted to GT")
    ax.plot(thresholds, gt_to_pred_dists, label="GT to Predicted")
    ax.plot(thresholds, [(pred_to_gt_dists[i]+gt_to_pred_dists[i])/2. for i in range(len(pred_to_gt_dists))], label="Mean")
    ax.set_title("Mesh Reconstruction Error by Intersection Mask Threshold")
    ax.legend()
    plt.show()


def generate_grid_coordinates(resolution=256, Device='cpu'):
    lin_coords = torch.linspace(-MESH_RADIUS,MESH_RADIUS,resolution).to(Device)
    x_coords, y_coords, z_coords = torch.meshgrid([lin_coords, lin_coords, lin_coords])
    # (resolution, resolution, resolution, 3)
    positional_coords = torch.stack([x_coords, y_coords, z_coords], dim=-1)
    sphere_positions = torch.linalg.norm(positional_coords, dim=-1) < 1.
    # get points on the sphere surface
    sides = [1, 0, 0, 0, 0, 0]
    inside_sphere_positions = pad(sphere_positions, 
                                  sides, 
                                  'constant', 
                                  (False))[sides[5]:resolution+sides[5],
                                           sides[3]:resolution+sides[3],
                                           sides[1]:resolution+sides[1]]

    for i in range(1, 6):
        sides = [0, 0, 0, 0, 0, 0]
        sides[i] = 1
        inside_sphere_positions = torch.logical_and(inside_sphere_positions, 
                                                    pad(sphere_positions, 
                                                        sides, 
                                                        'constant', 
                                                        (False))[sides[5]:resolution+sides[5],
                                                                 sides[3]:resolution+sides[3],
                                                                 sides[1]:resolution+sides[1]])     
    surface_sphere_position = torch.logical_and(torch.logical_not(inside_sphere_positions), sphere_positions)

    return positional_coords, surface_sphere_position, sphere_positions


def get_meshing_coordinates(resolution, Network, Device='cpu'):
    # construct the grid
    coordinate, surface_mask, sphere_positions  = generate_grid_coordinates(resolution=resolution, Device=Device)
    # initial the final object mask
    object_mask = torch.zeros(coordinate.size()[:-1], dtype=torch.bool).to(Device)
    # margin for distance
    margin_dis = 0.0 #0.1
    margin_cell = 0.0
    # run inference on six sides
    for side in range(3):
        # ray for a pair of side
        ray = torch.zeros(coordinate.shape).to(Device)
        ray[:,:,:,2-side][coordinate[:,:,:,2-side]>=0] = -1.
        ray[:,:,:,2-side][coordinate[:,:,:,2-side]<0] = 1.
        # testing data
        data = torch.cat((coordinate, ray), -1)[surface_mask]
        depths, intersection_mask = run_inference_v2(Network, Device, data)
        depths = depths.view(-1, 1)
        intersection_mask = intersection_mask.view(-1, 1)>MASK_THRESH
        
        # put intersection mask back to grid coordinate
        surface_intersection_mask = torch.unsqueeze(torch.zeros(coordinate.size()[:-1], dtype=torch.bool).to(Device), -1)
        surface_intersection_mask[surface_mask] = intersection_mask
        # whether the row has surface points or not
        surface_intersection_mask = torch.sum(surface_intersection_mask, 2-side).expand(coordinate.size()[:-1])
        # put the coordinate of surface back to grid coordinate
        surface_min = torch.unsqueeze(torch.ones(coordinate.size()[:-1]).to(Device), -1)*1e8
        surface_max = surface_min*-1       
        surface_min[surface_mask] = data[:, 2-side:2-side+1]+depths*data[:, 2-side+3:2-side+4]
        surface_max[surface_mask] = surface_min[surface_mask]
        maxi_surface, maxi_surface_ind = torch.max(surface_max, dim=2-side)
        mini_surface, mini_surface_ind = torch.min(surface_min, dim=2-side)
        # mask to remove the potential non-interescting points
        valid_surface_mask = (maxi_surface_ind>=mini_surface_ind-margin_cell).expand(coordinate.size()[:-1])
        # get center and radius for each row
        radius = torch.sqrt(torch.square(maxi_surface-mini_surface))/2
        center = (maxi_surface+mini_surface)/2
        # mask for points inside the distance with radius for each row
        cur_object_mask = torch.sqrt(torch.square(coordinate[:,:,:,2-side]-center))<=radius+margin_dis
        object_mask[torch.logical_and(torch.logical_and(cur_object_mask, valid_surface_mask), surface_intersection_mask)] = True
    object_mask = torch.logical_and(object_mask, sphere_positions)
    print("Points for inference: {}".format(torch.sum(object_mask)))
    return object_mask.cpu()

def extract_mesh_multiple_directions(Network, Device, resolution=256, ground_truth=None):
    Network.eval()  # switch to evaluation mode
    Network.to(Device)
    start = time.time()
    """
    directions = [[1.,0.,0.],
                   [-1.,0.,0.],
                   [0.,1.,0.],
                   [0.,-1.,0.],
                   [0.,0.,1.],
                   [0.,0.,-1.]]
    """
    """   
    directions = [[1.,0.,0.],
                  [-1.,0.,0.],
                  [0.,1.,0.],
                  [0.,-1.,0.],
                  [0.,0.,1.],
                  [0.,0.,-1.],
                  [1.,1.,1.],
                  [1.,1.,-1.],
                  [1.,-1.,1.],
                  [-1.,1.,1.],
                  [1.,-1.,-1.],
                  [-1.,-1.,1.],
                  [-1.,1.,-1.],
                  [-1.,-1.,-1.],
                  ]
    """
    directions = [[1.,0.,0.]]

    # n_dirs = 100
    # directions = [np.random.normal(size=3) for _ in range(n_dirs)]
    # directions = [x/np.linalg.norm(x) for x in directions]

    all_depths = []
    all_intersection_masks = []
    all_bounds_masks = []

    meshing_mask = get_meshing_coordinates(resolution, Network, Device)

    
    for dir in tqdm(directions):
        coordinates = generate_marching_cubes_coordinates(resolution=resolution, direction=dir)
        bounds_mask = meshing_mask.view(-1)       
        coordinates = coordinates[bounds_mask]
        bounds_mask = np.array(bounds_mask)
        # final_depths = np.ones(bounds_mask.shape, dtype=float)


        depths, intersection_mask = run_inference(Network, Device, coordinates)
        # print(f"applying mask: {MASK_THRESH}")
        # masks = masks > MASK_THRESH
        # depths[np.logical_not(masks)] = 1.0
        # final_depths[valid_mask] = depths
        # final_depths = final_depths.reshape((resolution, resolution, resolution))


        all_bounds_masks.append(bounds_mask)
        all_depths.append(depths)
        all_intersection_masks.append(intersection_mask)

    # if ground_truth is not None:
    #     print("Calculating distances")
    #     show_mask_threshold_curve(all_depths, all_bounds_masks, all_intersection_masks, resolution, ground_truth)

    all_final_depths = []
    for i in range(len(all_depths)):
        final_depths = np.ones(all_bounds_masks[i].shape, dtype=float)
        curr_intersection_mask = all_intersection_masks[i] > MASK_THRESH
        curr_depths = all_depths[i]
        curr_depths[np.logical_not(curr_intersection_mask)] = 1.0
        final_depths[all_bounds_masks[i]] = curr_depths
        final_depths = final_depths.reshape((resolution, resolution, resolution))
        all_final_depths.append(final_depths)

    threshold = len(directions)/2.
    stacked_depths = np.stack(all_final_depths, axis=-1)
    interior_count = np.sum(stacked_depths < 0., axis=-1)
    interior = interior_count > threshold
    implicit_values = np.ones(interior.shape)
    implicit_values[interior] = -1.
    
    # run marching cubes, compute mesh accuracy
    print(np.max(implicit_values), np.min(implicit_values))
    spacing = 2.*MESH_RADIUS/(resolution-1)
    verts, faces, normals, _ = marching_cubes(implicit_values, level=0.0, spacing=(spacing, spacing, spacing))
    verts = verts - 1.0
    end = time.time()
    print(end-start)
    return verts, faces, normals


def extract_mesh_multiple_directions_sh(Network, Device, degrees, resolution=256, ground_truth=None):
    Network.eval()  # switch to evaluation mode
    Network.to(Device)
    directions = fibonnacci_sphere_sampling(100000)
    """    
    directions = [[1.,0.,0.],
                  [-1.,0.,0.],
                  [0.,1.,0.],
                  [0.,-1.,0.],
                  [0.,0.,1.],
                  [0.,0.,-1.],
                  [1.,1.,1.],
                  [1.,1.,-1.],
                  [1.,-1.,1.],
                  [-1.,1.,1.],
                  [1.,-1.,-1.],
                  [-1.,-1.,1.],
                  [-1.,1.,-1.],
                  [-1.,-1.,-1.],
                  ]
    """
    #directions = [[1., 0., 0.]]
    threshold = len(directions)/2.
    # get all sh
    sh = SHV2(max(degrees), torch.tensor(directions).to(Device), Device) 
    
    # get coordinate from grid
    coordinates, bounds_mask = generate_marching_cubes_coordinates_sh(resolution=resolution)
    #coordinates = coordinates[bounds_mask]
    bounds_mask = torch.tensor(bounds_mask).to(Device) 
    implicit_values = torch.ones(bounds_mask.size()).to(Device)

    sigmoid = torch.nn.Sigmoid()
    # run inference
    batch_size = 5000
    for i in range(0, coordinates.shape[0], batch_size):
        DataTD = butils.sendToDevice(coordinates[i:i+batch_size, ...], Device)
        output = Network([DataTD], {})[0]
        if len(output) == 2:
            PredMaskConf_coeff, PredDepth_coeff = output
            PredDepth = sh.linear_combination(degrees[0], PredDepth_coeff)
            PredMaskConf = sigmoid(sh.linear_combination(degrees[1], PredMaskConf_coeff))
            
        else:
            PredMaskConf_coeff, PredDepth_coeff, PredMaskConst_coeff, PredConst_coeff = output
            PredDepth = sh.linear_combination(degrees[0], PredDepth_coeff)
            PredMaskConf = sigmoid(sh.linear_combination(degrees[1], PredMaskConf_coeff))
            PredConst = sh.linear_combination(degrees[2], PredConst_coeff)
            PredMaskConst = sh.linear_combination(degrees[3], PredMaskConst_coeff)
            PredDepth += sigmoid(PredMaskConst)*PredConst

        curr_intersection_mask = PredMaskConf<=MASK_THRESH
        #cur_batch_size = PredMaskConf.size()[0]
        PredDepth[curr_intersection_mask] = 1.0 
        inter_count = torch.sum(PredDepth<0., dim=-1)
        interior = inter_count > threshold
        implicit_values[i:i+batch_size][torch.logical_and(interior, bounds_mask[i:i+batch_size])] = -1.0

    
    # run marching cubes, compute mesh accuracy

    spacing = 2.*MESH_RADIUS/(resolution-1)
    implicit_values = implicit_values.view(resolution, resolution, resolution).cpu().numpy()
    verts, faces, normals, _ = marching_cubes(implicit_values, level=0.0, spacing=(spacing, spacing, spacing))
    verts = verts - 1.0

    
    return verts, faces, normals


Parser = argparse.ArgumentParser(description='Inference code for NeuralODFs.')
Parser.add_argument('--arch', help='Architecture to use.', choices=['standard', 'constant', 'SH', 'SH_constant'], default='standard')
Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('--no-posenc', help='Choose not to use positional encoding.', action='store_true', required=False)
Parser.add_argument('--resolution', help='Resolution of the mesh to extract', type=int, default=256)
Parser.add_argument('--mesh-dir', help="Mesh with ground truth .obj files", type=str)
Parser.add_argument('--object', help="Name of the object", type=str)
Parser.add_argument('--degrees', help='degree for [depth, intersect]|[depth, intersect, const, const mask]', type=lambda ds:[int(d) for d in ds.split(',')], required=False, default=[2, 2])


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
        print("Using original architecture")
        NeuralODF = ODFSingleV3(input_size=(120 if usePosEnc else 6), radius=DEPTH_SAMPLER_RADIUS, coord_type=Args.coord_type, pos_enc=usePosEnc, n_layers=10)
    elif Args.arch == 'SH':
        NeuralODF = ODFSingleV3SH(input_size=(120 if usePosEnc else 6), radius=DEPTH_SAMPLER_RADIUS, coord_type=Args.coord_type, pos_enc=usePosEnc, n_layers=10, degrees=Args.degrees, return_coeff=True)
        print("Using spherical harmonics architecture")
    elif Args.arch == 'constant':
        print("Using constant prediction architecture")
        NeuralODF = ODFSingleV3Constant(input_size=(120 if usePosEnc else 6), radius=DEPTH_SAMPLER_RADIUS, coord_type=Args.coord_type, pos_enc=usePosEnc, n_layers=10)
    elif Args.arch == 'SH_constant':
        NeuralODF = ODFSingleV3ConstantSH(input_size=(120 if usePosEnc else 6), radius=DEPTH_SAMPLER_RADIUS, coord_type=Args.coord_type, pos_enc=usePosEnc, n_layers=10, degrees=Args.degrees, return_coeff=True)
        print('[ INFO ]: Degress {}'.format(Args.degrees))


    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {Device}")
    # Device = torch.device("cpu")
    NeuralODF.setupCheckpoint(Device)

    gt_mesh = None
    if Args.object is not None and Args.mesh_dir is not None:
        gt_vertices, gt_faces, gt_mesh = load_object(Args.object, Args.mesh_dir)
        gt_mesh = trimesh.load(os.path.join(Args.mesh_dir, f"{Args.object}.obj"))
    else:
        print("Provide a mesh directory and object name if you want to visualize the ground truth.")


    # verts, faces, normals = extract_mesh(NeuralODF, Device, resolution=Args.resolution)
    verts, faces, normals = extract_mesh_multiple_directions(NeuralODF, Device, resolution=Args.resolution, ground_truth=gt_mesh)
    #verts, faces, normals = extract_mesh_multiple_directions_sh(NeuralODF, Device, resolution=Args.resolution, ground_truth=gt_mesh, degrees=Args.degrees)

    show_mesh(verts, faces, ground_truth=gt_mesh)







