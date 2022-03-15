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

# MASK_THRESH = 0.995
MASK_THRESH = 0.5 #0.80
# MASK_THRESH = 0.995

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


def extract_mesh_multiple_directions_sh(Network, Device, degrees, resolution=256, ground_truth=None):
    Network.eval()  # switch to evaluation mode
    Network.to(Device)
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
    #directions = [[1., 0., 0.]]
    directions = fibonnacci_sphere_sampling(10000)
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
    if Args.arch == 'SH':
        NeuralODF = ODFSingleV3SH(input_size=(120 if usePosEnc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=usePosEnc, n_layers=10, degrees=Args.degrees, return_coeff=True)
        print("Using spherical harmonics architecture")
    elif Args.arch == 'SH_constant':
        NeuralODF = ODFSingleV3ConstantSH(input_size=(120 if usePosEnc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=usePosEnc, n_layers=10, degrees=Args.degrees, return_coeff=True)
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


    verts, faces, normals = extract_mesh_multiple_directions_sh(NeuralODF, Device, resolution=Args.resolution, ground_truth=gt_mesh, degrees=Args.degrees)

    show_mesh(verts, faces, ground_truth=gt_mesh)
