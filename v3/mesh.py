import torch
import argparse
import math
import os, sys

import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from skimage.measure import marching_cubes
import trimesh
import matplotlib.pyplot as plt

# MASK_THRESH = 0.995
MASK_THRESH = 0.10
# MASK_THRESH = 0.995

MESH_RADIUS = 1.0

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from depth_sampler_5d import DEPTH_SAMPLER_RADIUS
from odf_models import ODFSingleV3, ODFSingleV3Constant
import v3_utils

# RADIUS = 1.25

def generate_marching_cubes_coordinates(resolution=256, direction=[0.,1.,0.]):
    direction = torch.tensor(direction).float()
    direction /= torch.linalg.norm(direction)
    lin_coords = torch.linspace(-MESH_RADIUS,MESH_RADIUS,resolution)
    x_coords, y_coords, z_coords = torch.meshgrid([lin_coords, lin_coords, lin_coords])
    positional_coords = torch.stack([x_coords, y_coords, z_coords], dim=-1).reshape((-1, 3))
    valid_positions = torch.linalg.norm(positional_coords, dim=-1) < 1.
    repeat_directions = torch.stack([direction]*positional_coords.shape[0], dim=0)
    coordinates = torch.cat([positional_coords, repeat_directions], dim=1)
    return coordinates, valid_positions

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
        DataTD = v3_utils.sendToDevice(coordinates[i:i+batch_size, ...], Device)

        output = Network([DataTD])[0]

        if len(output) == 2:
            PredMaskConf, PredDepth = output
        else:
            PredMaskConf, PredDepth, PredMaskConst, PredConst = output
            PredDepth += sigmoid(PredMaskConst)*PredConst

        depths = PredDepth.detach().cpu().numpy()
        masks = sigmoid(PredMaskConf)
        masks = masks.detach().cpu().numpy()
        
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


    f, ax = plt.subplots()
    ax.plot(thresholds, pred_to_gt_dists, label="Predicted to GT")
    ax.plot(thresholds, gt_to_pred_dists, label="GT to Predicted")
    ax.plot(thresholds, [(pred_to_gt_dists[i]+gt_to_pred_dists[i])/2. for i in range(len(pred_to_gt_dists))], label="Mean")
    ax.set_title("Mesh Reconstruction Error by Intersection Mask Threshold")
    ax.legend()
    plt.show()


def extract_mesh_multiple_directions(Network, Device, resolution=256, ground_truth=None, show_curve=True):
    Network.eval()  # switch to evaluation mode
    Network.to(Device)

    # directions = [[1.,0.,0.],
    #               [-1.,0.,0.],
    #               [0.,1.,0.],
    #               [0.,-1.,0.],
    #               [0.,0.,1.],
    #               [0.,0.,-1.]]


    # directions = [[1.,0.,0.],
    #               [-1.,0.,0.],
    #               [0.,1.,0.],
    #               [0.,-1.,0.],
    #               [0.,0.,1.],
    #               [0.,0.,-1.],
    #               [1.,1.,1.],
    #               [1.,1.,-1.],
    #               [1.,-1.,1.],
    #               [-1.,1.,1.],
    #               [1.,-1.,-1.],
    #               [-1.,-1.,1.],
    #               [-1.,1.,-1.],
    #               [-1.,-1.,-1.],
    #               ]

    directions = [[1.,1.,0.]]

    # n_dirs = 100
    # directions = [np.random.normal(size=3) for _ in range(n_dirs)]
    # directions = [x/np.linalg.norm(x) for x in directions]

    all_depths = []
    all_intersection_masks = []
    all_bounds_masks = []


    for dir in tqdm(directions):

        coordinates, bounds_mask = generate_marching_cubes_coordinates(resolution=resolution, direction=dir)
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

    if ground_truth is not None and show_curve:
        print("Calculating distances")
        show_mask_threshold_curve(all_depths, all_bounds_masks, all_intersection_masks, resolution, ground_truth)

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

    spacing = 2.*MESH_RADIUS/(resolution-1)
    verts, faces, normals, _ = marching_cubes(implicit_values, level=0.0, spacing=(spacing, spacing, spacing))
    verts = verts - 1.0
    
    return verts, faces, normals







if __name__ == '__main__':
    Parser = v3_utils.BaselineParser
    Parser.add_argument('--resolution', help='Resolution of the mesh to extract', type=int, default=256)
    Parser.add_argument('--mesh-dir', help="Mesh with ground truth .obj files", type=str)
    Parser.add_argument('--object', help="Name of the object", type=str)


    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    v3_utils.seedRandom(Args.seed)
    nCores = 0#mp.cpu_count()
    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {Device}")

    if Args.arch == 'standard':
        print("Using original architecture")
        NeuralODF = ODFSingleV3(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=10)
    elif Args.arch == 'constant':
        print("Using constant prediction architecture")
        NeuralODF = ODFSingleV3Constant(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=10)


    # check to see if we have a model checkpoint
    if os.path.exists(os.path.join(Args.output_dir, Args.expt_name, "checkpoints")):
        checkpoint_dict = v3_utils.load_checkpoint(Args.output_dir, Args.expt_name, device=Device, load_best=True)
        NeuralODF.load_state_dict(checkpoint_dict['model_state_dict'])
        NeuralODF.to(Device)
    else:
        print(f"Unable to extract mesh for model {Args.expt_name} because no checkpoints were found")

    gt_mesh = None
    if Args.mesh_dir is not None and Args.object is not None:
        gt_vertices, gt_faces, gt_mesh = v3_utils.load_object(Args.object, Args.mesh_dir)
    else:
        print("Provide a mesh directory and object name if you want to visualize the ground truth.")


    # verts, faces, normals = extract_mesh(NeuralODF, Device, resolution=Args.resolution)
    verts, faces, normals = extract_mesh_multiple_directions(NeuralODF, Device, resolution=Args.resolution, ground_truth=gt_mesh)

    show_mesh(verts, faces, ground_truth=gt_mesh)







