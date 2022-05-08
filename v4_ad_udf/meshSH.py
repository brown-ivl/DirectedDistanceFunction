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
MASK_THRESH = 0.5
# MASK_THRESH = 0.995

MESH_RADIUS = 1.0

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from depth_sampler_5d import DEPTH_SAMPLER_RADIUS
from odf_models import ODFSingleV3SH, ODFSingleV3ConstantSH
from spherical_harmonics import fibonnacci_sphere_sampling, SHV2
import v3_utils

# RADIUS = 1.25

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
        DataTD = v3_utils.sendToDevice(coordinates[i:i+batch_size, ...], Device)
        output = Network([DataTD])[0]
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







if __name__ == '__main__':
    Parser = v3_utils.BaselineParser
    Parser.add_argument('--resolution', help='Resolution of the mesh to extract', type=int, default=256)
    Parser.add_argument('--mesh-dir', help="Mesh with ground truth .obj files", type=str)
    Parser.add_argument('--n-layers', help="Number of layers in the network backbone", default=7, type=int)
    Parser.add_argument('--object', help="Name of the object", type=str)
    Parser.add_argument('--show-curve', action="store_true", help="Show the mesh similarity curve using different mask threshold values.")


    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    v3_utils.seedRandom(Args.seed)
    nCores = 0#mp.cpu_count()
    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {Device}")

    if Args.arch == 'SH':
        print("Using SH architecture")
        NeuralODF = ODFSingleV3SH(input_size=(120 if Args.use_posenc else 6), degrees=Args.degrees, radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers, return_coeff=True)
    elif Args.arch == 'SH_constant':
        print("Using SH constant prediction architecture")
        NeuralODF = ODFSingleV3SHConstant(input_size=(120 if Args.use_posenc else 6), degrees=Args.degrees, radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers, return_coeff=True)


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
    verts, faces, normals = extract_mesh_multiple_directions_sh(NeuralODF, Device, degrees=Args.degrees, resolution=Args.resolution, ground_truth=gt_mesh)

    show_mesh(verts, faces, ground_truth=gt_mesh)







