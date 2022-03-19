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
MASK_THRESH = 0.99
# MASK_THRESH = 0.995

MESH_RADIUS = 1.0

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from depth_sampler_5d import DEPTH_SAMPLER_RADIUS
from odf_models import ODFSingleV3, ODFSingleV3Constant, ODFV5, IntersectionMask3D, IntersectionMask3DV2
import v5_utils

# RADIUS = 1.25

def generate_marching_cubes_coordinates(resolution=256):
    lin_coords = torch.linspace(-MESH_RADIUS,MESH_RADIUS,resolution)
    x_coords, y_coords, z_coords = torch.meshgrid([lin_coords, lin_coords, lin_coords])
    positional_coords = torch.stack([x_coords, y_coords, z_coords], dim=-1).reshape((-1, 3, 1))
    valid_positions = torch.linalg.norm(positional_coords.view((-1,3)), dim=-1) < 1.
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
        DataTD = v5_utils.sendToDevice(coordinates[i:i+batch_size, ...], Device)

        output = Network([DataTD])[0]

        masks = output.detach().cpu().numpy()
        
        all_masks.append(masks)
    masks = np.concatenate(all_masks, axis=0)
    masks = masks.flatten()
    return masks


def extract_mesh_3D_mask(Network, Device, resolution=256, ground_truth=None):
    Network.eval()
    Network = Network.to(Device)

    coordinates, bounds_mask = generate_marching_cubes_coordinates(resolution=resolution)
    coordinates = coordinates[bounds_mask]
    bounds_mask = np.array(bounds_mask)

    masks = run_inference(Network, Device, coordinates)

    final_masks = np.ones((bounds_mask.shape[0],))
    masks[masks < 0.5] = -1.
    final_masks[bounds_mask] = masks
    final_masks = final_masks.reshape((resolution, resolution, resolution))

    spacing = 2.*MESH_RADIUS/(resolution-1)
    verts, faces, normals, _ = marching_cubes(final_masks, level=0.0, spacing=(spacing, spacing, spacing))
    verts = verts - 1.0

    return verts, faces, normals







if __name__ == '__main__':
    Parser = v5_utils.BaselineParser
    Parser.add_argument('--resolution', help='Resolution of the mesh to extract', type=int, default=256)
    Parser.add_argument('--mesh-dir', help="Mesh with ground truth .obj files", type=str)
    Parser.add_argument('--n-layers', help="Number of layers in the network backbone", default=7, type=int)
    Parser.add_argument('--object', help="Name of the object", type=str)
    Parser.add_argument('--show-curve', action="store_true", help="Show the mesh similarity curve using different mask threshold values.")


    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    v5_utils.seedRandom(Args.seed)
    nCores = 0#mp.cpu_count()
    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {Device}")

    Mask3D = IntersectionMask3DV2(dim=3, hidden_size=256)




    # check to see if we have a model checkpoint
    if os.path.exists(os.path.join(Args.output_dir, Args.expt_name, "checkpoints")):
        checkpoint_dict = v5_utils.load_checkpoint(Args.output_dir, Args.expt_name, device=Device, load_best=True)
        Mask3D.load_state_dict(checkpoint_dict['mask_model_state_dict'])
        Mask3D = Mask3D.to(Device)
    else:
        print(f"Unable to extract mesh for model {Args.expt_name} because no checkpoints were found")

    gt_mesh = None
    if Args.mesh_dir is not None and Args.object is not None:
        gt_vertices, gt_faces, gt_mesh = v5_utils.load_object(Args.object, Args.mesh_dir)
    else:
        print("Provide a mesh directory and object name if you want to visualize the ground truth.")


    # verts, faces, normals = extract_mesh(NeuralODF, Device, resolution=Args.resolution)
    verts, faces, normals = extract_mesh_3D_mask(Mask3D, Device, resolution=Args.resolution, ground_truth=gt_mesh)

    show_mesh(verts, faces, ground_truth=gt_mesh)







