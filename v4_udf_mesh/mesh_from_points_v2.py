import torch
import sys, os
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes


FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from depth_sampler_5d import DEPTH_SAMPLER_RADIUS
from losses import DepthLoss, IntersectionLoss, DepthFieldRegularizingLoss, ConstantRegularizingLoss
from odf_models import ODFSingleV3, ODFSingleV3Constant, ODFSingleV3SH, ODFSingleV3ConstantSH
from depth_odf_dataset_5d import DepthODFDatasetLoader as DDL
import v3_utils


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True)
    return points_grad


def generate_coordinates(resolution=256):
    direction = [0, 1, 0]
    direction = torch.tensor(direction).float()
    direction /= torch.linalg.norm(direction)
    lin_coords = torch.linspace(-MESH_RADIUS,MESH_RADIUS,resolution)
    x_coords, y_coords, z_coords = torch.meshgrid([lin_coords, -1, lin_coords])
    positional_coords = torch.stack([x_coords, y_coords, z_coords], dim=-1).reshape((-1, 3))

    repeat_directions = torch.stack([direction]*positional_coords.shape[0], dim=0)
    coordinates = torch.cat([positional_coords, repeat_directions], dim=1)
    return coordinates


def run_inference(Network, Device, coords, batch_size=256*256):
    Network.eval()
    Network.to(Device)
    all_depths = []
    all_masks = []

    sigmoid = torch.nn.Sigmoid()
    pred_depth = []
    pred_mask = []
    data = []
    loss = []
    data0 = []
    depthloss = IntersectionLoss()
    for i in range(0, coords.shape[0], batch_size):
        DataTD = v3_utils.sendToDevice(coords[i:i+batch_size, ...], Device)
        DataTD = torch.vstack(DataTD)
        DataTD0 = DataTD.cpu().numpy()

        step = torch.tensor(0.5).to(Device)
        clamp = torch.tensor(0.5).to(Device)
        for j in range(10):
            if j>0: 
                out = torch.where(output[1]>clamp, clamp, torch.min(step, output[1]))
                DataTD[:, :3] = (DataTD[:, :3]+DataTD[:, 3:]*out).detach()
                DataTD[:, 3:] *= -1
                output_rev = Network.forward([DataTD])[0]
                output_rev = list(output_rev)
                if len(output_rev) != 2:
                    output_rev[1] += sigmoid(output_rev[2])*output_rev[3]
                maskA = output_rev[1]>=clamp#step
                maskB = output_rev[1]>=first_out
                mask = torch.logical_not(torch.logical_or(maskA, maskB))
                #mask = torch.where(output_rev[1]>=first_out-0.05, torch.tensor(0.0).to(Device), torch.tensor(1.0).to(Device)) 
                #mask = torch.logical_and(mask, output_rev[1]<0.25)
                DataTD[:, :3] = (DataTD[:, :3]+DataTD[:, 3:]*output_rev[1]*mask).detach()
                DataTD[:, 3:] *= -1
            output = Network.forward([DataTD])[0]
            output = list(output)
            if len(output) != 2:
                output[1] += sigmoid(output[2])*output[3]
                #print(sigmoid(output[2])*output[3])
            if j==0:
                first_out = torch.maximum(output[1], step)
            
        pred_depth.append((output[1]).detach().cpu().numpy())
        pred_mask.append(sigmoid(output[0]).detach().cpu().numpy())
        data.append(DataTD.cpu().numpy())
        data0.append(DataTD0)
   
    return data, pred_depth, pred_mask, data0


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


def vis(data, pred_depth, pred_mask, data0):
    #print(data)
    mask = (pred_mask>0.5).reshape(-1)
    mask = (np.abs(pred_depth)<0.05).reshape(-1)
    mask = np.logical_and((pred_mask>0.5).reshape(-1), (np.abs(pred_depth)<0.05).reshape(-1))
    #mask = (gt_mask==1).reshape(-1)
    #print(np.unique(np.abs(pred_depth)[mask]))
    #maskD = (pred_depth<0.).reshape(-1) 
    #maskD = (gt_depth>=0).reshape(-1
    #maskboth = mask
    #maskboth[mask] = mask2[mask]
    depth = pred_depth
    #depth = gt_depth
    #mask = np.logical_and(mask, maskD)
    surf_pts = data[:,:3]+data[:,3:]*depth
    surf_pts = surf_pts[mask]
    #surf_pts = np.vstack((surf_pts, data[:,:3]))
    pcd = o3d.geometry.PointCloud()
    #pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 255], (len(surf_pts), 1)))
    pcd.points = o3d.utility.Vector3dVector(surf_pts)

    #gt_pts = data0[:,:3]+data0[:,3:]*gt_depth
    #gt_pts = gt_pts[(gt_mask==1).reshape(-1)]
    #surf_pts = np.vstack((surf_pts, data[:,:3]))
    #gt_pcd = o3d.geometry.PointCloud()
    #pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 255], (len(surf_pts), 1)))
    #gt_pcd.points = o3d.utility.Vector3dVector(gt_pts+[-2, 0, 0])

    #o3d.visualization.draw_geometries([pcd])
    spacing = 0.01
    radius = 1.0
    resolution = int(2*radius/spacing)+1
    print(resolution)
    implicit_values = np.ones((resolution, resolution, resolution))
    surf_pts = np.ceil((surf_pts+1)/spacing).astype(int)
    implicit_values[tuple(np.transpose(surf_pts))] = -1
    verts, faces, normals, _ = marching_cubes(implicit_values, level=0.0, spacing=(spacing, spacing, spacing))
    show_mesh(verts, faces)

    #o3d.io.write_point_cloud("100Depth_maskFromDepth_valSphereRad2.6.pcd", pcd)
    print(np.sum((gt_mask==1)==(pred_mask>0.5))/(gt_mask.reshape(-1).shape[0]))
    print(np.sum((gt_mask==1)==(np.abs(pred_depth)<0.05))/(gt_mask.reshape(-1).shape[0]))
    #print(np.sum((gt_mask==1)==(maskboth.reshape(-1, 1)))/(gt_mask.reshape(-1).shape[0]))
    print(np.sum((gt_mask==1)==(np.logical_and(pred_mask>0.5, np.abs(pred_depth)<0.05)))/(gt_mask.reshape(-1).shape[0]))


if __name__ == '__main__':
    Parser = v3_utils.BaselineParser
    #Parser.add_argument('--batch-size', help='Choose mini-batch size.', required=False, default=16, type=int)
    #Parser.add_argument('--learning-rate', help='Choose the learning rate.', default=0.001, type=float)
    Parser.add_argument('--n-layers', help="Number of layers in the network backbone", default=7, type=int)
    #Parser.add_argument('--rays-per-shape', help='Number of samples to use during testing.', default=1000, type=int)
    #Parser.add_argument('--val', help='Choose to test on the training data.', action='store_false', required=False)
    #Parser.add_argument('--additional-intersections', type=int, default=0, help="The number of addtional intersecting rays to generate per surface point")
    #Parser.add_argument('--near-surface-threshold', type=float, default=-1., help="Sample an additional near-surface (within threshold) point for each intersecting ray. No sampling if negative.")
    #Parser.add_argument('--tangent-rays-ratio', type=float, default=0., help="The proportion of sampled rays that should be roughly tangent to the object.")
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    v3_utils.seedRandom(Args.seed)
    
    nCores = 0#mp.cpu_count()

    if Args.arch == 'standard':
        NeuralODF = ODFSingleV3(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers)
    elif Args.arch == 'constant':
        NeuralODF = ODFSingleV3Constant(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers)
    elif Args.arch == 'SH':
        NeuralODF = ODFSingleV3SH(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers, degrees=Args.degrees)
        print('[ INFO ]: Degrees {}'.format(Args.degrees))
    elif Args.arch == 'SH_constant':
        NeuralODF = ODFSingleV3ConstantSH(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers, degrees=Args.degrees)
        print('[ INFO ]: Degrees {}'.format(Args.degrees))


    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # check to see if we have a model checkpoint
    if os.path.exists(os.path.join(Args.output_dir, Args.expt_name, "checkpoints")):
        checkpoint_dict = v3_utils.load_checkpoint(Args.output_dir, Args.expt_name, device=Device, load_best=True)
        NeuralODF.load_state_dict(checkpoint_dict['model_state_dict'])
        NeuralODF.to(Device)
    else:
        print(f"Unable to extract mesh for model {Args.expt_name} because no checkpoints were found")

    
    data, pred_depth, pred_mask, data0  = run_inference(NeuralODF, Device)
    vis(np.vstack(data), np.vstack(pred_depth), np.vstack(pred_mask), np.vstack(data0))
