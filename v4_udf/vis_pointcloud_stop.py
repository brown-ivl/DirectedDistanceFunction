import torch
import sys, os
import numpy as np
import open3d as o3d


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


def run_inference(Network, Device, DataLoader):
    Network.eval()
    Network.to(Device)
    all_depths = []
    all_masks = []

    sigmoid = torch.nn.Sigmoid()
    pred_depth = []
    pred_mask = []
    data = []
    gt_depth = []
    gt_mask = []
    loss = []
    data0 = []
    depthloss = IntersectionLoss()
    for i, (Data, Targets) in enumerate(DataLoader, 0):
        DataTD = v3_utils.sendToDevice(Data, Device)
        #TargetsTD = butils.sendToDevice(Targets, Device)  
        DataTD = torch.vstack(DataTD)
        TargetsTDInt = np.vstack(np.array(Targets)[:, 0])
        TargetsTDDep = np.vstack(np.array(Targets)[:, 1])
        TargetsTD = (torch.tensor(TargetsTDInt).to(Device), torch.tensor(TargetsTDDep).to(Device))
        mask = torch.ones(TargetsTDDep.shape).to(Device)
        DataTD0 = DataTD.cpu().numpy()
        for j in range(14):
            output = Network.forward([DataTD])[0]
            mask = torch.where(output[1]<0.5, torch.tensor(0.0).to(Device), torch.tensor(1.0).to(Device))
            DataTD[:, :3] = (DataTD[:, :3]+DataTD[:, 3:]*0.5*mask).detach()                     
        output = Network.forward([DataTD])[0]           

            #normals = gradient(DataTD, output[1])[0][:, :3]
        #DataTD.requires_grad = False           
        loss.append(depthloss.forward([output], [TargetsTD]).detach().cpu().numpy())
        pred_depth.append((output[1]).detach().cpu().numpy())
        pred_mask.append(sigmoid(output[0]).detach().cpu().numpy())
        gt_mask.append(TargetsTDInt)
        gt_depth.append(TargetsTDDep)
        data.append(DataTD.cpu().numpy())
        data0.append(DataTD0)
    print(np.mean(loss))
    return data, gt_depth, gt_mask, pred_depth, pred_mask, data0


def vis(data, gt_depth, gt_mask, pred_depth, pred_mask, data0):
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

    gt_pts = data0[:,:3]+data0[:,3:]*gt_depth
    gt_pts = gt_pts[(gt_mask==1).reshape(-1)]
    #surf_pts = np.vstack((surf_pts, data[:,:3]))
    gt_pcd = o3d.geometry.PointCloud()
    #pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 255], (len(surf_pts), 1)))
    gt_pcd.points = o3d.utility.Vector3dVector(gt_pts+[-2, 0, 0])

    o3d.visualization.draw_geometries([pcd, gt_pcd])
    #o3d.io.write_point_cloud("100Depth_maskFromDepth_valSphereRad2.6.pcd", pcd)
    print(np.sum((gt_mask==1)==(pred_mask>0.5))/(gt_mask.reshape(-1).shape[0]))
    print(np.sum((gt_mask==1)==(np.abs(pred_depth)<0.2))/(gt_mask.reshape(-1).shape[0]))
    #print(np.sum((gt_mask==1)==(maskboth.reshape(-1, 1)))/(gt_mask.reshape(-1).shape[0]))
    print(np.sum((gt_mask==1)==(np.logical_and(pred_mask>0.5, np.abs(pred_depth)<0.2)))/(gt_mask.reshape(-1).shape[0]))


if __name__ == '__main__':
    Parser = v3_utils.BaselineParser
    Parser.add_argument('--batch-size', help='Choose mini-batch size.', required=False, default=16, type=int)
    Parser.add_argument('--learning-rate', help='Choose the learning rate.', default=0.001, type=float)
    Parser.add_argument('--n-layers', help="Number of layers in the network backbone", default=7, type=int)
    Parser.add_argument('--rays-per-shape', help='Number of samples to use during testing.', default=1000, type=int)
    Parser.add_argument('--val', help='Choose to test on the training data.', action='store_false', required=False)
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
    Data = DDL(root=Args.input_dir, name=Args.dataset, train=Args.val, download=False, target_samples=Args.rays_per_shape, usePositionalEncoding=Args.use_posenc, aug=False)
    print(f"DATA SIZE: {len(Data)}")
    print('[ INFO ]: Data has {} shapes and {} rays per sample.'.format(len(Data), Args.rays_per_shape))

    DataLoader = torch.utils.data.DataLoader(Data, batch_size=Args.batch_size, shuffle=False, num_workers=nCores, collate_fn=DDL.collate_fn)


    # check to see if we have a model checkpoint
    if os.path.exists(os.path.join(Args.output_dir, Args.expt_name, "checkpoints")):
        checkpoint_dict = v3_utils.load_checkpoint(Args.output_dir, Args.expt_name, device=Device, load_best=True)
        NeuralODF.load_state_dict(checkpoint_dict['model_state_dict'])
        NeuralODF.to(Device)
    else:
        print(f"Unable to extract mesh for model {Args.expt_name} because no checkpoints were found")

    
    data, gt_depth, gt_mask, pred_depth, pred_mask, data0  = run_inference(NeuralODF, Device, DataLoader)
    vis(np.vstack(data), np.vstack(gt_depth), np.vstack(gt_mask), np.vstack(pred_depth), np.vstack(pred_mask), np.vstack(data0))
