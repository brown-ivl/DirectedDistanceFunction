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
from odf_models import ODFSingleV3, ODFSingleV3Constant, ODFSingleV3SH, ODFSingleV3ConstantSH, ODFADV3
from depth_odf_dataset_5d import DepthODFDatasetLoader as DDL
import v3_utils

def run_inference(Network, Device, DataLoader, embedding):
    Network.eval()
    Network.to(Device)
    all_depths = []
    all_masks = []

    sigmoid = torch.nn.Sigmoid()
    pred_depth = []
    last_pred_depth = []
    pred_mask = []
    data = []
    gt_depth = []
    gt_mask = []
    loss = []
    depthloss = IntersectionLoss()
    for i, (Data, Targets) in enumerate(DataLoader, 0):
        
        #DataTD = v3_utils.sendToDevice(Data, Device)
        #TargetsTD = butils.sendToDevice(Targets, Device)  
        #DataTD = torch.vstack(DataTD)
        #print(Data)
        DataTDA = np.vstack(np.array(Data)[:, 0])
        DataTDB = np.hstack(np.array(Data)[:, 1])
        DataTD = [torch.tensor(DataTDA).to(Device), torch.tensor(DataTDB).to(Device)]
        data.append(DataTDA)
        #print([DataTD])
        TargetsTDInt = np.vstack(np.array(Targets)[:, 0])
        TargetsTDDep = np.vstack(np.array(Targets)[:, 1])
        TargetsTD = (torch.tensor(TargetsTDInt).to(Device), torch.tensor(TargetsTDDep).to(Device))
        pred_acc_depth = np.zeros(TargetsTDDep.shape)
        for j in range(20):
            if j!=0:
                DataTD[0][:, :3] = (DataTD[0][:, :3]+DataTD[0][:, 3:]*torch.clamp(output[1], min=-0.5, max=0.5)).detach()
            output = Network.forward([DataTD], embedding)[0]
            pred_acc_depth += torch.clamp(output[1], min=-0.5, max=0.5).detach().cpu().numpy()
        loss.append(depthloss.forward([output], [TargetsTD]).detach().cpu().numpy())
        last_pred_depth.append(output[1].detach().cpu().numpy())
        pred_depth.append(pred_acc_depth)
        pred_mask.append(sigmoid(output[0]).detach().cpu().numpy())
        gt_mask.append(TargetsTDInt)
        gt_depth.append(TargetsTDDep)
        #data.append(DataTD.cpu().numpy())
    print(np.mean(loss))
    return data, gt_depth, gt_mask, pred_depth, pred_mask, last_pred_depth


def vis(data, gt_depth, gt_mask, pred_depth, pred_mask, last_pred_depth):
    #print(data)
    mask = (pred_mask>0.5).reshape(-1)
    mask = (np.abs(last_pred_depth)<0.01).reshape(-1)
    mask = np.logical_and((pred_mask>0.5).reshape(-1), (np.abs(last_pred_depth)<0.01).reshape(-1))
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
    #color = surf_pts
    #color = (color-np.min(color, 0))/(np.max(color, 0)-np.min(color, 0))
    #color = (color*255).astype(int)
    #print(pcd.colors)
    #pcd.colors = o3d.utility.Vector3dVector(color)
    pcd.points = o3d.utility.Vector3dVector(surf_pts)
    #o3d.visualization.draw_geometries([pcd])
    #print(np.unique(np.where(gt_mask<99, gt_depth, 0)))
    gt_depth[gt_depth<0] = 0
    gt_pts = data[:,:3]+data[:,3:]*gt_depth
    gt_pts = gt_pts[gt_mask.reshape(-1)==1]
    pcd_gt = o3d.geometry.PointCloud()
    #pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 255], (len(surf_pts), 1)))
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pts+[-2, 0, 0])
    o3d.visualization.draw_geometries([pcd, pcd_gt])

    #o3d.io.write_point_cloud("20Depth_augbothdep_maskFromBoth_valInside.pcd", pcd)
    print(np.sum((gt_mask==1)==(pred_mask>0.5))/(gt_mask.reshape(-1).shape[0]))
    print(np.sum((gt_mask==1)==(np.abs(last_pred_depth)<0.01))/(gt_mask.reshape(-1).shape[0]))
    #print(np.sum((gt_mask==1)==(maskboth.reshape(-1, 1)))/(gt_mask.reshape(-1).shape[0]))
    print(np.sum((gt_mask==1)==(np.logical_and(pred_mask>0.5, np.abs(last_pred_depth)<0.01)))/(gt_mask.reshape(-1).shape[0]))



if __name__ == '__main__':
    Parser = v3_utils.BaselineParser
    Parser.add_argument('--batch-size', help='Choose mini-batch size.', required=False, default=16, type=int)
    Parser.add_argument('--inst', help='Choose instance', required=False, default=0, type=int)
    Parser.add_argument('--learning-rate', help='Choose the learning rate.', default=0.001, type=float)
    Parser.add_argument('--n-layers', help="Number of layers in the network backbone", default=7, type=int)
    Parser.add_argument('--rays-per-shape', help='Number of samples to use during testing.', default=1000, type=int)
    Parser.add_argument('--val', help='Choose to test on the training data.', action='store_false', required=False)
    Parser.add_argument('--latent-size', type=int, default=256, help="Size of latent vectors in autodecoder")

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
        NeuralODF = ODFADV3(input_size=(120 if Args.use_posenc else 6), latent_size=Args.latent_size, radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers)
    elif Args.arch == 'constant':
        NeuralODF = ODFSingleV3Constant(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers)
    elif Args.arch == 'SH':
        NeuralODF = ODFSingleV3SH(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers, degrees=Args.degrees)
        print('[ INFO ]: Degrees {}'.format(Args.degrees))
    elif Args.arch == 'SH_constant':
        NeuralODF = ODFSingleV3ConstantSH(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers, degrees=Args.degrees)
        print('[ INFO ]: Degrees {}'.format(Args.degrees))


    instance_index_map_org = v3_utils.read_instance_index_map(Args.output_dir, Args.expt_name)
    ins_key = list(instance_index_map_org.keys())[Args.inst]
    instance_index_map = {ins_key: instance_index_map_org[ins_key]}
    print(instance_index_map)
    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Data = DDL(root=Args.input_dir, name=Args.dataset, train=Args.val, download=False, ad=True, target_samples=Args.rays_per_shape, usePositionalEncoding=Args.use_posenc, instance_index_map=instance_index_map, aug=False)
    print(f"DATA SIZE: {len(Data)}")
    print('[ INFO ]: Data has {} shapes and {} rays per sample.'.format(len(Data), Args.rays_per_shape))

    DataLoader = torch.utils.data.DataLoader(Data, batch_size=Args.batch_size, shuffle=False, num_workers=nCores, collate_fn=DDL.collate_fn)


    N_LATENTS = 20 #20 #1000

    # Initialize embeddings for the training examples
    lat_vecs = torch.nn.Embedding(N_LATENTS, Args.latent_size)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        # get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
        0.001**2, #1.0,
    )


    # check to see if we have a model checkpoint
    if os.path.exists(os.path.join(Args.output_dir, Args.expt_name, "checkpoints")):
        checkpoint_dict = v3_utils.load_checkpoint(Args.output_dir, Args.expt_name, device=Device, load_best=True)
        NeuralODF.load_state_dict(checkpoint_dict['model_state_dict'])
        NeuralODF = NeuralODF.to(Device)
        lat_vecs.load_state_dict(checkpoint_dict["latent_vectors"])
        lat_vecs = lat_vecs.to(Device)
    else:
        print(f"Unable to extract mesh for model {Args.expt_name} because no checkpoints were found")

    
    data, gt_depth, gt_mask, pred_depth, pred_mask, last_pred_depth  = run_inference(NeuralODF, Device, DataLoader, lat_vecs)
    vis(np.vstack(data), np.vstack(gt_depth), np.vstack(gt_mask), np.vstack(pred_depth), np.vstack(pred_mask), np.vstack(last_pred_depth))

