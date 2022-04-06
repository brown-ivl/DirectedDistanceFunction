import sys
import os
import numpy as np
import torch
import open3d as o3d

FileDirPath = os.path.abspath(__file__)
FileDirPath = os.path.dirname(FileDirPath)
FileDirPath = os.path.dirname(FileDirPath)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from depth_sampler_5d import DEPTH_SAMPLER_RADIUS
from odf_models import ODFSingleV3, ODFSingleV3Constant, ODFSingleV3SH, ODFSingleV3ConstantSH, ODFV5, IntersectionMask3D
from occnet_loader import OccNetLoader as ONL
import v5_utils

def make_point_cloud(points, colors=None):
    '''
    Returns an open3d point cloud given a list of points and optional colors
    '''
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    return point_cloud

def show_data(TrainData, interior_only=False):
    positive_points = []
    negative_points = []
    _,_,mask_points, mask_labels = TrainData[0]
    mask_labels = mask_labels.view(-1)
    positive_points.append(mask_points[mask_labels > 0.5])
    negative_points.append(mask_points[mask_labels < 0.5])
    positive_points = np.asarray(torch.cat(positive_points, dim=0))
    negative_points = np.asarray(torch.cat(negative_points, dim=0))
    positive_point_cloud = make_point_cloud(positive_points, colors=[np.array([0.2,0.2,1.0]),]*len(positive_points))
    negative_point_cloud = make_point_cloud(negative_points, colors=[np.array([1.0,0.4,0.1]),]*len(negative_points))
    if not interior_only:
        o3d.visualization.draw_geometries([positive_point_cloud, negative_point_cloud])
    else:
        o3d.visualization.draw_geometries([positive_point_cloud])


if __name__ == "__main__":
    Parser = v5_utils.BaselineParser
    Parser.add_argument('--n-samples', help='Number of samples to show.', default=1000, type=int)
    Parser.add_argument('--interior-only', action="store_true", help="Only show the interior points")
    Args, _ = Parser.parse_known_args()
    
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    v5_utils.seedRandom(Args.seed)
    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TrainData = ONL(root=Args.input_dir, name=Args.dataset, train=True, download=False, target_samples=Args.n_samples)
    show_data(TrainData, interior_only=Args.interior_only)
