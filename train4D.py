'''
Train a directed sdf network
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import trimesh
import math
# from beacon.utils import saveLossesCurve

from data import DepthData, MultiDepthDataset
from model import LF4D, AdaptedLFN, SimpleMLP
import utils
from camera import Camera, DepthMapViewer, save_video, save_video_4D
import sampling
import rasterization
import meshing_3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

def l2_loss(labels, predictions):
    '''
    L2 loss
    '''
    # print("L2 Loss")
    # print(labels[-10:])
    # print(predictions[-10:])
    return torch.mean(torch.square(labels - predictions))

def chamfer_loss_1d(ground_truth, predictions, gt_mask, pred_mask):
    '''
    A chamfer distance measure between a set of ground truth and predicted depth points
    '''
    ground_truth = ground_truth.unsqueeze(2)
    predictions = predictions.unsqueeze(2)

    # we need to mask out the elements that aren't labeled as true intersections
    extended_gt_mask = gt_mask.unsqueeze(2)
    extended_gt_mask = extended_gt_mask.tile((1,1,pred_mask.shape[1]))
    extended_pred_mask = pred_mask.unsqueeze(1)
    extended_pred_mask = extended_pred_mask.tile((1,gt_mask.shape[1],1))
    joint_mask = torch.logical_and(extended_pred_mask, extended_gt_mask)

    dists = torch.cdist(ground_truth, predictions)
    # this step is solely to allow us to mask out certain values in a differentiable manner
    # dists = 1. / (dists + 0.01)
    # dists[torch.logical_not(joint_mask)] *= -1
    # gt_term = torch.mean(1. / torch.max(dists, dim=2)[0] - 0.01) 
    # pred_term = torch.mean(1. / torch.max(dists, dim=1)[0] - 0.01) 

    masked_dists = torch.where(joint_mask, dists, torch.tensor(np.inf).to(device))

    # find the nearest point in the opposing set (mask out inf values in current set)
    gt_term = torch.min(masked_dists, dim=2)[0]
    gt_term = torch.where(gt_mask, gt_term, torch.tensor(0.).to(device))
    gt_term = torch.sum(gt_term, dim=1) / torch.sum(gt_mask, dim=1)
    gt_term = torch.mean(gt_term)
    # print("GT TERM")
    # print(torch.min(masked_dists, dim=2)[0][gt_mask])

    # pred_term = torch.mean(torch.min(masked_dists, dim=1)[0][pred_mask])
    pred_term = torch.min(masked_dists, dim=1)[0]
    pred_term = torch.where(pred_mask, pred_term, torch.tensor(0.).to(device))
    pred_term = torch.sum(pred_term, dim=1) / torch.sum(pred_mask, dim=1)
    pred_term = torch.mean(pred_term)
    # print("PRED TERM")
    # print(torch.min(masked_dists, dim=1)[0][pred_mask])
    return 0.5 * (gt_term + pred_term)

def intersection_count_loss(ground_truth, predictions):
    # seems like this might zero out the gradients
    return torch.mean(torch.sqrt(torch.square(torch.sum(ground_truth > 0.5, dim=1) - torch.sum(predictions > 0.5, dim=1))))

def push_top_n(gt_int, pred_int):
    '''
    If there are n intersections, labels the top n intersection outputs as the largest
    '''
    n_ints = torch.sum(gt_int, dim=1)
    pred_sorted = torch.sort(pred_int, dim=1)[0]
    sorted_labels = torch.zeros(pred_sorted.shape)
    for i in sorted_labels.shape[0]:
        sorted_labels[i, :n_ints[i]] = 1.
    bce = nn.BCELoss(reduction="mean")
    return bce(pred_sorted, sorted_labels.to(device))


def train_epoch(model, train_loader, optimizer, lmbda, coord_type, unordered=False):
    ce = nn.CrossEntropyLoss(reduction="mean")
    bce = nn.BCELoss(reduction="mean")
    total_loss = 0.
    sum_int_loss = 0.
    sum_depth_loss = 0.
    total_batches = 0
    for batch in tqdm(train_loader):
        coordinates = batch[f"coordinates_{coord_type}"].to(device)
        intersect = batch["intersect"].to(device)
        n_ints = batch["n_ints"].to(device)
        depth = batch["depths"].to(device)
        pred_int, pred_depth = model(coordinates)
        if unordered:
            # mask of rays that have any intersections (gt & predicted)
            gt_any_int_mask = torch.any(intersect > 0.5, dim=1)
            pred_any_int_mask = torch.any(pred_int > 0.5, dim=1)
            combined_int_mask = torch.logical_and(gt_any_int_mask, pred_any_int_mask)
            depth_loss = lmbda * chamfer_loss_1d(depth[combined_int_mask], pred_depth[combined_int_mask], (intersect > 0.5)[combined_int_mask], (pred_int > 0.5)[combined_int_mask])
            intersect_loss = push_top_n(intersect, pred_int)
        else:
            intersect = intersect.reshape((-1,))
            depth = depth.reshape((-1,))
            pred_depth = pred_depth.reshape((-1,))
            n_ints = n_ints.reshape((-1))
            
            # create mask from intersection values
            # depth_mask = torch.nn.functional.one_hot(intersect.to(torch.int64), pred_int.shape[1])
            # depth_mask = torch.cumsum(depth_mask, dim=1)
            # depth_mask = torch.logical_not(depth_mask)
            # depth_mask = depth_mask[:,:-1]
            # depth_mask = depth_mask.reshape((-1))
            
            depth_loss = lmbda * l2_loss(depth[intersect > 0.5], pred_depth[intersect > 0.5])
            intersect_loss = ce(pred_int, n_ints.long())    
        loss = intersect_loss + depth_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_int_loss += intersect_loss.detach()
        sum_depth_loss += depth_loss.detach()
        total_loss += loss.detach()
        total_batches += 1.
    avg_loss = float(total_loss/total_batches)
    avg_int_loss = float(sum_int_loss/total_batches)
    avg_depth_loss = float(sum_depth_loss/total_batches)
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Intersect Loss: {avg_int_loss:.4f}")
    print(f"Average Depth Loss: {avg_depth_loss:.4f}")
    return avg_loss, avg_int_loss, avg_depth_loss

def test(model, test_loader, lmbda, coord_type, unordered=False):
    ce = nn.CrossEntropyLoss(reduction="mean")
    bce = nn.BCELoss(reduction="mean")
    total_loss = 0.
    total_batches = 0.
    total_chamfer = 0.

    all_depth_errors = []
    all_int_pred = []
    all_int_label = []

    int_tn = 0.
    int_fp = 0.
    int_fn = 0.
    int_tp = 0.


    with torch.no_grad():
        for batch in tqdm(test_loader):
            coordinates = batch[f"coordinates_{coord_type}"].to(device)
            intersect = batch["intersect"].to(device)
            n_ints = batch["n_ints"].to(device)
            depth = batch["depths"].to(device)
            pred_int, pred_depth = model(coordinates)
            if unordered:
                # mask of rays that have any intersections (gt & predicted)
                gt_any_int_mask = torch.any(intersect > 0.5, dim=1)
                pred_any_int_mask = torch.any(pred_int > 0.5, dim=1)
                combined_int_mask = torch.logical_and(gt_any_int_mask, pred_any_int_mask)
                depth_loss = lmbda * chamfer_loss_1d(depth[combined_int_mask], pred_depth[combined_int_mask], (intersect > 0.5)[combined_int_mask], (pred_int > 0.5)[combined_int_mask])
                intersect_loss = push_top_n(intersect, pred_int)
            else:
                intersect = intersect.reshape((-1,))
                depth = depth.reshape((-1,))
                pred_depth = pred_depth.reshape((-1,))
                n_ints = n_ints.reshape((-1))
                
                depth_loss = lmbda * l2_loss(depth[intersect > 0.5], pred_depth[intersect > 0.5])
                intersect_loss = ce(pred_int, n_ints.long())

                # this is all to compute the confusion matrix 
                pred_n_ints = torch.argmax(pred_int, dim=1)
                # create binary labels for each intersection
                pred_int_mask = torch.nn.functional.one_hot(pred_n_ints.to(torch.int64), pred_int.shape[1])
                pred_int_mask = torch.cumsum(pred_int_mask, dim=1)
                pred_int_mask = torch.logical_not(pred_int_mask)
                pred_int_mask = pred_int_mask[:,:-1]
                pred_int_mask = pred_int_mask.reshape((-1)).cpu().numpy()

                all_int_pred.append(pred_int_mask)

            loss = intersect_loss + depth_loss
            all_depth_errors.append(torch.abs(depth[intersect > 0.5] - pred_depth[intersect > 0.5]).cpu().numpy())
            # all_int_pred.append(pred_int.cpu().numpy().flatten())
            all_int_label.append(intersect.cpu().numpy().flatten())
            if unordered:
                total_chamfer += depth_loss / lmbda
            total_loss += loss.detach()
            total_batches += 1.

    print(f"\nAverage Test Loss: {float(total_loss/total_batches):.4f}")
    if unordered:
        print(f"Average Chamfer Loss: {(total_chamfer / total_batches):.4f}")
    print("Confusion Matrix Layout:")
    print("[[TN    FP]\n [FN    TP]]")

    print("\nIntersection-")
    int_confusion_mat = confusion_matrix(np.hstack(all_int_label), np.hstack(all_int_pred)>0.5)
    int_tn = int_confusion_mat[0][0]
    int_fp = int_confusion_mat[0][1]
    int_fn = int_confusion_mat[1][0]
    int_tp = int_confusion_mat[1][1]
    

    int_precision = int_tp/(int_tp + int_fp)
    int_recall = int_tp/(int_tp + int_fn)
    int_accuracy = (int_tn + int_tp)/np.sum(int_confusion_mat)
    print(f"Average Intersect Accuracy: {float(int_accuracy*100):.2f}%")
    print(f"Intersect Precision: {int_precision*100:.2f}%")
    print(f"Intersect Recall: {int_recall*100:.2f}%")
    print(f"Intersect F1: {2*(int_precision*int_recall)/(int_precision + int_recall):.4f}")
    print(int_confusion_mat)

    print("\nDepth-")
    all_depth_errors = np.hstack(all_depth_errors)
    print(f"Average Depth Error: {np.mean(all_depth_errors):.4f}")
    print(f"Median Depth Error: {np.median(all_depth_errors):.4f}\n")

def viz_depth(model, verts, faces, radius, show_rays=False):
    '''
    Visualize learned depth map and intersection mask compared to the ground truth
    TODO: add depth map legend
    '''
    # these are the normalization bounds for coloring in the video
    # vmin = radius - 1.
    # vmax = radius + 1.

    fl = 1.0
    sensor_size = [1.0,1.0]
    resolution = [100,100]
    zoom_out_cameras = [Camera(center=[1.25 + 0.2*x,0.0,0.0], direction=[-1.0,0.0,0.0], focal_length=fl, sensor_size=sensor_size, sensor_resolution=resolution) for x in range(4)]
    data = [cam.mesh_and_model_depthmap(model, verts, faces, radius, show_rays=show_rays, fourd=True) for cam in zoom_out_cameras]
    vmin = [min(np.min(mesh_depths[mesh_n_ints > 0.5]) if np.any(mesh_n_ints > 0.5) else np.inf, np.min(model_depths[model_n_ints > 0.5]) if np.any(model_n_ints > 0.5) else np.inf) for mesh_n_ints, mesh_depths, model_n_ints, model_depths in data]
    vmax = [max(np.max(mesh_depths[mesh_n_ints > 0.5]) if np.any(mesh_n_ints > 0.5) else -np.inf, np.max(model_depths[model_n_ints > 0.5]) if np.any(model_n_ints > 0.5) else -np.inf) for mesh_n_ints, mesh_depths, model_n_ints, model_depths in data]
    vmin = [vmin[i] if vmin[i] < np.inf else np.min(data[i][3]) for i in range(len(vmin))]
    vmax = [vmax[i] if vmax[i] > -np.inf else np.max(data[i][3]) for i in range(len(vmax))]
    DepthMapViewer(data, vmin, vmax, fourd=True)

def equatorial_video(model, verts, faces, radius, n_frames, resolution, save_dir, name):
    '''
    Saves a rendered depth video from around the equator of the object
    '''
    video_dir = os.path.join(save_dir, "depth_videos")
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)

    # these are the normalization bounds for coloring in the video
    vmin = radius - 1.
    vmax = radius + 1.

    fl = 1.0
    sensor_size = [1.0,1.0]
    resolution = [resolution,resolution]
    angle_increment = 2*math.pi / n_frames
    z_vals = [np.cos(angle_increment*i)*radius for i in range(n_frames)]
    x_vals = [np.sin(angle_increment*i)*radius for i in range(n_frames)]
    circle_cameras = [Camera(center=[x_vals[i],0.0,z_vals[i]], direction=[-x_vals[i],0.0,-z_vals[i]], focal_length=fl, sensor_size=sensor_size, sensor_resolution=resolution, verbose=False) for i in range(n_frames)]
    rendered_views = [cam.mesh_and_model_depthmap(model, verts, faces, radius, fourd=True) for cam in tqdm(circle_cameras)]

    save_video_4D(rendered_views, os.path.join(video_dir, f'4D_equatorial_{name}_rad{radius*100:.0f}.mp4'), vmin, vmax)

def generate_point_cloud(model, sphere_vertices, vertices, faces, focal_point=[0., 0., 0.], show=True):
    '''
    Returns the 1st, 2nd, 3rd, and 4th+ intersection point clouds produced by shooting rays from the 
    sphere vertices towards the focal point
    '''
    focal_point = np.array(focal_point)
    # pointclouds = [[],[],[],[]]
    ray_directions = [(focal_point-v) / np.linalg.norm(focal_point-v) for v in sphere_vertices]
    with torch.no_grad():
        # pass in surface point, direction
        _, depths, n_ints = model.query_rays(torch.tensor(sphere_vertices, dtype=torch.float32), torch.tensor(ray_directions, dtype=torch.float32))
    
    n_ints = n_ints.cpu()
    model_depths = depths.cpu()
    model_depths = torch.min(model_depths, dim=1)[0]
    model_depths = model_depths.numpy()
    new_points = [sphere_vertices[i] + ray_directions[i]*model_depths[i] if model_depths[i] < np.inf else None for i in range(len(sphere_vertices))]

    if show:
        # can't import visualization on OSCAR because it uses Open3D and OpenGL
        import visualization
        lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)
        visualizer = visualization.RayVisualizer(vertices, lines)
        for point in new_points:
            if point is not None:
                visualizer.add_point(point, [52./255., 88./255., 235./255.])
        visualizer.display()
    # TODO: save to file
    return new_points

def generate_simple_mesh(model, sphere_vertices, sphere_faces, focal_point=[0.,0.,0.], show=True):
    '''
    Returns a mesh produced by shooting rays from the sphere vertices towards the focal point
    '''
    focal_point = np.array(focal_point)
    ray_directions = [(focal_point-v) / np.linalg.norm(focal_point-v) for v in sphere_vertices]
    with torch.no_grad():
        # pass in surface point, direction
        _, depths, n_ints = model.query_rays(torch.tensor(sphere_vertices, dtype=torch.float32), torch.tensor(ray_directions, dtype=torch.float32))

    n_ints = n_ints.cpu()
    model_depths = depths.cpu()
    model_depths = torch.min(model_depths, dim=1)[0]
    model_depths = model_depths.numpy()
    new_points = [sphere_vertices[i] + ray_directions[i]*model_depths[i] if model_depths[i] < np.inf else sphere_vertices[i] for i in range(len(sphere_vertices))]

    if show:
        # can't import visualization on OSCAR because it uses Open3D and OpenGL
        import visualization
        import open3d as o3d
        o3d.visualization.draw_geometries([visualization.make_mesh(np.array(new_points), sphere_faces)])
    # TODO: save to file
    return new_points

if __name__ == "__main__":
    print(f"Using {device}")
    parser = argparse.ArgumentParser(description="A script to train and evaluate a directed distance function network")

    # CONFIG
    parser.add_argument("--n_workers", type=int, default=0, help="Number of workers for dataloaders. Recommended is 2*num cores")
    parser.add_argument("--save_dir", type=str, default="/gpfs/data/ssrinath/human-modeling/DirectedDF/large_files/", help="a directory where model weights, loss curves, and visualizations will be saved")
    parser.add_argument("-n", "--name", type=str, required=True, help="The name of the model")

    # DATA
    parser.add_argument("--samples_per_mesh", type=int, default=1000000, help="Number of rays to sample for each mesh")
    parser.add_argument("--mesh_file", default="/gpfs/data/ssrinath/human-modeling/large_files/sample_data/stanford_bunny.obj", help="Source of mesh to train on")
    # NOTE: Double check LF4D and Camera class if coord type/ pos enc change
    parser.add_argument("--coord_type", default="direction", help="Type of coordinates to use, valid options are 'points' | 'direction' | 'pluecker' ")
    parser.add_argument("--pos_enc", default=True, type=bool, help="Whether NeRF-style positional encoding should be applied to the data")
    parser.add_argument("--vert_noise", type=float, default=0.02, help="Standard deviation of noise to add to vertex sampling methods")
    parser.add_argument("--tan_noise", type=float, default=0.02, help="Standard deviation of noise to add to tangent sampling method")
    parser.add_argument("--uniform", type=int, default=100, help="What percentage of the data should be uniformly sampled (0 -> 0%, 100 -> 100%)")
    parser.add_argument("--vertex", type=int, default=0, help="What percentage of the data should use vertex sampling (0 -> 0%, 100 -> 100%)")
    parser.add_argument("--tangent", type=int, default=0, help="What percentage of the data should use vertex tangent sampling (0 -> 0%, 100 -> 100%)")
    # "F:\\ivl-data\\sample_data\\stanford_bunny.obj"

    # MODEL
    parser.add_argument("--lmbda", type=float, default=100., help="Multiplier for depth l2 loss")
    parser.add_argument("--intersect_limit", type=int, default=20, help="Max number of intersections that the network will predict per ray (should be even number)")
    parser.add_argument("--unordered", action="store_true", help="The intersection outputs will have no ordering constraint if this argument is passed")


    # HYPERPARAMETERS
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=1000, help="Train batch size")
    parser.add_argument("--test_batch_size", type=int, default=1000, help="Test batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train (overrides --iterations)")
    parser.add_argument("--radius", type=float, default=1.25, help="The radius at which all rays start and end (mesh is normalized to be in unit sphere)")

    # ACTIONS
    parser.add_argument("-T", "--train", action="store_true", help="Train the network")
    parser.add_argument("-t", "--test", action="store_true", help="Test the network")
    parser.add_argument("-s", "--save", action="store_true", help="Save the trained network")
    parser.add_argument("-l", "--load", action="store_true", help="Load the model from file")
    parser.add_argument("-d", "--viz_depth", action="store_true", help="Visualize the learned depth map and intersection mask versus the ground truth")
    parser.add_argument("-v", "--video", action="store_true", help="Render a video of the learned mask and depth map compared to the ground truth")
    parser.add_argument("-p", "--pointcloud", action="store_true", help="Generate a point cloud of the object based on the learned ODF")
    parser.add_argument("-m", "--mesh", action="store_true", help="Visualize a mesh generated from rays starting on a sphere surface looking inwards")


    # VISUALIZATION
    parser.add_argument("--show_rays", action="store_true", help="Visualize the camera's rays relative to the scene when rendering depthmaps")
    parser.add_argument("--n_frames", type=int, default=200, help="Number of frames to render if saving video")
    parser.add_argument("--video_resolution", type=int, default=250, help="The height and width of the rendered video (in pixels)")

    args = parser.parse_args()

    # make sure the output directory is setup correctly
    assert(os.path.exists(args.save_dir))
    necessary_subdirs = ["saved_models", "loss_curves"]
    for subdir in necessary_subdirs:
        if not os.path.exists(os.path.join(args.save_dir, subdir)):
            os.mkdir(os.path.join(args.save_dir, subdir))

    model_path = os.path.join(args.save_dir, "saved_models", f"{args.name}.pt")
    loss_path = os.path.join(args.save_dir, "loss_curves", args.name)
    model = LF4D(input_size=(120 if args.pos_enc else 6), n_intersections=args.intersect_limit, radius=args.radius, coord_type=args.coord_type, pos_enc=args.pos_enc).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mesh = trimesh.load(args.mesh_file)
    faces = mesh.faces
    verts = mesh.vertices
    verts = utils.mesh_normalize(verts)

    sampling_methods = [sampling.sample_uniform_4D, 
                        sampling.sampling_preset_noise(sampling.sample_vertex_4D, args.vert_noise),
                        sampling.sampling_preset_noise(sampling.sample_tangential_4D, args.tan_noise)]
    sampling_frequency = [0.01 * args.uniform, 0.01 * args.vertex, 0.01*args.tangent]
    assert(sum(sampling_frequency) == 1.0)
    test_sampling_frequency = [1., 0., 0.]

    train_data = MultiDepthDataset(faces, verts, args.radius, sampling_methods, sampling_frequency, size=args.samples_per_mesh, intersect_limit=args.intersect_limit, pos_enc=args.pos_enc)
    test_data = MultiDepthDataset(faces,verts,args.radius, sampling_methods, sampling_frequency, size=int(args.samples_per_mesh*0.1), intersect_limit=args.intersect_limit, pos_enc=args.pos_enc)

    # TODO: num_workers=args.n_workers
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=args.n_workers)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=args.n_workers)

    if args.load:
        print("Loading saved model...")
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    if args.train:
        print(f"Training for {args.epochs} epochs...")
        model=model.train()
        total_loss = []
        int_loss = []
        depth_loss = []
        for e in range(args.epochs):
            print(f"EPOCH {e+1}")
            tl, il, dl = train_epoch(model, train_loader, optimizer, args.lmbda, args.coord_type, unordered=args.unordered)
            total_loss.append(tl)
            int_loss.append(il)
            depth_loss.append(dl)
            utils.saveLossesCurve(total_loss, int_loss, depth_loss, legend=["Total", "Intersection", "Depth"], out_path=loss_path, log=True)
            if args.save:
                print("Saving model...")
                torch.save(model.state_dict(), model_path)
    if args.test:
        print("Testing model ...")
        model=model.eval()
        test(model, test_loader, args.lmbda, args.coord_type, unordered=args.unordered)
    if args.viz_depth:
        print("Visualizing depth map...")
        model=model.eval()
        viz_depth(model, verts, faces, args.radius, args.show_rays)
    if args.pointcloud:
        model = model.eval()
        sphere_vertices, _ = meshing_3d.icosahedron_sphere_tessalation(args.radius, subdivisions=4)
        generate_point_cloud(model, sphere_vertices, verts, faces)
    if args.mesh:
        model = model.eval()
        meshing_3d.make_model_mesh(model, initial_tessalation_factor=3, radius=args.radius, focal_point=[0.,0.,0.])
        # sphere_vertices, sphere_faces = meshing_3d.icosahedron_sphere_tessalation(args.radius, subdivisions=4)
        # generate_simple_mesh(model, sphere_vertices, sphere_faces)
    if args.video:
        print(f"Rendering ({args.video_resolution}x{args.video_resolution}) video with {args.n_frames} frames...")
        model=model.eval()
        equatorial_video(model, verts, faces, args.radius, args.n_frames, args.video_resolution, args.save_dir, args.name)
    # print name again so it's at the bottom of the slurm output
    print(f"{args.name} finished")



