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
import matplotlib.pyplot as plt
import trimesh

from data import DepthData
from model import SimpleMLP
import utils
import sampling
import rasterization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def l2_loss(labels, predictions):
    '''
    L2 loss
    '''
    return torch.mean(torch.square(labels - predictions))

def train_epoch(model, train_loader, optimizer, lmbda):
    bce = nn.BCELoss()
    total_loss = 0.
    sum_occ_loss = 0.
    sum_int_loss = 0.
    sum_depth_loss = 0.
    total_batches = 0
    for batch in tqdm(train_loader):
        coordinates = batch["coordinates"].to(device)
        # print(torch.max(coordinates))
        occ = batch["occ"].to(device).reshape((-1,))
        not_occ = torch.logical_not(occ)
        intersect = batch["intersect"].to(device).reshape((-1,))
        depth = batch["depth"].to(device).reshape((-1,))
        pred_occ, pred_int, pred_depth = model(coordinates)
        pred_occ = pred_occ.reshape((-1,))
        pred_int = pred_int.reshape((-1,))
        pred_depth = pred_depth.reshape((-1,))
        # print("pred occ")
        # print(torch.max(pred_occ))
        # print(torch.min(pred_occ))
        # print('pred int')
        # print(torch.max(pred_int))
        # print(torch.min(pred_int))
        occ_loss = bce(pred_occ, occ)
        sum_occ_loss += occ_loss.detach()
        # print(occ_loss.detach())
        intersect_loss = bce(pred_int[not_occ], intersect[not_occ])
        sum_int_loss += intersect_loss.detach()
        # print(intersect_loss.detach())
        depth_loss = lmbda * l2_loss(depth[torch.logical_and(not_occ,intersect)], pred_depth[torch.logical_and(not_occ,intersect)])
        sum_depth_loss += depth_loss.detach()
        # print(depth_loss.detach())
        loss = occ_loss + intersect_loss + depth_loss
        # print(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()
        total_batches += 1
    print(f"Average Loss: {float(total_loss/total_batches):.4f}")
    print(f"Average Occ Loss: {float(sum_occ_loss/total_batches):.4f}")
    print(f"Average Intersect Loss: {float(sum_int_loss/total_batches):.4f}")
    print(f"Average Depth Loss: {float(sum_depth_loss/total_batches):.4f}")

def test(model, test_loader, lmbda):
    bce = nn.BCELoss()
    total_batches = 0.
    total_loss = 0.
    depth_error = 0.
    occ_accuracy = 0.
    int_accuracy = 0.

    with torch.no_grad():
        for batch in tqdm(test_loader):
            coordinates = batch["coordinates"].to(device)
            occ = batch["occ"].to(device).reshape((-1,))
            not_occ = torch.logical_not(occ)
            intersect = batch["intersect"].to(device).reshape((-1,))
            depth = batch["depth"].to(device).reshape((-1,))
            pred_occ, pred_int, pred_depth = model(coordinates)
            pred_occ = pred_occ.reshape((-1,))
            pred_int = pred_int.reshape((-1,))
            pred_depth = pred_depth.reshape((-1,))
            occ_loss = bce(pred_occ, occ)
            intersect_loss = bce(pred_int[not_occ], intersect[not_occ])
            depth_loss = lmbda * l2_loss(depth[torch.logical_and(not_occ,intersect)], pred_depth[torch.logical_and(not_occ,intersect)])
            loss = occ_loss + intersect_loss + depth_loss
            total_loss += loss
            total_batches += 1
            depth_error += torch.mean(torch.abs(depth[torch.logical_and(not_occ,intersect)] - pred_depth[torch.logical_and(not_occ,intersect)]))
            occ_accuracy += torch.mean((occ == (pred_occ>0.5)).double())
            int_accuracy += torch.mean((intersect[not_occ] == (pred_int[not_occ]>0.5)).double())
    print(f"Average Test Loss: {float(total_loss/total_batches):.4f}")
    print(f"Average Occ Accuracy: {float(occ_accuracy/total_batches*100):.2f}%")
    print(f"Average Intersect Accuracy: {float(int_accuracy/total_batches*100):.2f}%")
    print(f"Average Depth Error: {float(depth_error/total_batches):.2f}")

def viz_depth(model, verts, faces):
    '''
    Visualize learned depth map and intersection mask compared to the ground truth
    '''
    cam_center = [-1.0,0.0,-1.0]
    direction = [1.0,0.0,1.0]
    focal_length = 1.5
    sensor_size = [1.0,1.0]
    resolution = [100,100]
    gt_intersection, gt_depth = rasterization.camera_ray_depth(verts, faces, cam_center, direction, focal_length, sensor_size, resolution, near_face_threshold=rasterization.max_edge(verts, faces))
    rays = utils.camera_view_rays(cam_center, direction, focal_length, sensor_size, resolution)
    with torch.no_grad():
        angle_rays = torch.tensor([list(ray[0]) + list(utils.vector_to_angles(ray[1]-ray[0])) for ray in rays], dtype=torch.float32).to(device)
        print(angle_rays.shape)
        _, intersect, depth = model(angle_rays)
        depth = np.array(torch.reshape(depth.cpu(), tuple(resolution)))
        intersect = np.array(torch.reshape(intersect.cpu() > 0.5, tuple(resolution))).astype(np.float)
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax1.imshow(gt_intersection)
    ax1.set_title("GT Intersect")
    ax2.imshow(intersect)
    ax2.set_title("Intersect")
    ax3.imshow(gt_depth)
    ax3.set_title("GT Depth")
    ax4.imshow(depth)
    ax4.set_title("Depth")
    plt.show()


if __name__ == "__main__":
    print(f"Using {device}")
    parser = argparse.ArgumentParser(description="A script to train and evaluate a directed distance function network")

    # CONFIG
    parser.add_argument("--n_workers", type=int, default=1, help="Number of workers for dataloaders. Recommended is 2*num cores")

    # DATA
    parser.add_argument("--samples_per_mesh", type=int, default=1000000, help="Number of rays to sample for each mesh")

    # MODEL
    parser.add_argument("--lmbda", type=float, default=100., help="Multiplier for depth l2 loss")


    # HYPERPARAMETERS
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=1000, help="Train batch size")
    parser.add_argument("--test_batch_size", type=int, default=1000, help="Test batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train (overrides --iterations)")
    parser.add_argument("--iterations", type=int, default=200000, help="Number of iterations to train (NASA says 200k)")
    parser.add_argument("--radius", type=float, default=1.25, help="The radius within which all rays should orginiate (mesh is normalized to be in unit sphere")

    # ACTIONS
    parser.add_argument("-T", "--train", action="store_true", help="Train the network")
    parser.add_argument("-t", "--test", action="store_true", help="Test the network")
    parser.add_argument("-s", "--save", action="store_true", help="Save the trained network")
    parser.add_argument("-l", "--load", action="store_true", help="Load the model from file")
    parser.add_argument("-d", "--viz_depth", action="store_true", help="Visualize the learned depth map and intersection mask versus the ground truth")
    parser.add_argument("-n", "--name", type=str, required=True, help="The name of the model")
    # parser.add_argument("--model_dir", type=str, default="F:\\ivl-data\\DirectedDF\\large_files\\models")
    parser.add_argument("--model_dir", type=str, default="/gpfs/data/ssrinath/human-modeling/large_files/ddf_models")
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, f"{args.name}.pt")
    model = SimpleMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # base_path = "C:\\Users\\Trevor\\Brown\\ivl-research\\large_files\\sample_data"
    # instance = "50002_hips_poses_0694"
    # gender = "male"
    # smpl_data_path = os.path.join(base_path, f"{instance}_smpl.npy")
    # faces_path = os.path.join(base_path, f"{gender}_template_mesh_faces.npy")

    # smpl_data = np.load(smpl_data_path, allow_pickle=True).item()
    # verts = np.array(smpl_data["smpl_mesh_v"])
    # faces = np.array(np.load(faces_path, allow_pickle=True))

    mesh_path = "/gpfs/data/ssrinath/human-modeling/large_files/sample_data/stanford_bunny.obj"
    mesh = trimesh.load(mesh_path)
    faces = mesh.faces
    verts = mesh.vertices
    verts = utils.mesh_normalize(verts)

    sampling_methods = [sampling.sample_uniform_ray_space, sampling.sample_vertex_noise, sampling.sample_vertex_all_directions, sampling.sample_vertex_tangential]
    sampling_frequency = [1.0, 0.0, 0.0, 0.0]

    train_data = DepthData(faces,verts,args.radius,sampling_methods,sampling_frequency,size=args.samples_per_mesh)
    test_data = DepthData(faces,verts,args.radius,sampling_methods,sampling_frequency,size=int(args.samples_per_mesh*0.1))

    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=args.n_workers)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, drop_last=True, pin_memory=True)

    if args.load:
        print("Loading saved model...")
        model.load_state_dict(torch.load(model_path))
    if args.train:
        print(f"Training for {args.epochs} epochs...")
        model=model.train()
        for e in range(args.epochs):
            print(f"EPOCH {e+1}")
            train_epoch(model, train_loader, optimizer, args.lmbda)
            if args.save:
                print("Saving model...")
                torch.save(model.state_dict(), model_path)
    if args.test:
        print("Testing model ...")
        model=model.eval()
        test(model, test_loader, args.lmbda)
    if args.viz_depth:
        print("Visualizing depth map...")
        model=model.eval()
        viz_depth(model, verts, faces)



