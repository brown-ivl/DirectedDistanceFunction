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
# from beacon.utils import saveLossesCurve

from data import DepthData
from model import AdaptedLFN, SimpleMLP
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
    avg_loss = float(total_loss/total_batches)
    avg_occ_loss = float(sum_occ_loss/total_batches)
    avg_int_loss = float(sum_int_loss/total_batches)
    avg_depth_loss = float(sum_depth_loss/total_batches)
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Occ Loss: {avg_occ_loss:.4f}")
    print(f"Average Intersect Loss: {avg_int_loss:.4f}")
    print(f"Average Depth Loss: {avg_depth_loss:.4f}")
    return avg_loss, avg_occ_loss, avg_int_loss, avg_depth_loss

def test(model, test_loader, lmbda):
    bce = nn.BCELoss()
    total_loss = 0.
    total_batches = 0.

    all_depth_errors = []
    all_occ_pred = []
    all_occ_label = []
    all_int_pred = []
    all_int_label = []

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
            all_depth_errors.append(torch.abs(depth[torch.logical_and(not_occ,intersect)] - pred_depth[torch.logical_and(not_occ,intersect)]).cpu().numpy())
            all_occ_pred.append(pred_occ.cpu().numpy())
            all_occ_label.append(occ.cpu().numpy())
            all_int_pred.append(pred_int[not_occ].cpu().numpy())
            all_int_label.append(intersect[not_occ].cpu().numpy())
            total_batches+=1.

    print(f"\nAverage Test Loss: {float(total_loss/total_batches):.4f}")
    print("Confusion Matrix Layout:")
    print("[[TN    FP]\n [FN    TP]]")

    print("\nOccupancy-")
    occ_confusion_mat = confusion_matrix(np.hstack(all_occ_label), np.hstack(all_occ_pred)>0.5)
    occ_tn = occ_confusion_mat[0][0]
    occ_fp = occ_confusion_mat[0][1]
    occ_fn = occ_confusion_mat[1][0]
    occ_tp = occ_confusion_mat[1][1]
    occ_precision = occ_tp/(occ_tp + occ_fp)
    occ_recall = occ_tp/(occ_tp + occ_fn)
    occ_accuracy = (occ_tn+occ_tp)/np.sum(occ_confusion_mat)
    print(f"Average Occ Accuracy: {float(occ_accuracy*100):.2f}%")
    print(f"Occ Precision: {occ_precision*100:.2f}%")
    print(f"Occ Recall: {occ_recall*100:.2f}%")
    print(f"Occ F1: {2*(occ_precision*occ_recall)/(occ_precision + occ_recall):.4f}")
    print(occ_confusion_mat)

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

def viz_depth(model, verts, faces):
    '''
    Visualize learned depth map and intersection mask compared to the ground truth
    '''
    cam_center = [-0.8,0.0,-0.8]
    direction = [1.0,0.0,1.0]
    focal_length = 1.5
    sensor_size = [1.0,1.0]
    resolution = [100,100]
    gt_intersection, gt_depth = rasterization.camera_ray_depth(verts, faces, cam_center, direction, focal_length, sensor_size, resolution, near_face_threshold=rasterization.max_edge(verts, faces))
    rays = utils.camera_view_rays(cam_center, direction, focal_length, sensor_size, resolution)
    with torch.no_grad():
        # angle_rays = torch.tensor([list(ray[0]) + list(utils.vector_to_angles(ray[1]-ray[0])) for ray in rays], dtype=torch.float32).to(device)
        angle_rays = torch.tensor([[x for val in list(ray[0])+list((ray[1]-ray[0])/np.linalg.norm(ray[1]-ray[0])) for x in utils.positional_encoding(val)] for ray in rays]).to(device)
        _, intersect, depth = model(angle_rays)
        depth = np.array(torch.reshape(depth.cpu(), tuple(resolution)))
        intersect = np.array(torch.reshape(intersect.cpu() > 0.5, tuple(resolution))).astype(float)
    depth_learned_mask = depth + np.where(intersect, 0., np.inf)
    depth_gt_mask = depth + np.where(gt_intersection, 0., np.inf)
    _, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
    ax1.imshow(gt_intersection)
    ax1.set_title("GT Intersect")
    ax2.imshow(intersect)
    ax2.set_title("Intersect")
    ax3.imshow(gt_depth)
    ax3.set_title("GT Depth")
    ax4.imshow(depth_learned_mask)
    ax4.set_title("Depth - Learned Mask")
    ax5.imshow(depth_gt_mask)
    ax5.set_title("Depth - GT Mask")
    ax6.imshow(depth)
    ax6.set_title("Depth")
    plt.show()


if __name__ == "__main__":
    print(f"Using {device}")
    parser = argparse.ArgumentParser(description="A script to train and evaluate a directed distance function network")

    # CONFIG
    parser.add_argument("--n_workers", type=int, default=1, help="Number of workers for dataloaders. Recommended is 2*num cores")

    # DATA
    parser.add_argument("--samples_per_mesh", type=int, default=1000000, help="Number of rays to sample for each mesh")
    parser.add_argument("--mesh_file", default="/gpfs/data/ssrinath/human-modeling/large_files/sample_data/stanford_bunny.obj", help="Source of mesh to train on")
    # "F:\\ivl-data\\sample_data\\stanford_bunny.obj"

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
    # parser.add_argument("--loss_dir", type=str, default="F:\\ivl-data\\DirectedDF\\large_files\\loss_curves")
    # parser.add_argument("--model_dir", type=str, default="/data/gpfs/ssrinath/human-modeling/large_files/directedDF/model_weights")
    # parser.add_argument("--loss_dir", type=str, default="/data/gpfs/ssrinath/human-modeling/large_files/directedDF/loss_curves")
    parser.add_argument("--save_dir", type=str, default="/gpfs/data/ssrinath/human-modeling/DirectedDF/large_files", help="a directory where model weights, loss curves, and visualizations will be saved")

    args = parser.parse_args()

    # make sure the output directory is setup correctly
    assert(os.path.exists(args.save_dir))
    necessary_subdirs = ["saved_models", "loss_curves"]
    for subdir in necessary_subdirs:
        if not os.path.exists(os.path.join(args.save_dir, subdir)):
            os.mkdir(os.path.join(args.save_dir, subdir))

    model_path = os.path.join(args.save_dir, "saved_models", f"{args.name}.pt")
    loss_path = os.path.join(args.save_dir, "loss_curves", args.name)
    model = AdaptedLFN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # base_path = "C:\\Users\\Trevor\\Brown\\ivl-research\\large_files\\sample_data"
    # instance = "50002_hips_poses_0694"
    # gender = "male"
    # smpl_data_path = os.path.join(base_path, f"{instance}_smpl.npy")
    # faces_path = os.path.join(base_path, f"{gender}_template_mesh_faces.npy")

    # smpl_data = np.load(smpl_data_path, allow_pickle=True).item()
    # verts = np.array(smpl_data["smpl_mesh_v"])
    # faces = np.array(np.load(faces_path, allow_pickle=True))

    mesh = trimesh.load(args.mesh_file)
    faces = mesh.faces
    verts = mesh.vertices
    verts = utils.mesh_normalize(verts)

    sampling_methods = [sampling.sample_uniform_ray_space, sampling.sample_vertex_noise, sampling.sample_vertex_all_directions, sampling.sample_vertex_tangential]
    sampling_frequency = [0.4, 0.0, 0.4, 0.2]
    test_sampling_frequency = [1., 0., 0., 0.]

    train_data = DepthData(faces,verts,args.radius,sampling_methods,sampling_frequency,size=args.samples_per_mesh)
    test_data = DepthData(faces,verts,args.radius,sampling_methods,test_sampling_frequency,size=int(args.samples_per_mesh*0.1))

    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=args.n_workers)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, drop_last=True, pin_memory=True)

    if args.load:
        print("Loading saved model...")
        model.load_state_dict(torch.load(model_path))
    if args.train:
        print(f"Training for {args.epochs} epochs...")
        model=model.train()
        total_loss = []
        occ_loss = []
        int_loss = []
        depth_loss = []
        for e in range(args.epochs):
            print(f"EPOCH {e+1}")
            tl, ol, il, dl = train_epoch(model, train_loader, optimizer, args.lmbda)
            total_loss.append(tl)
            occ_loss.append(ol)
            int_loss.append(il)
            depth_loss.append(dl)
            utils.saveLossesCurve(total_loss, occ_loss, int_loss, depth_loss, legend=["Total", "Occupancy", "Intersection", "Depth"], out_path=loss_path, log=True)
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



