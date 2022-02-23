# from beacon import utils as butils
import numpy as np
import torch
import os
import argparse
import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from tqdm import tqdm
import math

MASK_THRESH = 0.995
CONSTANT_MASK_THRESH = 0.5
MAX_RADIUS = 1.25

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from single_models import ODFSingleV3, ODFSingleV3Constant
from depth_sampler_5d import DEPTH_SAMPLER_RADIUS


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True)
    return points_grad

def grid_positions(resolution, bounds=MAX_RADIUS):
    width = 2.*bounds
    step_size = width/resolution
    linear_coords = np.array([-bounds + (i+0.5)*step_size for i in range(resolution)])
    xs, zs = np.meshgrid(linear_coords, np.flip(linear_coords))
    xs = xs.flatten()
    zs = zs.flatten()
    ys = np.zeros(xs.shape)
    coords = np.stack([xs,ys,zs], axis=-1)
    return torch.tensor(coords, dtype=torch.float)

def get_depth_cmap(vmin, vmax):
    #middle is the 0-1 value that 0.0 maps to after normalization
    middle = (0.-vmin)/(vmax-vmin)
    cdict = {
        'red': [[0.0, 217./255., 217./255.],
                [middle, 232./255, 173./255.],
                [1.0, 2./255., 2./255.]],
        'green': [[0.0, 114./255., 114./255.],
                [middle, 196./255., 196./255.],
                [1.0, 10./255., 10./255.]],
        'blue': [[0.0, 4./255., 4./255.],
                [middle, 158./255., 247./255.],
                [1.0, 94./255., 94./255.]],
    }

    depth_cmap = matplotlib.colors.LinearSegmentedColormap('depth_cmap', segmentdata=cdict, N=256)

    return depth_cmap

def save_video_frames(model, direction=[1.,0.,0.], resolution=256, device="cpu", multiview=False):
    model.eval()
    direction = np.array(direction) / np.linalg.norm(direction)
    coords = grid_positions(resolution)
    coords = np.concatenate([coords, np.repeat(direction[None,:], coords.shape[0], axis=0)], axis=1)
    # gt_mask, gt_depths = torus_depth(coords[...,:2], coords[...,2:])
    # gt_depths = np.ma.masked_where(np.logical_not(gt_mask > MASK_THRESH), gt_depths)
    # gt_depths = gt_depths.reshape((resolution, resolution))
    gt_mask, gt_depths = np.zeros((resolution, resolution)), np.zeros((resolution, resolution))
    torch_coords = torch.tensor(coords).float().to(device)
    sigmoid = torch.nn.Sigmoid()

    # TODO: batch this part too??
    torch_coords = torch_coords.to(device)
    model = model.to(device)
    output = model([torch_coords], {})[0]
    if len(output) == 2:
        masks, depths = output
        depths = depths.detach().cpu().numpy().flatten()
    else:
        masks, depths, constant_mask, constants = output
        constant_mask = sigmoid(constant_mask)
        # constant_mask = constant_mask > CONSTANT_MASK_THRESH
        constant_mask = constant_mask.detach().cpu().numpy().flatten()
        constants = constants.detach().cpu().numpy().flatten()
        depths = depths.detach().cpu().numpy().flatten()
        original_depths = np.copy(depths)
        depths += constant_mask*constants
    masks = sigmoid(masks)
    masks = masks.detach().cpu().numpy().flatten()
    masks = masks > MASK_THRESH
    # np.save("F:\\sample_frame.npy", (masks*depths).reshape((resolution, resolution)))
    depths = np.ma.masked_where(np.logical_not(masks), depths)

    gradients = []
    grad_masks = []
    batch_size = 10000
    for i in range(0, coords.shape[0], batch_size):
        batch_coords = torch_coords[i:i+batch_size,...]
        batch_coords.requires_grad_()
        output=model([batch_coords], {})[0]
        intersections, grad_depths = output[:2]
        intersections = sigmoid(intersections)
        intersections = intersections.detach().cpu().numpy().flatten()
        grad_depths = grad_depths.flatten()
        x_grads = gradient(batch_coords, grad_depths)[0][...,:3].detach().cpu().numpy()
        directional_gradients = np.sum(x_grads * coords[i:i+batch_size,3:], axis=-1)
        gradients.append(directional_gradients)
        grad_masks.append(intersections > MASK_THRESH)
    gradients = np.concatenate(gradients)
    grad_masks = np.concatenate(grad_masks)
    gradients = gradients * grad_masks
    gradients = gradients.reshape((resolution, resolution))

    vid_dir = "video" if not multiview else "multiview_video"
    if not os.path.exists(os.path.join(save_dir, name, vid_dir)):
        os.mkdir(os.path.join(save_dir, name, vid_dir))
    if not os.path.exists(os.path.join(save_dir, name, vid_dir, "frames")):
        os.mkdir(os.path.join(save_dir, name, vid_dir, "frames"))
    all_frame_files = (glob.glob(os.path.join(save_dir, name, vid_dir, "frames", '*.npy')))
    epoch = len(all_frame_files)+1

    frame_file = os.path.join(save_dir, name, vid_dir, "frames", f"{name}_video_frame_{epoch:03}.npy")
    frames_data = {}
    frames_data["odf"] = depths.reshape((resolution, resolution))
    frames_data["gradients"] = gradients
    frames_data["ground_truth"] = gt_depths
    frames_data["mask"] = masks.reshape((resolution, resolution))
    if len(output) > 2:
        frames_data["constants"] = constants.reshape((resolution, resolution))
        frames_data["constant_mask"] = constant_mask.reshape((resolution, resolution))
        frames_data["original_depths"] = original_depths.reshape((resolution, resolution))
    np.save(frame_file, frames_data)

def render_video(name, save_dir, multiview=False):
    print("Rendering Video...")
    vid_dir = "video" if not multiview else "multiview_video"
    frames_dir = os.path.join(save_dir, name, vid_dir, "frames")
    video_file = os.path.join(save_dir, name, vid_dir, f"{name}_video.mp4")
    if not os.path.exists(frames_dir):
        print(f"Video frame direction doesn't exist: {frames_dir}")

    all_frame_files = (glob.glob(os.path.join(save_dir, name, vid_dir, "frames", '*.npy')))
    n_frames = len(all_frame_files)
    all_frame_files.sort()

    if n_frames == 0:
        print("Can't render video because no video frames have been rendered to disk")

    odf_frames = []
    mask_frames = []
    gradient_frames = []
    ground_truth_frames = []
    constant_frames = []
    constant_mask_frames = []
    baseline_depth_frames = []
    for frame_file in all_frame_files:
        # frame_file = os.path.join(save_dir, name, vid_dir, "frames", f"{name}_video_frame_{(i+1):03}.npy")
        frame_data = np.load(frame_file, allow_pickle=True).item()
        odf_frames.append(frame_data["odf"])
        gradient_frames.append(frame_data["gradients"])
        mask_frames.append(frame_data["mask"])
        ground_truth_frames.append(frame_data["ground_truth"])
        if "constants" in frame_data:
            constant_frames.append(frame_data["constants"])
            constant_mask_frames.append(frame_data["constant_mask"])
            baseline_depth_frames.append(frame_data["original_depths"])
    
    min_grad, max_grad = -5.,5.
    gradient_frames = [np.clip(frame, min_grad, max_grad) for frame in gradient_frames]
    
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation

    # frames_data = np.load(frames_file, allow_pickle=True).item()
    # odf_frames = frames_data["odf"]
    # gradient_frames = frames_data["gradients"]


    # all_frame_files = (glob.glob(os.path.join(save_dir, name, vid_dir, "frames", '*.npy')))
    # epoch = len(all_frame_files)+1

    # frame_file = os.path.join(save_dir, name, vid_dir, "frames", f"{name}_video_frame_{epoch:03}.npy")



    if len(constant_frames) > 0:
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)
        f.set_size_inches(18.,12.)
        all_axes = [ax1,ax2,ax3,ax4,ax5,ax6]
    else:
        f, ((ax1,ax3), (ax4, ax2)) = plt.subplots(2,2)
        f.set_size_inches(12.,12.)
        all_axes = [ax1,ax2,ax3,ax4]
    # f, ((ax1, ax2)) = plt.subplots(1,2)
    vmin = -1.*MAX_RADIUS
    vmax = 1.*MAX_RADIUS
    cmap=get_depth_cmap(vmin, vmax)


    # display first view
    # gt_intersect, gt_depth, learned_intersect, learned_depth = rendered_views[0]
    # odf_utils.show_depth_data(gt_intersect, gt_depth, learned_intersect, learned_depth, all_axes, vmin, vmax)
    for ax in all_axes:
        ax.clear()
    if not multiview:
        f.suptitle(f"Epoch 1")

    ax1.set_title("Ground Truth ODF")
    ax1.imshow(ground_truth_frames[0], extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS), cmap=cmap)
    f.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True), cmap=cmap), ax=ax1)

    ax2.set_title("Learned Mask")
    ax2.imshow(mask_frames[0], extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS))
    f.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True), cmap="viridis"), ax=ax2)

    ax3.set_title("Directional Gradients")
    ax3.imshow(gradient_frames[0], norm=matplotlib.colors.Normalize(vmin=min_grad, vmax=max_grad, clip=True), extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS))
    f.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min_grad, vmax=max_grad, clip=True), cmap="viridis"), ax=ax3)

    ax4.set_title("Learned ODF")
    ax4.imshow(odf_frames[0], extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS), cmap=cmap)
    f.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True), cmap=cmap), ax=ax4)

    if len(constant_frames) > 0:
        ax6.set_title("Learned Constant")
        ax6.imshow(np.ma.masked_where(constant_mask_frames[0] < CONSTANT_MASK_THRESH, constant_frames[0]), extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS), cmap=cmap)
        f.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True), cmap=cmap), ax=ax6)

        ax5.set_title("Baseline Depth")
        ax5.imshow(baseline_depth_frames[0], extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS), cmap=cmap)
        f.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True), cmap=cmap), ax=ax5)


    

    # Set up formatting for movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist="Trevor Houchens"), bitrate=1800)

    def update_depthmap(num, odf_frames, gradient_frames, axes):
        for ax in axes:
            ax.clear()
        # gt_intersect, gt_depth, learned_intersect, learned_depth = rendered_views[num]
        # odf_utils.show_depth_data(gt_intersect, gt_depth, learned_intersect, learned_depth, all_axes, vmin, vmax)
        if not multiview:
            f.suptitle(f"Epoch {num}")
        # axes[0].imshow(odf_frames[num], extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS), cmap=cmap)
        # axes[0].set_title("Learned ODF")
        # axes[1].imshow(gradient_frames[num], norm=matplotlib.colors.Normalize(vmin=min_grad, vmax=max_grad, clip=True), extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS))
        # axes[1].set_title("Directional Gradients")
        if len(axes) ==4:
            ax1, ax2, ax3, ax4 = axes
        else:
            ax1,ax2,ax3,ax4,ax5,ax6 = axes
        ax1.set_title("Ground Truth ODF")
        ax1.imshow(ground_truth_frames[num], extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS), cmap=cmap)
        ax2.set_title("Learned Mask")
        ax2.imshow(mask_frames[num], extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS))
        ax3.set_title("Directional Gradients")
        ax3.imshow(gradient_frames[num], norm=matplotlib.colors.Normalize(vmin=min_grad, vmax=max_grad, clip=True), extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS))
        ax4.set_title("Learned ODF")
        ax4.imshow(odf_frames[num], extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS), cmap=cmap)
        if len(constant_frames) > 0:
            ax6.set_title("Learned Constant")
            ax6.imshow(np.ma.masked_where(constant_mask_frames[num] < CONSTANT_MASK_THRESH, constant_frames[num]), extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS), cmap=cmap)

            ax5.set_title("Baseline Depth")
            ax5.imshow(baseline_depth_frames[num], extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS), cmap=cmap)

    depthmap_ani = animation.FuncAnimation(f, update_depthmap, n_frames, fargs=(odf_frames, gradient_frames, all_axes),
                                   interval=50)
    depthmap_ani.save(video_file, writer=writer)

def make_multiview_video(model, name, save_dir, frames=100, device="cpu"):
    for i in tqdm(range(0, frames)):
        theta = i*2*math.pi/frames
        save_video_frames(model, direction=[math.cos(theta), 0., math.sin(theta)], resolution=256, device=device, multiview=True)
    render_video(name, save_dir, multiview=True)


if __name__ == "__main__":
    Parser = argparse.ArgumentParser(description='Inference code for NeuralODFs.')
    Parser.add_argument('--arch', help='Architecture to use.', choices=['standard', 'constant'], default='standard')
    Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
    Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
    Parser.add_argument('--no-posenc', help='Choose not to use positional encoding.', action='store_true', required=False)
    Parser.add_argument('--resolution', help='Resolution of the mesh to extract', type=int, default=256)
    Parser.add_argument("--save-dir", type=str, help="Directory to save video in")
    Parser.add_argument("--model-name", type=str, help="name of model")
    # Parser.add_argument('--mesh-dir', help="Mesh with ground truth .obj files", type=str)
    # Parser.add_argument('--object', help="Name of the object", type=str)
    Parser.set_defaults(no_posenc=False)

    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    butils.seedRandom(Args.seed)
    nCores = 0#mp.cpu_count()

    usePosEnc = not Args.no_posenc
    print('[ INFO ]: Using positional encoding:', usePosEnc)

    if Args.arch == 'standard':
        print("Using original architecture")
        NeuralODF = ODFSingleV3(input_size=(120 if usePosEnc else 6), radius=DEPTH_SAMPLER_RADIUS, coord_type=Args.coord_type, pos_enc=usePosEnc, n_layers=10)
    elif Args.arch == 'constant':
        print("Using constant prediction architecture")
        NeuralODF = ODFSingleV3Constant(input_size=(120 if usePosEnc else 6), radius=DEPTH_SAMPLER_RADIUS, coord_type=Args.coord_type, pos_enc=usePosEnc, n_layers=10)

    save_dir = Args.save_dir
    name = Args.model_name
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    # Device = torch.device("cpu")
    NeuralODF.setupCheckpoint(device)
    
    make_multiview_video(NeuralODF, name, save_dir, device=device)