import torch
import numpy as np
import math
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import glob
from matplotlib import cm
import matplotlib

np.random.seed(42)
# torch.manual_seed(42)

MAX_RADIUS = 1.25

OUTER_RADIUS = 1.0
INNER_RADIUS = 0.8

# ######################## DATA SAMPLING ########################

def sample_position(n_samples, radius=MAX_RADIUS):
    radii = radius * np.sqrt(np.random.random(n_samples))
    thetas = 2*math.pi*np.random.random(n_samples)
    xs = np.cos(thetas)*radii
    ys = np.sin(thetas)*radii
    positions = np.stack([xs,ys], axis=-1)
    return positions

def sample_direction(n_samples):
    thetas = 2*math.pi*np.random.random(n_samples)
    xs = np.cos(thetas)
    ys = np.sin(thetas)
    directions = np.stack([xs,ys], axis=-1)
    return directions

def circle_depth(positions, directions, radius):
    # quadratic equation for circle intersection given line q = p + td
    # a = dx^2 + dy^2
    # b = 2pxdx + 2pydy
    # c = px^2 + py^2 + r^2
    a = np.sum(np.square(directions), axis=1)
    b = np.sum(2*positions*directions, axis=1)
    c = np.sum(np.square(positions), axis=1) - radius**2.

    inner_term = np.square(b) - 4.*a*c
    intersection_mask = inner_term >= 0.0

    depths = np.ones((positions.shape[0], 2))
    depths *= np.inf
    first_depth = (-b[intersection_mask] + np.sqrt(inner_term[intersection_mask]))/(2.*a[intersection_mask])
    second_depth = (-b[intersection_mask] - np.sqrt(inner_term[intersection_mask]))/(2.*a[intersection_mask])
    depths[intersection_mask] = np.stack([first_depth, second_depth], axis=-1)
    intersection_mask[np.max(depths, axis=1) < 0.0] = False
    depths[depths < 0.0] = np.inf
    return intersection_mask, depths

def torus_depth(positions, directions):
    radii = np.linalg.norm(positions, axis=-1)
    interior_points = np.logical_and(radii < OUTER_RADIUS, radii > INNER_RADIUS)
    adjusted_directions = np.copy(directions)
    adjusted_directions[interior_points, :] = adjusted_directions[interior_points, :] * -1.
    intersection_mask_inner, depths_inner = circle_depth(positions, adjusted_directions, INNER_RADIUS)
    intersection_mask_outer, depths_outer = circle_depth(positions, adjusted_directions, OUTER_RADIUS)
    intersection_mask = np.logical_or(intersection_mask_inner, intersection_mask_outer)
    depths = np.min(np.concatenate([depths_inner, depths_outer], axis=1), axis=1)
    depths[interior_points] = -1 * depths[interior_points]
    return intersection_mask, depths

def inf_grad_coords(size=32):
    valid_positions = []
    valid_directions = []
    valid_depths = []
    total_sampled = 0

    # rejection sampling
    while total_sampled < size:
        positions = sample_position(2*size)
        directions = sample_direction(2*size)
        masks, depths = torus_depth(positions, directions)
        total_sampled += np.sum(masks)
        if np.any(total_sampled):
            valid_positions.append(positions[masks])
            valid_directions.append(directions[masks])
            valid_depths.append(depths[masks])

    positions = np.concatenate(valid_positions, axis=0)[:size,...]
    directions = np.concatenate(valid_directions, axis=0)[:size,...]
    depths = np.concatenate(valid_depths, axis=0)[:size,...]

    surface_positions = positions + np.stack([depths, depths], axis=-1)*directions
    surface_directions = directions*-1
    surface_coords = np.concatenate([surface_positions, surface_directions], axis=1)
    return surface_coords



class TorusDataset2D(torch.utils.data.Dataset):

    def __init__(self, size=10000):
        self.size = size
        self.positions = sample_position(size)
        self.directions = sample_direction(size)
        self.coords = torch.cat([torch.tensor(self.positions), torch.tensor(self.directions)], axis=-1).float()
        self.intersection_mask, self.depths = torus_depth(self.positions, self.directions)
        self.depths[self.depths == np.inf] = 0.0

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.coords[index], (self.intersection_mask[index], self.depths[index])


# ######################## MODEL ########################

class ODF2D(torch.nn.Module):

    def __init__(self, input_size=4, n_layers=6, hidden_size=256, radius=MAX_RADIUS):
        super().__init__()

        self.radius=radius

        #main network layers
        main_layers = []
        main_layers.append(torch.nn.Linear(input_size, hidden_size))
        for l in range(n_layers-1):
            main_layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.network = torch.nn.ModuleList(main_layers)

        
        #intersection head
        intersection_layers = [torch.nn.Linear(hidden_size, hidden_size), torch.nn.Linear(hidden_size, 1)]
        self.intersection_head = torch.nn.ModuleList(intersection_layers)

        #depth head
        depth_layers = [torch.nn.Linear(hidden_size, hidden_size), torch.nn.Linear(hidden_size, 1)]
        self.depth_head = torch.nn.ModuleList(depth_layers)

        self.relu = torch.nn.ReLU()

    def forward(self, input):

        x = input
        for i in range(len(self.network)):
            x = self.network[i](x)
            x = self.relu(x)

        
        intersections = self.intersection_head[0](x)
        intersections = self.relu(intersections)
        intersections = self.intersection_head[1](intersections)

        depths = self.depth_head[0](x)
        depths = self.relu(depths)
        depths = self.depth_head[1](depths)

        return (intersections, depths)

class ODF2DV2(torch.nn.Module):

    def __init__(self, input_size=4, n_layers=6, hidden_size=256, radius=MAX_RADIUS):
        super().__init__()

        self.radius=radius

        #main network layers
        main_layers = []
        main_layers.append(torch.nn.Linear(input_size, hidden_size))
        for l in range(n_layers-1):
            main_layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.network = torch.nn.ModuleList(main_layers)

        
        #intersection head
        intersection_layers = [torch.nn.Linear(hidden_size, hidden_size), torch.nn.Linear(hidden_size, 1)]
        self.intersection_head = torch.nn.ModuleList(intersection_layers)

        #depth head
        depth_layers = [torch.nn.Linear(hidden_size, hidden_size), torch.nn.Linear(hidden_size, 1)]
        self.depth_head = torch.nn.ModuleList(depth_layers)

        # constant head
        constant_layers = [torch.nn.Linear(hidden_size, hidden_size), torch.nn.Linear(hidden_size, 1)]
        self.constant_head = torch.nn.ModuleList(constant_layers)

        #constant mask head
        constant_mask_layers = [torch.nn.Linear(hidden_size, hidden_size), torch.nn.Linear(hidden_size, 1)]
        self.constant_mask_head = torch.nn.ModuleList(constant_mask_layers)

        self.relu = torch.nn.ReLU()

    def forward(self, input):

        x = input
        for i in range(len(self.network)):
            x = self.network[i](x)
            x = self.relu(x)

        
        intersections = self.intersection_head[0](x)
        intersections = self.relu(intersections)
        intersections = self.intersection_head[1](intersections)

        depths = self.depth_head[0](x)
        depths = self.relu(depths)
        depths = self.depth_head[1](depths)

        constants = self.constant_head[0](x)
        constants = self.relu(constants)
        constants = self.constant_head[1](constants)

        constant_mask = self.constant_mask_head[0](x)
        constant_mask = self.relu(constant_mask)
        constant_mask = self.constant_mask_head[1](constant_mask)

        return (intersections, depths, constant_mask, constants)



# ######################## LOSSES ########################

MASK_THRESH = 0.5
DEPTH_LAMBDA = 5.0
class DepthLoss(torch.nn.Module):

    def __init__(self, thresh=MASK_THRESH, lmbda=DEPTH_LAMBDA):
        super().__init__()
        self.thresh = thresh
        self.lmbda = lmbda
        self.mask_loss_fn = torch.nn.BCELoss(reduction='mean')
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, output, target):
        gt_mask, gt_depth = target
        if len(output) == 2:
            pred_mask, pred_depth = output
            pred_mask = pred_mask.flatten()
            pred_depth = pred_depth.flatten()
        else:
            pred_mask, pred_depth, constant_mask, constants = output
            pred_mask = pred_mask.flatten()
            pred_depth = pred_depth.flatten()
            pred_depth += self.sigmoid(constant_mask.flatten())*constants.flatten()
        pred_mask_confidence = self.sigmoid(pred_mask)
        # valid_rays = pred_mask_confidence > self.thresh
        valid_rays = gt_mask > self.thresh
        # print(pred_mask_confidence)
        # print(gt_mask)
        mask_loss = self.mask_loss_fn(pred_mask_confidence.to(torch.float), gt_mask.to(torch.float))
        # print(valid_rays)
        l2_loss = torch.mean(torch.square(gt_depth[valid_rays]-pred_depth[valid_rays]))
        # if l2_loss > 10.:
        #     print("GT DEPTH")
        #     print(gt_depth)
        #     print("PRED Depth")
        #     print(pred_depth)
        if math.isnan(l2_loss) or math.isinf(l2_loss):
            l2_loss = torch.tensor(0)
        # if mask_loss > 10.:
        #     print(f"mask loss: {mask_loss}")
        # if l2_loss > 10.:
        #     print(f"l2 loss: {l2_loss}")
        loss = self.lmbda * l2_loss + mask_loss
        return loss

def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True)
    return points_grad


class DepthFieldRegularizingLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, model):
        train_coords = input
        additional_coords = torch.cat([torch.tensor(sample_position(train_coords.shape[0])), torch.tensor(sample_direction(train_coords.shape[0]))], axis=-1).to(train_coords.device)
        coords = torch.cat([train_coords, additional_coords], axis=0).float()

        coords.requires_grad_()
        output = model(coords)
        intersections, depths = output[:2]
        intersections = intersections.flatten()
        depths = depths.flatten()

        x_grads = gradient(coords, depths)[0][...,:2]
        odf_gradient_directions = coords[:, 2:]

        dfr_mask = intersections > 0.5
        # thresh = 1.5
        # dfr_mask = torch.logical_and(intersections > 0.5, torch.sum(odf_gradient_directions*x_grads, dim=-1) < thresh)

        if torch.sum(dfr_mask) != 0.:
            grad_dir_loss = torch.mean(torch.abs(torch.sum(odf_gradient_directions[dfr_mask]*x_grads[dfr_mask], dim=-1) + 1.))
            # grad_dir_loss = torch.mean(torch.square(torch.sum(odf_gradient_directions[intersections>0.5]*x_grads[intersections>0.5], dim=-1) + 1.))
        else:
            grad_dir_loss = torch.tensor(0.)

        # promote increasing large gradients
        # large_grad_mask = torch.logical_and(intersections > 0.5, torch.sum(odf_gradient_directions*x_grads, dim=-1) >= thresh)
        # if torch.sum(large_grad_mask) != 0.:
        #     large_grad_loss = torch.sum(odf_gradient_directions[dfr_mask]*x_grads[dfr_mask], dim=-1)
        #     start_loss = thresh + 1.
        #     large_grad_loss -= (thresh-1.)
        #     large_grad_loss = torch.sum(torch.reciprocal(large_grad_loss)*start_loss)
        #     grad_dir_loss += large_grad_loss

        return grad_dir_loss

class ConstantRegularizingLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, model):
        train_coords = input
        additional_coords = torch.cat([torch.tensor(sample_position(train_coords.shape[0])), torch.tensor(sample_direction(train_coords.shape[0]))], axis=-1).to(train_coords.device)
        coords = torch.cat([train_coords, additional_coords], axis=0).float()

        coords.requires_grad_()
        output = model(coords)
        assert(len(output) > 2)
        constant_mask, constants = output[2:]
        constant_mask = constant_mask.flatten()
        constants = constants.flatten()

        x_grads = gradient(coords, constants)[0][...,:2]
        view_dirs = coords[:, 2:]
        # orthogonal_dirs = torch.stack([view_dirs[:,1], -1.*view_dirs[:,0]], dim=-1)

        dfr_mask = constant_mask > 0.5
        # thresh = 1.5
        # dfr_mask = torch.logical_and(intersections > 0.5, torch.sum(odf_gradient_directions*x_grads, dim=-1) < thresh)

        if torch.sum(dfr_mask) != 0.:
            grad_dir_loss = torch.mean(torch.abs(torch.sum(view_dirs[dfr_mask]*x_grads[dfr_mask], dim=-1)))
        else:
            grad_dir_loss = torch.tensor(0.)

        return grad_dir_loss

class InfiniteSurfaceGradientLoss(torch.nn.Module):

    def __init__(self, batch_size, device):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
    
    def forward(self, model):
        coords = torch.tensor(inf_grad_coords(size=self.batch_size)).float().to(self.device)
        coords.requires_grad_()
        output = model(coords)
        if len(output) == 2:
            intersections, depths = output[:2]
        else:
            intersections, depths = output[:2]
        intersections = intersections.flatten()
        depths = depths.flatten()
        x_grads = gradient(coords, depths)[0][...,:2]
        odf_gradient_directions = coords[:, 2:]
        # grad_dir_loss = torch.mean(-1.*torch.sum(odf_gradient_directions[intersections>0.5]*x_grads[intersections>0.5], dim=-1))
        grad_dir_loss = torch.mean(torch.reciprocal(torch.exp(torch.sum(odf_gradient_directions[intersections>0.5]*x_grads[intersections>0.5], dim=-1))))
        return grad_dir_loss



# ######################## VIZUALIZATION ########################

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

def show_data(size=10):
    positions = sample_position(size)
    directions = sample_direction(size)
    # print(positions)
    # print(directions)
    intersection_mask, depths = torus_depth(positions, directions)
    # print(positions)
    # print(directions)
    # print(depths)

    fig, ax = plt.subplots()

    circle1 = plt.Circle((0,0), 1.0, color="tab:blue")
    circle2 = plt.Circle((0,0), 0.8, color="white")



    ax.plot(positions[:,0], positions[:,1], "o", color="blue")

    for i in range(size):
        start = positions[i]
        depth = depths[i] if intersection_mask[i] else 0.1
        end = start + depth*directions[i]
        color = "tab:orange" if intersection_mask[i] else "red"

        ax.plot([start[0], end[0]], [start[1], end[1]], color=color)

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    plt.show()

def show_surface_points(size=10):
    coords = inf_grad_coords(size=size)

    fig, ax = plt.subplots()

    circle1 = plt.Circle((0,0), 1.0, color="tab:blue")
    circle2 = plt.Circle((0,0), 0.8, color="white")

    ax.plot(coords[:,0], coords[:,1], "o", color="green")

    starts = coords[:,:2]
    ends = starts + coords[:,2:]*0.1
    for i in range(size):
        ax.plot([starts[i,0], ends[i,0]], [starts[i,1], ends[i,1]], color="blue")

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    plt.show()

def show_gradient_histogram(model, n_points=100000, device="cpu"):
    model.eval()
    positions = sample_position(n_points)
    directions = sample_direction(n_points)
    coords = np.concatenate([positions, directions], axis=-1)
    torch_coords = torch.tensor(coords).float().to(device)
    torch_coords.requires_grad_()
    masks, depths = model(torch_coords)[:2]
    masks = masks.detach().cpu().numpy().flatten()
    masks = masks > MASK_THRESH
    depths = depths.flatten()

    x_grads = gradient(torch_coords, depths)[0][...,:2].detach().cpu().numpy()
    directional_gradients = np.sum(x_grads*coords[:,2:], axis=-1)
    directional_gradients = directional_gradients[masks]

    min_hist = -3.
    max_hist = 5.
    hist_vals = np.clip(directional_gradients, min_hist, max_hist)
    plt.hist(hist_vals, bins=100, range=(min_hist,max_hist), density=True)
    plt.show()


def grid_positions(resolution, bounds=MAX_RADIUS):
    width = 2.*bounds
    step_size = width/resolution
    linear_coords = np.array([-bounds + (i+0.5)*step_size for i in range(resolution)])
    xs, ys = np.meshgrid(linear_coords, np.flip(linear_coords))
    xs = xs.flatten()
    ys = ys.flatten()
    coords = np.stack([xs,ys], axis=-1)
    return torch.tensor(coords, dtype=torch.float)

# def color_in_odf(masks, depths, resolution, bounds=MAX_RADIUS):
#     max_depth = 1.*MAX_RADIUS
#     min_depth = -1.*MAX_RADIUS
#     max_pos_color = np.array([2./255., 10./255., 94./255.]).reshape((1,-1))
#     min_pos_color = np.array([173./255., 196./255., 247./255.]).reshape((1,-1))
#     max_neg_color = np.array([217./255., 114./255., 4./255.]).reshape((1,-1))
#     min_neg_color = np.array([232./255., 196./255., 158./255.]).reshape((1,-1))

#     img = np.ones((resolution * resolution, 3))
#     positive_indices = np.logical_and(masks, depths > 0.0)
#     negative_indices = np.logical_and(masks, depths < 0.0)
#     zero_indices = np.logical_and(masks, depths == 0.0)
#     img[zero_indices, :] = min_pos_color
#     depth_pos_frac = np.clip((depths/max_depth).reshape((-1,1)), 0., 1.)
#     depth_pos_colors = np.matmul(depth_pos_frac,max_pos_color) + np.matmul(-1.*depth_pos_frac+1., min_pos_color)
#     img[positive_indices, :] = depth_pos_colors[positive_indices, :]
#     depth_neg_frac = np.clip((depths/min_depth).reshape((-1,1)), 0., 1.)
#     # print(np.max(depth_neg_frac))
#     # depth_neg_frac = (depths/min_depth).reshape((-1,1))
#     depth_neg_colors = np.matmul(depth_neg_frac,max_neg_color) + np.matmul(-1.*depth_neg_frac+1., min_neg_color)
#     img[negative_indices, :] = depth_neg_colors[negative_indices, :]
#     return img.reshape((resolution, resolution, 3))

def show_gradient_vectors(ax, model, coordinates, resolution, stride=32, device="cpu"):
    gradient_coordinates = []
    for i in range(0, resolution, stride):
        for j in range(0, resolution, stride):
            gradient_coordinates.append(coordinates[i*resolution+j, :])
    gradient_coordinates = np.array(gradient_coordinates)
    torch_gradient_coordinates = torch.tensor(gradient_coordinates).float().to(device)
    torch_gradient_coordinates.requires_grad_()
    intersections, depths = model(torch_gradient_coordinates)[:2]
    intersections = intersections.flatten()
    depths = depths.flatten()

    x_grads = gradient(torch_gradient_coordinates, depths)[0][...,:2].detach().cpu().numpy()
    # odf_gradient_directions = gradient_coordinates[:, 2:]

    # plot the gradient vectors
    vec_mag = 0.1
    count = 0
    sum_mse = 0
    for i in range(gradient_coordinates.shape[0]):
        if intersections[i] > MASK_THRESH:
            # grad_vec = x_grads[i] / np.linalg.norm(x_grads[i]) * vec_mag
            grad_vec = x_grads[i] 
            view_vec = gradient_coordinates[i,2:] * vec_mag
            # plot centers
            xs = [gradient_coordinates[i,0]]
            ys = [gradient_coordinates[i,1]]
            ax.scatter(xs,ys,color="green")
            # plot gradients
            xs.append(xs[0]+grad_vec[0])
            ys.append(ys[0]+grad_vec[1])
            ax.plot(xs,ys,color="red")
            # plot view dirs
            xs.pop()
            ys.pop()
            xs.append(xs[0]+view_vec[0])
            ys.append(ys[0]+view_vec[1])
            ax.plot(xs,ys,color="blue")
            sum_mse += (grad_vec[1]+1.)**2.
            count += 1
    print(f"MSE: {sum_mse/count}")


def show_odf(model, direction=[0.,1.], resolution=512, device="cpu"):
    model.eval()
    direction = np.array(direction) / np.linalg.norm(direction)
    coords = grid_positions(resolution)
    coords = np.concatenate([coords, np.repeat(direction[None,:], coords.shape[0], axis=0)], axis=1)
    torch_coords = torch.tensor(coords).float().to(device)
    sigmoid = torch.nn.Sigmoid()
    outputs = model(torch_coords)
    if len(outputs) == 2:
        masks, depths = outputs
        depths = depths.detach().cpu().numpy().flatten()
    else:
        masks, depths, constant_mask, constants = outputs
        constant_mask = sigmoid(constant_mask)
        constant_mask = constant_mask > MASK_THRESH
        constant_mask = constant_mask.detach().cpu().numpy().flatten()
        constants = constants.detach().cpu().numpy().flatten()
        depths = depths.detach().cpu().numpy().flatten()
        depths += constant_mask*constants

    masks = masks.detach().cpu().numpy().flatten()
    masks = masks > MASK_THRESH
    depths = np.ma.masked_where(np.logical_not(masks), depths)
    fig, ax = plt.subplots()
    vmin = np.min(depths)
    vmax = np.max(depths)
    cmap=get_depth_cmap(vmin, vmax)
    ax.imshow(depths, extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS), cmap=cmap)
    fig.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True), cmap=cmap))
    # ax.plot([0,10], [0,10])
    show_gradient_vectors(ax, model, coords, resolution, stride=64, device=device)
    plt.show()

def show_gradients(model, direction=[0.,1.], resolution=512, device="cpu"):
    model.eval()
    direction = np.array(direction) / np.linalg.norm(direction)
    coords = grid_positions(resolution)
    coords = np.concatenate([coords, np.repeat(direction[None,:], coords.shape[0], axis=0)], axis=1)
    torch_coords = torch.tensor(coords).float().to(device)
    gradients = []
    masks = []
    batch_size = 100000
    for i in range(0, coords.shape[0], batch_size):
        batch_coords = torch_coords[i:i+batch_size,...]
        batch_coords.requires_grad_()
        intersections, depths = model(batch_coords)[:2]
        intersections = intersections.detach().cpu().numpy().flatten()
        depths = depths.flatten()
        x_grads = gradient(batch_coords, depths)[0][...,:2].detach().cpu().numpy()
        directional_gradients = np.sum(x_grads * coords[i:i+batch_size,2:], axis=-1)
        gradients.append(directional_gradients)
        masks.append(intersections > MASK_THRESH)
    gradients = np.concatenate(gradients)
    masks = np.concatenate(masks)
    gradients = gradients * masks
    gradients = gradients.reshape((resolution, resolution))
    print(gradients.shape)
    fig, ax = plt.subplots()
    ax.imshow(gradients, extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS))
    plt.show()

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
        ax6.imshow(np.ma.masked_where(constant_mask_frames[0] < MASK_THRESH, constant_frames[0]), extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS), cmap=cmap)
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
            ax6.imshow(np.ma.masked_where(constant_mask_frames[num] < MASK_THRESH, constant_frames[num]), extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS), cmap=cmap)

            ax5.set_title("Baseline Depth")
            ax5.imshow(baseline_depth_frames[num], extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS), cmap=cmap)

    depthmap_ani = animation.FuncAnimation(f, update_depthmap, n_frames, fargs=(odf_frames, gradient_frames, all_axes),
                                   interval=50)
    depthmap_ani.save(video_file, writer=writer)

def make_multiview_video(model, name, save_dir, frames=100, device="cpu"):
    for i in tqdm(range(0, frames)):
        theta = i*2*math.pi/frames
        save_video_frames(model, direction=[math.cos(theta), math.sin(theta)], resolution=256, device=device, multiview=True)
    render_video(name, save_dir, multiview=True)



# ######################## TRAINING ########################

def load_model(name, save_dir, last_checkpoint=True, device="cpu"):
    experiment_dir = os.path.join(save_dir, name)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    model_file = f"{name}_last.pt" if last_checkpoint else f"{name}_best.pt"
    model_path = os.path.join(experiment_dir, model_file)
    model = ODF2DV2().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model

def save_epoch(name, save_dir, epoch_losses):
    loss_file = os.path.join(save_dir, name, "losses.npy")
    plot_file = os.path.join(save_dir, name, "loss", f"{name}_loss_curves.png")
    if not os.path.exists(os.path.join(save_dir, name, "loss")):
        os.mkdir(os.path.join(save_dir, name, "loss"))
    if os.path.exists(loss_file):
        loss_history = np.load(loss_file, allow_pickle=True).item()
    else:
        loss_history = {}
    n_epochs = 0
    for loss, hist in loss_history.items():
        n_epochs = len(hist)
    for loss, val in epoch_losses.items():
        if not loss in loss_history:
            loss_history[loss] = []
        loss_history[loss] += [0.,]*int(n_epochs-len(loss_history[loss]))
        loss_history[loss].append(val)
    # plot losses
    fig, ax = plt.subplots()
    ax.set_title(f"{name} Loss Curves")
    epochs = np.arange(n_epochs+1) + 1.
    for loss, hist in loss_history.items():
        style = "solid" if "main" in loss else "dashed"
        ax.plot(epochs, hist, linestyle=style, label=loss)
    ax.set_xlabel("Epochs")
    ax.legend()
    plt.savefig(plot_file)
    plt.close()
    np.save(loss_file, loss_history)

def save_video_frames(model, direction=[0.,1.], resolution=256, device="cpu", multiview=False):
    model.eval()
    direction = np.array(direction) / np.linalg.norm(direction)
    coords = grid_positions(resolution)
    coords = np.concatenate([coords, np.repeat(direction[None,:], coords.shape[0], axis=0)], axis=1)
    gt_mask, gt_depths = torus_depth(coords[...,:2], coords[...,2:])
    gt_depths = np.ma.masked_where(np.logical_not(gt_mask > MASK_THRESH), gt_depths)
    gt_depths = gt_depths.reshape((resolution, resolution))
    torch_coords = torch.tensor(coords).float().to(device)
    sigmoid = torch.nn.Sigmoid()

    # TODO: batch this part too??
    output = model(torch_coords)

    if len(output) == 2:
        masks, depths = output
        depths = depths.detach().cpu().numpy().flatten()
    else:
        masks, depths, constant_mask, constants = output
        constant_mask = sigmoid(constant_mask)
        # constant_mask = constant_mask > MASK_THRESH
        constant_mask = constant_mask.detach().cpu().numpy().flatten()
        constants = constants.detach().cpu().numpy().flatten()
        depths = depths.detach().cpu().numpy().flatten()
        original_depths = np.copy(depths)
        depths += constant_mask*constants
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
        output=model(batch_coords)
        intersections, grad_depths = output[:2]
        intersections = intersections.detach().cpu().numpy().flatten()
        grad_depths = grad_depths.flatten()
        x_grads = gradient(batch_coords, grad_depths)[0][...,:2].detach().cpu().numpy()
        directional_gradients = np.sum(x_grads * coords[i:i+batch_size,2:], axis=-1)
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

def train(model, name, batch_size=32, epochs=100, save_dir="F:\\ivl-data\\ODF2D", device="cpu"):
    residuals = False
    last_checkpoint_path = os.path.join(save_dir, name, f"{name}_last.pt")
    best_checkpoint_path = os.path.join(save_dir, name, f"{name}_best.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    train_dataset = TorusDataset2D()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = TorusDataset2D()
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    depth_loss = DepthLoss()
    reg_loss = DepthFieldRegularizingLoss()
    const_reg_loss = ConstantRegularizingLoss()
    inf_grad_loss = InfiniteSurfaceGradientLoss(batch_size=batch_size, device=device)
    best_val_loss = np.inf
    
    for e in range(epochs):
        print(f"EPOCH: {e+1}")
        # Training
        epoch_train_loss = 0.
        epoch_losses = {}
        # train_losses = ["train_main", "train_depth", "train_dfr", "train_boundary"]
        # train_losses = ["train_main", "train_depth", "train_dfr", "train_const"]
        train_losses = ["train_main", "train_depth", "train_dfr"]
        val_losses = ["val_main"]
        # TODO: zip names and losses to make changes easier --> and weights?
        for ln in train_losses+val_losses:
            epoch_losses[ln] = 0.
        model.train()
        for coords, (masks, depths) in tqdm(train_dataloader):
            coords = coords.to(device)
            masks = masks.to(device)
            depths = depths.to(device)
            
            optimizer.zero_grad()
            output = model(coords)
            d_loss = depth_loss(output, (masks, depths))
            loss = d_loss
            regularization_loss = reg_loss(coords, model)
            loss += regularization_loss
            # constant_regularization_loss = const_reg_loss(coords, model)
            # loss += constant_regularization_loss
            # boundary_loss = inf_grad_loss(model)
            # loss += boundary_loss
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.detach().cpu()
            epoch_losses["train_depth"] += d_loss.detach().cpu().numpy()
            epoch_losses["train_dfr"] += regularization_loss.detach().cpu().numpy()
            # epoch_losses["train_const"] += constant_regularization_loss.detach().cpu().numpy()
            # epoch_losses["train_boundary"] += boundary_loss.detach().cpu().numpy()


        print(f"Train Loss: {epoch_train_loss/len(train_dataloader)}")
        epoch_losses["train_main"] = epoch_train_loss.numpy()
        for loss in train_losses:
            epoch_losses[loss] /= len(train_dataloader)
        torch.save(model.state_dict(), last_checkpoint_path)

        # Validation
        epoch_val_loss = 0.
        model.eval()
        for coords, (masks, depths) in tqdm(val_dataloader):
            coords = coords.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            output = model(coords)
            loss = depth_loss(output, (masks, depths))
            loss += reg_loss(coords, model)
            # loss += const_reg_loss(coords, model)
            # loss += inf_grad_loss(model)

            epoch_val_loss += loss.detach().cpu()
        print(f"Validation Loss: {epoch_val_loss/len(val_dataloader)}")
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
        
        epoch_losses["val_main"] = epoch_val_loss.numpy()
        for loss in val_losses:
            epoch_losses[loss] /= len(val_dataloader)
        
        save_epoch(name, save_dir, epoch_losses)
        save_video_frames(model, device=device)




if __name__ == "__main__":
    name = "jan27_reg_const"
    save_dir = "F:\\ivl-data\\ODF2D"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(name, save_dir, last_checkpoint=True, device=device)

    # show_surface_points()
    # show_gradients(model, device=device)
    # train(model,name, epochs=50, save_dir=save_dir, device=device)
    # show_odf(model, device=device)
    # render_video(name, save_dir)
    # show_gradient_histogram(model, device=device)
    # show_data()
    make_multiview_video(model, name, save_dir, frames=100, device=device)