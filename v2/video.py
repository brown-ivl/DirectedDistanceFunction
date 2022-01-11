import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import sys, os
from beacon import utils as butils
import pyrender
import trimesh
import math
from tqdm import tqdm


FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from single_models import ODFSingleV3
from depth_sampler_5d import DEPTH_SAMPLER_RADIUS

def load_object(obj_name, data_path):
    obj_file = os.path.join(data_path, obj_name)

    obj_mesh = trimesh.load(obj_file)
    # obj_mesh.show()

    ## deepsdf normalization
    mesh_vertices = obj_mesh.vertices
    mesh_faces = obj_mesh.faces
    center = (mesh_vertices.max(axis=0) + mesh_vertices.min(axis=0))/2.0
    max_dist = np.linalg.norm(mesh_vertices - center, axis=1).max()
    max_dist = max_dist * 1.03
    mesh_vertices = (mesh_vertices - center) / max_dist
    obj_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
    
    return mesh_vertices, mesh_faces, obj_mesh


class PyrenderCamera():
    '''
    This class represents a camera and allows views to be rendered by rasterizing a mesh.
        cam_center        - the coordinates of the camera center (x,y,z)
        direction         - a vector defining the direction that the camera is pointing, relative to the camera center
        up                - the up vector of the camera
        focal_length      - the focal length of the camera
        sensor_size       - the dimensions of the sensor (u,v)
        sensor_resolution - The number of pixels on each edge of the sensor (u,v)
    '''
    def __init__(self, center=[1.,1.,1.], direction=[1.,1.,1.], up=[0.,1.,0.], focal_length=1.0, sensor_size=[1.,1.], sensor_resolution=[256,256]):
        super().__init__()
        self.center = np.array(center)
        assert(np.linalg.norm(direction) != 0.)
        assert(np.linalg.norm(up) != 0.)
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.up = np.array(up) / np.linalg.norm(up)
        self.right = np.cross(self.up, self.direction) # cross up first because the look is in the negative direction for pyrender
        assert(np.linalg.norm(self.right) != 0.)
        self.focal_length = focal_length
        self.sensor_size = sensor_size
        self.sensor_resolution = sensor_resolution

        print("+++++  PYRENDER CAMERA +++++")
        print(f"\tLook: {self.direction}")
        print(f"\tUp: {self.up}")
        print(f"\tRight: {self.right}")

        self.rotation_matrix = self.compute_opengl_rotation_matrix()
        self.reverse_rotation_matrix = np.copy(self.rotation_matrix)
        self.reverse_rotation_matrix[:3,1:3] = self.reverse_rotation_matrix[:3,1:3] * -1.


    def compute_opengl_rotation_matrix(self):
        
        camera_pose_matrix = np.concatenate((self.right[:,None],
                                        self.up[:,None],
                                        self.direction[:,None],
                                        self.center[:,None]), axis=1)
        
        camera_pose_matrix = np.concatenate((camera_pose_matrix, 
                np.asarray([[0., 0., 0., 1.]])), axis=0)
        return camera_pose_matrix

    def render_mesh(self, mesh_vertices, mesh_faces, obj_mesh):

        inside_mesh = obj_mesh.contains([self.center])[0]

        trimesh_mesh = trimesh.base.Trimesh(mesh_vertices, mesh_faces)
        mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)

        opengl_camera = pyrender.PerspectiveCamera(yfov=np.pi/2.0, aspectRatio=1, znear=0.00001, zfar=3)
        scene = pyrender.Scene()
        scene.add_node(pyrender.Node(mesh=mesh))
        camera_functional_pose = self.reverse_rotation_matrix if inside_mesh else self.rotation_matrix
        scene.add(opengl_camera, pose=camera_functional_pose)
        r = pyrender.OffscreenRenderer(self.sensor_resolution[0], self.sensor_resolution[1])
        _, depth_map = r.render(scene, flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES)

        if inside_mesh:
            print("INSIDE GT MESH")
            depth_map = -1. * depth_map
            # flip to account for not negating the right vector in the reverse camera matrix
            depth_map = np.flip(depth_map, axis=1)

        intersects = np.logical_not((depth_map==0))

        return intersects, depth_map




class ODFCamera():
    '''
    This class represents a camera and allows views to be rendered by querying a learned ODF.
        cam_center        - the coordinates of the camera center (x,y,z)
        direction         - a vector defining the direction that the camera is pointing, relative to the camera center
        up                - the up vector of the camera
        focal_length      - the focal length of the camera
        sensor_size       - the dimensions of the sensor (u,v)
        sensor_resolution - The number of pixels on each edge of the sensor (u,v)
    '''

    def __init__(self, center=[1.,1.,1.], direction=[-1.,-1.,-1.], up=[0.,1.,0.], focal_length=1.0, sensor_size=[1.,1.], sensor_resolution=[256,256]):
        super().__init__()
        self.center = np.array(center)
        assert(np.linalg.norm(direction) != 0.)
        assert(np.linalg.norm(up) != 0.)
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.up = np.array(up) / np.linalg.norm(up)
        self.right = np.cross(self.direction, self.up)
        assert(np.linalg.norm(self.right) != 0.)
        self.focal_length = focal_length
        self.sensor_size = sensor_size
        self.sensor_resolution = sensor_resolution

        print("+++++  ODF CAMERA +++++")
        print(f"\tLook: {self.direction}")
        print(f"\tUp: {self.up}")
        print(f"\tRight: {self.right}")

    def change_resolution(self, resolution):
        self.sensor_resolution = resolution

    def generate_rays(self):
        '''
        Returns a list of rays ( [start point, end point] ), where each ray intersects one pixel. The start point of each ray is the camera center.
        Rays are returned top to bottom, left to right.
        '''
        u_steps = np.linspace(-self.sensor_size[0], self.sensor_size[0], num=self.sensor_resolution[0])
        v_steps = np.linspace(-self.sensor_size[1], self.sensor_size[1], num=self.sensor_resolution[1])
        us, vs = np.meshgrid(u_steps, v_steps)
        us = us.flatten()
        vs = vs.flatten()
        rays = [[np.array(self.center), np.array(self.center + self.focal_length * self.direction + us[i]*self.up) + vs[i]*self.right] for i in range(us.shape[0])]
        return rays


    def odf_depths(self, odf, device="cuda", radius=1.25):
        '''
        Returns an intersection map and a depthmap from a learned model from the camera's perspective
        '''
        print("DEVICE ", device)
        rays = self.generate_rays()
        odf=odf.eval()
        # rays_in_scene_mask = np.array([True if ray != None else False for ray in rays])
        # rays_in_scene = torch.tensor([list(ray[0]) + list(ray[1]-ray[0]) for ray in rays if ray != None])
        rays_in_scene = [ray for ray in rays if np.linalg.norm(ray[0]) <= radius]
        if len(rays_in_scene) > 0:
            with torch.no_grad():
                # encoded_rays = torch.tensor([[x for val in list(ray[0]) + list((ray[1]-ray[0])/np.linalg.norm(ray[1]-ray[0])) for x in odf_utils.positional_encoding(val)] for ray in rays_in_scene]).to(device)
                input_rays = torch.tensor([list(ray[0]) + list((ray[1]-ray[0])/np.linalg.norm(ray[1]-ray[0])) for ray in rays_in_scene]).float().to(device)
                intersect, depth = odf([input_rays], {})[0]
                intersect = intersect.cpu().squeeze()
                model_depths = depth.cpu().squeeze()
            # intersection_mask = rays_in_scene_mask.astype(float)
            # intersection_mask[rays_in_scene_mask] = intersect
            intersection_mask = intersect.reshape(self.sensor_resolution)
            depth = np.zeros((len(rays),))
            # depth[intersect < 0.5] = np.inf
            depth[intersect >= 0.5] = model_depths[intersect >= 0.5]
            depth = depth.reshape((self.sensor_resolution))
        else:
            intersection_mask = np.zeros(self.sensor_resolution)
            depth = np.ones(self.sensor_resolution) * np.inf
        return np.array(intersection_mask > 0.5).transpose(), np.array(depth).transpose()

def depths_to_img(depths, intersects, max_depth=None, min_depth=None):
    if max_depth is None:
        max_depth = np.max(depths)
    if min_depth is None:
        min_depth = np.min(depths)

    no_intersect_color = [1.,1.,1.]
    max_depth_color = [0.4,0.0,0.5]
    surface_color = [1.,0.8,0.]
    min_depth_color = [0.1,0.6,0.2]
    depth_img = np.zeros(depths.shape + (3,))
    
    # set positive depth colors
    pos_depth_inds = depths >= 0.
    pos_depths = depths[pos_depth_inds]
    pos_depth_frac_max = pos_depths/max_depth if max_depth > 0. else pos_depths
    pos_depth_colors = np.outer(pos_depth_frac_max, max_depth_color)
    pos_depth_colors += np.outer(1.-pos_depth_frac_max, surface_color)
    depth_img[pos_depth_inds] = pos_depth_colors

    # set negative depth colors
    neg_depth_inds = depths <= 0.
    neg_depths = depths[neg_depth_inds]
    neg_depth_frac_min = neg_depths/min_depth if min_depth < 0. else neg_depths
    neg_depth_colors = np.outer(neg_depth_frac_min, min_depth_color)
    neg_depth_colors += np.outer(1.-neg_depth_frac_min, surface_color)
    depth_img[neg_depth_inds] = neg_depth_colors

    # set non intersecting to white
    depth_img[np.logical_not(intersects)] = no_intersect_color

    return depth_img

def save_video(rendered_views_model, rendered_views_mesh, file_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation

    n_frames = len(rendered_views_model)

    f, ((ax1, ax2)) = plt.subplots(1,2)
    f.set_size_inches(12.,6.)
    all_axes = [ax1,ax2]

    # display first view
    # gt_intersect, gt_depth, learned_intersect, learned_depth = rendered_views[0]
    # odf_utils.show_depth_data(gt_intersect, gt_depth, learned_intersect, learned_depth, all_axes, vmin, vmax)
    for ax in all_axes:
        ax.clear()
    ax1.imshow(rendered_views_mesh[0])
    ax1.set_title("Ground Truth")
    ax2.imshow(rendered_views_model[1])
    ax2.set_title("Predicted")

    # Set up formatting for movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist="Trevor Houchens"), bitrate=1800)

    def update_depthmap(num, mesh_views, model_views, axes):
        for ax in axes:
            ax.clear()
        # gt_intersect, gt_depth, learned_intersect, learned_depth = rendered_views[num]
        # odf_utils.show_depth_data(gt_intersect, gt_depth, learned_intersect, learned_depth, all_axes, vmin, vmax)
        axes[0].imshow(mesh_views[num])
        axes[0].set_title("Ground Truth")
        axes[1].imshow(model_views[num])
        axes[1].set_title("Predicted")

    depthmap_ani = animation.FuncAnimation(f, update_depthmap, n_frames, fargs=(rendered_views_mesh, rendered_views_mesh, all_axes),
                                   interval=50)
    depthmap_ani.save(file_name, writer=writer)

def equatorial_video(output_dir, expt_name, model, mesh_vertices, mesh_faces, obj_mesh, resolution=256, n_frames=50):

    video_dir = os.path.join(output_dir, expt_name, "videos")
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    

    cam_radius = 1.2
    # these are the normalization bounds for coloring in the video
    vmin = max(cam_radius-1.0, 0.)
    vmax = vmin + 2.0
    fl = 1.0
    sensor_size = [1.0,1.0]
    resolution = [resolution,resolution]
    angle_increment = 2*math.pi / n_frames
    z_vals = [np.cos(angle_increment*i)*cam_radius for i in range(n_frames)]
    x_vals = [np.sin(angle_increment*i)*cam_radius for i in range(n_frames)]
    circle_cameras_odf = [ODFCamera(center=[x_vals[i],0.0,z_vals[i]], direction=[-x_vals[i],0.0,-z_vals[i]], focal_length=fl, sensor_size=sensor_size, sensor_resolution=resolution) for i in range(n_frames)]
    ODFCamera(center=[0.4,0.,1.], direction=[-0.4,0.,-1.])
    circle_cameras_pyrender = [PyrenderCamera(center=[x_vals[i],0.0,z_vals[i]], direction=[-x_vals[i],0.0,-z_vals[i]], focal_length=fl, sensor_size=sensor_size, sensor_resolution=resolution) for i in range(n_frames)]
    rendered_views_model = [cam.odf_depths(model) for cam in tqdm(circle_cameras_odf)]
    rendered_views_mesh = [cam.render_mesh(mesh_vertices, mesh_faces, obj_mesh) for cam in tqdm(circle_cameras_pyrender)]

    rendered_views_model = [depths_to_img(depths, intersects, min_depth=vmin, max_depth=vmax) for intersects, depths in rendered_views_model]
    rendered_views_mesh = [depths_to_img(depths, intersects, min_depth=vmin, max_depth=vmax) for intersects, depths in rendered_views_mesh]

    save_video(rendered_views_model, rendered_views_mesh, os.path.join(video_dir, f'equatorial_video_{expt_name}.mp4'))

        

if __name__ == "__main__":
    Parser = argparse.ArgumentParser(description='Video generation code for NeuralODFs.')
    Parser.add_argument('--output-dir', type=str, required=True, help='Directory where model files reside and where videos should be written.')
    Parser.add_argument('--mesh-dir', type=str, required=True, help="The directory containin the ground truth meshes.")
    Parser.add_argument('--object', type=str, required=True, help="The name of the object that is being visualized.")
    Parser.add_argument('--expt-name', type=str, required=True, help='The name of the model to load and visualize.')
    Parser.add_argument("--equatorial", action="store_true", help="Render and save a video where the camera orbits on the equator.")
    Parser.add_argument("--no-posenc", action="store_true", help="Don't use positional encoding.")
    Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
    Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
    # Args = Parser.parse_args()
    Args, _ = Parser.parse_known_args()
    
    butils.seedRandom(Args.seed)
    usePosEnc = not Args.no_posenc

    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    odf = ODFSingleV3(input_size=(120 if usePosEnc else 6), radius=DEPTH_SAMPLER_RADIUS, coord_type=Args.coord_type, pos_enc=usePosEnc, n_layers=10)
    odf.setupCheckpoint(Device)
    odf = odf.to(Device)

    ODFCam = ODFCamera(center=[0.4,0.,1.], direction=[-0.4,0.,-1.])
    GTCam = PyrenderCamera(center=[0.4,0.,1.], direction=[0.4,0.,1.])
    intersects, depths = ODFCam.odf_depths(odf, device=Device)
    mesh_vertices, mesh_faces, obj_mesh = load_object(Args.object + ".obj", Args.mesh_dir)
    gt_intersects, gt_depths = GTCam.render_mesh(mesh_vertices, mesh_faces, obj_mesh)

    max_depth = max(np.max(gt_depths), np.max(depths))
    min_depth = min(np.min(gt_depths), np.min(depths))

    img = depths_to_img(depths, intersects, min_depth=min_depth, max_depth=max_depth)
    gt_img = depths_to_img(gt_depths, gt_intersects, min_depth=min_depth, max_depth=max_depth)
    
    # f = plt.figure()
    ax = plt.subplot(121)
    ax.imshow(img)
    ax.set_title("Predicted")
    ax2 = plt.subplot(122)
    ax2.imshow(gt_img)
    ax2.set_title("Ground Truth")
    plt.show()

    equatorial_video(Args.output_dir, Args.expt_name, odf, mesh_vertices, mesh_faces, obj_mesh)


