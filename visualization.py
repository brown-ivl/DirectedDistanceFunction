'''
Visualization utility functions using open3d
'''

import open3d as o3d
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import torch

import utils
import rasterization


def make_line_set(verts, lines, colors=None):
    '''
    Returns an open3d line set given vertices, line indices, and optional color
    '''
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(verts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    if colors is not None:
        line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def make_point_cloud(points, colors=None):
    '''
    Returns an open3d point cloud given a list of points and optional colors
    '''
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    return point_cloud

def make_mesh(verts, faces, color=None):
    '''
    Returns an open3d triangle mesh given vertices, mesh indices, and optional color
    '''
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    if color is not None:
        if len(color.shape) == 1:
            mesh.paint_uniform_color(color)
        else:
            mesh.vertex_colors = o3d.utility.Vector3dVector(color)
    return mesh


class RayVisualizer():
    '''
    This class helps visualize rays and their intersection with a 3D mesh object
    '''
    
    def __init__(self, vertices, lines):
        super().__init__()
        self.verts = vertices
        self.wireframe = make_line_set(vertices, lines)
        self.points = []
        self.point_colors = []
        self.ray_points = []
        self.ray_inds = []
        self.ray_colors = []
        self.mesh_inds = []
        self.colored_meshes = []
        # self.mesh_colors = []

    def add_point(self, point, color):
        self.points.append(point)
        self.point_colors.append(color)

    def add_ray(self, ray_points, ray_color):
        self.ray_points += ray_points
        self.ray_inds.append([len(self.ray_points)-2, len(self.ray_points)-1])
        self.ray_colors.append(ray_color)

    def add_mesh_faces(self, faces):
        self.mesh_inds += faces

    def add_colored_mesh(self, verts, faces, colors):
        self.colored_meshes.append(make_mesh(verts, faces, color=colors))


    def add_sample(self, ray_start, ray_end, occ, depth, intersected_faces):
        '''
        Adds all components of a newly sampled ray
        '''
        vec_mag = np.linalg.norm(ray_end-ray_start)
        self.add_point(ray_start, [1.,0.8,0.])
        self.add_point(ray_end, [1.,0.,0.])
        intersection_point = ray_start + (ray_end - ray_start)*depth / vec_mag            
        if np.linalg.norm(intersection_point) < np.inf:
            self.add_point(intersection_point, [0., 1., 0.])
            
            # draw all the way to intersection if it is past ray end
            if np.linalg.norm(intersection_point-ray_start) > vec_mag:
                ray_end = intersection_point
            # color the ray purple if it originates within the mesh
            if occ:
                self.add_ray([ray_start, ray_end], [1., 0., 1.])
            else:
                self.add_ray([ray_start, ray_end], [0., 0., 1.])
        else:
            self.add_ray([ray_start, ray_end], [0.,1.,1.])

        self.add_mesh_faces(list(intersected_faces))

    def display(self):
        to_show = [self.wireframe]
        if len(self.points) > 0:
            to_show.append(make_point_cloud(np.array(self.points), np.array(self.point_colors)))
        if len(self.ray_points) > 0:
            to_show.append(make_line_set(np.array(self.ray_points), np.array(self.ray_inds), np.array(self.ray_colors)))
        if len(self.mesh_inds) > 0:
            to_show.append(make_mesh(self.verts, self.mesh_inds, color=np.array([1.,0.,0.])))
        to_show += self.colored_meshes
        o3d.visualization.draw_geometries(to_show)

class Camera():
    '''
    This class represents a camera and allows view to be rendered either by rasterizing a mesh or by querying a learned network.
        cam_center        - the coordinates of the camera center (x,y,z)
        direction         - a vector defining the direction that the camera is pointing, relative to the camera center
        focal_length      - the focal length of the camera
        sensor_size       - the dimensions of the sensor (u,v)
        sensor_resolution - The number of pixels on each edge of the sensor (u,v)
    '''

    def __init__(self, center=[1.,1.,1.], direction=[-1.,-1.,-1.], focal_length=1.0, sensor_size=[1.,1.], sensor_resolution=[100,100], verbose=True):
        super().__init__()
        self.verbose = verbose
        self.center = np.array(center)
        assert(np.linalg.norm(direction) != 0.)
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.focal_length = focal_length
        self.sensor_size = sensor_size
        self.sensor_resolution = sensor_resolution

    def change_resolution(self, resolution):
        self.sensor_resolution = resolution

    def generate_rays(self):
        '''
        Returns a list of rays ( [start point, end point] ), where each ray intersects one pixel. The start point of each ray is the camera center.
        Rays are returned top to bottom, left to right.
        '''
        if self.direction[0] == 0. and self.direction[2] == 0.:
            u_direction = np.array([1.,0.,0.])
            v_direction = np.array([0.,0.,1.])*(-1. if self.direction[1] > 0. else 1.)
        else:
            # v_direction = np.cross(np.array([0.,0.,1.]), direction)
            # u_direction = np.cross(v_direction, direction)
            u_direction = np.cross(self.direction, np.array([0.,1.,0.]))
            v_direction = np.cross(self.direction, u_direction)
            v_direction /= np.linalg.norm(v_direction)
            u_direction /= np.linalg.norm(u_direction)
        u_steps = np.linspace(-self.sensor_size[0], self.sensor_size[0], num=self.sensor_resolution[0])
        v_steps = np.linspace(-self.sensor_size[1], self.sensor_size[1], num=self.sensor_resolution[1])
        us, vs = np.meshgrid(u_steps, v_steps)
        us = us.flatten()
        vs = vs.flatten()
        rays = [[np.array(self.center), np.array(self.center + self.focal_length * self.direction + us[i]*u_direction) + vs[i]*v_direction] for i in range(us.shape[0])]
        return rays

    def rays_on_sphere(self, rays, radius):
        '''
        Calls generate_rays, but then reformulates each ray so that it starts on the surface of an origin-centered sphere with the provided radius. 
        '''
        first_elt_if_not_none = lambda x: x[0] if x is not None else None
        sphere_surface_rays = [first_elt_if_not_none(utils.get_sphere_intersections(ray[0], ray[1]-ray[0], radius)) if np.linalg.norm(ray[0]) > radius else ray[0] for ray in rays]
        sphere_surface_rays = [[x, x+rays[i][1]-rays[i][0]] if x is not None else None for i,x in enumerate(sphere_surface_rays)]
        # intersection_mask = [1. if intersect is not None else 0. for intersect in intersections]
        return sphere_surface_rays

    def mesh_depthmap(self, rays, verts, faces):
        '''
        Returns an intersection map and a depthmap of a mesh from the camera's perspective
        '''
        near_face_threshold = rasterization.max_edge(verts, faces)
        depth = np.array([rasterization.ray_occ_depth(faces, rasterization.rotate_mesh(verts, ray[0], ray[1]), ray_start_depth=np.linalg.norm(ray[1]-ray[0]), near_face_threshold=near_face_threshold)[1] if ray is not None else np.inf for ray in (tqdm(rays) if self.verbose else rays)])
        depth = np.reshape(depth, tuple(self.sensor_resolution))
        intersection_mask = (depth < np.inf).astype(np.float)
        return np.array(intersection_mask), np.array(depth)

    def model_depthmap(self, rays, model):
        '''
        Returns an intersection map and a depthmap from a learned model from the camera's perspective
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=model.eval()
        rays_in_scene_mask = np.array([True if ray != None else False for ray in rays])
        rays_in_scene = [ray for ray in rays if ray != None]
        # print(rays_in_scene)
        if len(rays_in_scene) > 0:
            with torch.no_grad():
                # angle_rays = torch.tensor([list(ray[0]) + list(utils.vector_to_angles(ray[1]-ray[0])) for ray in rays], dtype=torch.float32).to(device)
                encoded_rays = torch.tensor([[x for val in list(ray[0])+list((ray[1]-ray[0])/np.linalg.norm(ray[1]-ray[0])) for x in utils.positional_encoding(val)] for ray in rays_in_scene]).to(device)
                _, intersect, depth = model(encoded_rays)
                intersect = intersect.cpu()
                model_depths = depth.cpu()
            intersection_mask = rays_in_scene_mask.astype(float)
            intersection_mask[rays_in_scene_mask] = intersect
            intersection_mask = intersection_mask.reshape(self.sensor_resolution)
            depth = np.zeros((len(rays),))
            depth[np.logical_not(rays_in_scene_mask)] = np.inf
            depth[rays_in_scene_mask] = model_depths
            depth = depth.reshape((self.sensor_resolution))
        else:
            intersection_mask = np.zeros(self.sensor_resolution)
            depth = np.ones(self.sensor_resolution) * np.inf
        return np.array(intersection_mask > 0.5), np.array(depth)
        
    def mesh_and_model_depthmap(self, model, verts, faces, radius, show_rays=False):
        '''
        Convenience function that also allows us to generate rays only once for both depthmap generations
        Returns depthmaps and intersections for the mesh and the learned model
        '''
        rays = self.rays_on_sphere(self.generate_rays(), radius)
        if show_rays:
            visualizer = RayVisualizer(verts, np.vstack([faces[:,:2], faces[:,1:], faces[:,[0,2]]]))
            defined_rays= [ray for ray in rays if ray is not None]
            for ray in defined_rays:
                visualizer.add_point(ray[0], [1.,0.,0.])
                visualizer.add_ray(ray, [0.,0.,1.])
            visualizer.display()
        mesh_int_mask, mesh_depth = self.mesh_depthmap(rays, verts, faces)
        model_int_mask, model_depth = self.model_depthmap(rays, model)
        return np.array(mesh_int_mask), np.array(mesh_depth), np.array(model_int_mask), np.array(model_depth)


class DepthMapViewer():
    '''
    Allows for sequential viewing of multiple depth maps
    '''

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.i = 0
        self.fig, ((self.ax1, self.ax2, self.ax3), (self.ax4, self.ax5, self.ax6)) = plt.subplots(2,3)
        self.all_axes = [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]
        self.show_data()

        callback = self
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(callback.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(callback.prev)
        plt.show()

    def add_data(self, new_data):
        self.data.append(new_data)

    def show_data(self):
        gt_intersect, gt_depth, learned_intersect, learned_depth = self.data[self.i]
        depth_learned_mask = np.where(learned_intersect, learned_depth, np.inf)
        for ax in self.all_axes:
            ax.clear()
        self.ax1.imshow(gt_intersect)
        self.ax1.set_title("GT Intersect")
        self.ax2.imshow(gt_depth)
        self.ax2.set_title("GT Depth")
        self.ax4.imshow(learned_intersect)
        self.ax4.set_title("Intersect")
        self.ax5.imshow(depth_learned_mask)
        self.ax5.set_title("Depth (Masked)")
        self.ax6.imshow(learned_depth)
        self.ax6.set_title("Depth")

    def next(self,event):
        if self.i < len(self.data)-1:
            self.i += 1
            self.show_data()
            plt.draw()

    def prev(self, event):
        if self.i > 0:
            self.i -= 1
            self.show_data()
            plt.draw()

def save_video(rendered_views, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation

    n_frames = len(rendered_views)

    f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2)
    all_axes = [ax1,ax2,ax3,ax4]

    # display first view
    gt_intersect, gt_depth, learned_intersect, learned_depth = rendered_views[0]
    depth_learned_mask = np.where(learned_intersect, learned_depth, np.inf)
    ax1.imshow(gt_intersect)
    ax1.set_title("GT Intersect")
    ax2.imshow(gt_depth)
    ax2.set_title("GT Depth")
    ax3.imshow(learned_intersect)
    ax3.set_title("Intersect")
    ax4.imshow(depth_learned_mask)
    ax4.set_title("Depth (Masked)")

    # Set up formatting for movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist="Trevor Houchens"), bitrate=1800)

    def update_depthmap(num, views, axes):
        for ax in axes:
            ax.clear()
        gt_intersect, gt_depth, learned_intersect, learned_depth = rendered_views[num]
        depth_learned_mask = np.where(learned_intersect, learned_depth, np.inf)
        axes[0].imshow(gt_intersect)
        axes[0].set_title("GT Intersect")
        axes[1].imshow(gt_depth)
        axes[1].set_title("GT Depth")
        axes[2].imshow(learned_intersect)
        axes[2].set_title("Intersect")
        axes[3].imshow(depth_learned_mask)
        axes[3].set_title("Depth (Masked)")

    depthmap_ani = animation.FuncAnimation(f, update_depthmap, n_frames, fargs=(rendered_views, all_axes),
                                   interval=50)
    depthmap_ani.save(save_path, writer=writer)