'''
Visualization utility functions using open3d
'''

import open3d as o3d
import numpy as np

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