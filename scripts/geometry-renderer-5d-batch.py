# %%
import numpy as np
import open3d as o3d
import trimesh
import os, glob
from scipy.spatial import KDTree
import random
import math
import torch
import torch.nn as nn
import pyrender
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as scipy_rotation
import math

# %%
# open3d utils

def make_pcd(point_cloud, color=None, per_vertex_color=None, estimate_normals=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    if color is not None:
        # pcd.colors = o3d.utility.Vector3dVector(color)
        pcd.paint_uniform_color(color)
    if per_vertex_color is not None:
        pcd.colors = o3d.utility.Vector3dVector(per_vertex_color)
    
    if estimate_normals:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                radius=0.05, max_nn=100))
    return pcd

def make_line_set(points, edges, line_color = None, per_line_color=None):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    if per_line_color is not None:
        line_set.colors = o3d.utility.Vector3dVector(per_line_color)
    
    return line_set

def make_mesh(vertices, faces):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()
    
    return o3d_mesh

# open3d                    0.13.0                   py38_0    open3d-admin


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

# %%
## opengl utils

def normalize(vector):
    return vector / np.linalg.norm(vector, axis=1)[:, None]

def compute_opengl_rotation_matrix(viewpoints, camera_directions=None, scene_origin=np.asarray([[0.,0.,0.]])):
    if camera_directions is None:
        camera_directions = viewpoints - scene_origin
        camera_directions = normalize(camera_directions)
    else:
        assert(camera_directions.shape[0] == viewpoints.shape[0])
    
    up_direction = np.asarray([[0., 1., 0.]])
    right_x_direction = normalize(np.cross(up_direction, camera_directions))
    
    up_direction = normalize(np.cross(camera_directions, right_x_direction))
    
    camera_pose_matrix = np.concatenate((right_x_direction[:,:,None],
                                    up_direction[:,:,None],
                                    camera_directions[:,:,None],
                                    viewpoints[:,:,None]), axis=2)
    
    camera_pose_matrix = np.concatenate((camera_pose_matrix, 
             np.repeat(np.asarray([[0., 0., 0., 1.]])[None], camera_pose_matrix.shape[0], axis=0)), axis=1)
    return camera_pose_matrix

def get_pose_and_reverse_pose(viewpoints, camera_directions=None):
    camera_poses = compute_opengl_rotation_matrix(viewpoints, camera_directions=camera_directions)
    reverse_poses = np.copy(camera_poses)
    reverse_poses[:,:3,1:3] = reverse_poses[:,:3,1:3] * -1.
    # print("++++"*10)
    # print(camera_poses[0])
    # print(reverse_poses[0])
    return camera_poses, reverse_poses

def get_viewpoint_samples_5d(camcenters, lookvecs, selected_samples_count=50):
    
    sample_indices = np.arange(camcenters.shape[0])
    # np.random.shuffle(viewpoint_samples_indices)
    
    center_samples = camcenters[sample_indices[:selected_samples_count]]
    look_samples = lookvecs[sample_indices[:selected_samples_count]]
    print(center_samples.shape)
    
    
    # ray_directions = all_viewpoints / np.linalg.norm(all_viewpoints, axis=1)[:,None]
    # ray_directions = scipy_rotation.from_euler('zyx',ray_directions)
    # ray_directions = scipy_rotation.as_matrix(ray_directions)
    opengl_camera_pose_matrix, reverse_camera_pose_matrix = get_pose_and_reverse_pose(center_samples, camera_directions=look_samples)
    # print(ray_directions.shape)
    return opengl_camera_pose_matrix, reverse_camera_pose_matrix, center_samples, look_samples#all_viewpoints, ray_directions


# rendering
# example code: https://github.com/silence401/pyrender/blob/e30f7866bc233f4b573ac39d9f371d59a720276d/examples/get_pose.py#L26
def unproject(depth_map, camera_pose, opengl_projective_matrix, z_near=0.05, z_far=3, width=256, height=256):
    
    # projection matrix converts view coordinates to NDC: https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#reference-camera
    # opengl_projective_matrix converts view space to NDC space. So to unproject, we first need to convert points in screen/ image space to NDC space
    # transform image_space depth map to normalized device coordinates depth map
    # https://github.com/mmatl/pyrender/blob/4a289a6205c5baa623cd0e7da1be3d898bcbc4da/pyrender/renderer.py#L274
    
    depth_map = depth_map.reshape(-1, 1)
    y_coord, x_coord = np.indices((height, width))
    invalid_depth_mask = (depth_map==0)
    depth_map = depth_map.reshape(-1, 1)
    x_coord = x_coord.reshape(-1, 1)
    y_coord = y_coord.reshape(-1, 1)
    depth_map[invalid_depth_mask] = 2
#     depth_map = depth_map[valid_depth][:, None]
#     x_coord = x_coord[valid_depth][:, None]
#     y_coord = y_coord[valid_depth][:, None]
    
    depth_map_NDC = ((z_far + z_near)* depth_map - (2*z_far*z_near)) / (depth_map * (z_far - z_near))
    
    # convert x and y coordinates from image space to NDC coordinate space.
    # mentioned in https://docs.microsoft.com/en-us/windows/win32/opengl/glviewport
    x_lower = 0 
    y_lower = 0 
    x_coord_NDC = ((x_coord - x_lower) / (width/ 2.0))  -1
    y_coord_NDC = ((y_coord - y_lower) / (height/ 2.0)) -1
    
    # NDC is in left hand coordinate system: https://learnopengl.com/Getting-started/Coordinate-Systems
    homogeneous_coord_NDC = np.concatenate((
        x_coord_NDC,
        -1.0*y_coord_NDC,
        depth_map_NDC,
        np.ones(depth_map_NDC.shape)
    ), axis=1)
    
    
    homogeneous_coord_view = np.dot(np.linalg.inv(opengl_projective_matrix), homogeneous_coord_NDC.transpose())
    homogeneous_coord_world = np.dot(camera_pose, homogeneous_coord_view).transpose()
    homogeneous_coord_world = homogeneous_coord_world[:,:3] / homogeneous_coord_world[:,3:]
    # homogeneous_coord_world[:,2:][invalid_depth_mask] = -1 
    
    return homogeneous_coord_world, invalid_depth_mask

# uniform sphere sampling: https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
def sphere_sampler(num_points, radius=1.25):
    phi = np.random.uniform(0,2*np.pi,size=num_points)
    costheta = np.random.uniform(-1,1,size=num_points)
    theta = np.arccos(costheta)

    x = radius * np.sin( theta) * np.cos( phi )
    y = radius * np.sin( theta) * np.sin( phi )
    z = radius * np.cos( theta )
    return np.concatenate((x[:,None], y[:,None], z[:,None]), axis=-1)

# uniform sampling of sphere volume: https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
def sphere_interior_sampler(num_points, radius=1.25):
    phi = np.random.uniform(0,2*np.pi,size=num_points)
    costheta = np.random.uniform(-1,1,size=num_points)
    theta = np.arccos(costheta)
    u = np.random.uniform(0,1,size=num_points)
    r = radius * np.cbrt(u)

    x = r * np.sin( theta) * np.cos( phi )
    y = r * np.sin( theta) * np.sin( phi )
    z = r * np.cos( theta )
    return np.concatenate((x[:,None], y[:,None], z[:,None]), axis=-1)

# uniform sampling from sphere surface by drawing from normal distribution
def sphere_surface_sampler(num_points, radius=1.00):
    normal_samples = np.random.normal(size=(num_points,3))
    surface_samples = normal_samples / np.stack((np.linalg.norm(normal_samples,axis=1),)*3, axis=1) * radius
    return surface_samples


def main(mesh_vertices, mesh_faces, obj_mesh, obj_name, data_save_dir=None, train_size=20000, val_size=200):  
    trimesh_mesh = trimesh.base.Trimesh(mesh_vertices, mesh_faces)
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)

    if not os.path.exists(data_save_dir):
        os.makedirs(os.path.join(data_save_dir))
    
    generate_data(batch=10000, data_split='train', mesh=mesh, start=0, size=train_size)

    generate_data(batch=10000, data_split='val', mesh=mesh, start=train_size, size=val_size)

def generate_data(batch, data_split, mesh, start, size):
    
    inside_counter = 0
    outside_counter = 0
    while inside_counter+outside_counter<size:
        num_viewpoints = batch
        # sample viewpoints on sphere
        sampled_camcenters = sphere_interior_sampler(num_viewpoints)
        sampled_lookvecs = sphere_surface_sampler(num_viewpoints)

        opengl_camera_pose_matrix, reverse_camera_pose_matrix, viewpoint_samples, look_samples = get_viewpoint_samples_5d(sampled_camcenters, sampled_lookvecs, selected_samples_count=sampled_camcenters.shape[0])

        inside_mesh = obj_mesh.contains(viewpoint_samples)
        #reverse matrix just has negative u,v,w
    

        # pyrender camera looks at negative z direction.
        # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
        for i in range(num_viewpoints): 
            camera_viewpoint_pose = opengl_camera_pose_matrix[i]
            camera_functional_pose = (opengl_camera_pose_matrix[i] if not inside_mesh[i] else reverse_camera_pose_matrix[i]) #reverse camera to get negative depths if we're inside the object
    
            opengl_camera = pyrender.PerspectiveCamera(yfov=np.pi/2.0, aspectRatio=1, znear=0.00001, zfar=3)
            scene = pyrender.Scene()
            scene.add_node(pyrender.Node(mesh=mesh))
            scene.add(opengl_camera, pose=camera_functional_pose)
            r = pyrender.OffscreenRenderer(256, 256)
            _, depth_map = r.render(scene, flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES)
            
            if inside_mesh[i]:
                if inside_counter>=math.floor(size/2):
                    continue
                depth_map = -1. * depth_map
                # flip to account for not negating the right vector in the reverse camera matrix
                depth_map = np.flip(depth_map, axis=1)
            elif outside_counter>=math.ceil(size/2):
                continue
            # invalid depth map and unprojected skinning weights have -1 values
            unprojected_3D_points, invalid_depth_mask = unproject(np.copy(depth_map), camera_viewpoint_pose, opengl_camera.get_projection_matrix(), z_near=0.00001)

            if inside_mesh[i]:
                zeros_mask = invalid_depth_mask.reshape(depth_map.shape)
                depth_map[zeros_mask] = 0.
            
            # update corresponding counter if we save data
            if inside_mesh[i]:
                inside_counter += 1
            else:
                outside_counter += 1

            print("iter: {}/{}, inside:{}, outside:{}".format(inside_counter+outside_counter, size, inside_counter, outside_counter))
            data_dict = {}
            data_dict['depth_map'] = depth_map
            data_dict['camera_viewpoint_pose'] = camera_viewpoint_pose
            data_dict['camera_projection_matrix'] = opengl_camera.get_projection_matrix()
            data_dict['viewpoint'] = viewpoint_samples[i]
            data_dict['unprojected_normalized_pts'] = unprojected_3D_points
            data_dict['invalid_depth_mask'] = invalid_depth_mask
            if not os.path.exists(os.path.join(data_save_dir, obj_name, 'depth', data_split)):
                os.makedirs(os.path.join(data_save_dir, obj_name, 'depth', data_split))
            #print(os.path.join(data_save_dir, obj_name, 'depth', data_split, 'data_{}.npy'.format(str(inside_counter+outside_counter+start).zfill(4))))
            np.save(os.path.join(data_save_dir, obj_name, 'depth', data_split, 'data_{}.npy'.format(str(inside_counter+outside_counter+start-1).zfill(5))), data_dict)




common_objects_data_path = '/home/johnny/Documents/DDF/DirectedDistanceFunction/common-3d-test-models/data/'
all_obj_files = glob.glob(os.path.join(common_objects_data_path, '*.obj'))
data_save_dir = '/home/johnny/Documents/DDF/DirectedDistanceFunction/common-3d-test-models/rendered-data-5d/new/'
for obj_file in all_obj_files:
    obj_name = os.path.basename(obj_file).split('.')[0]
    if obj_name in ["bunny_watertight"]:
        print(obj_name, "obj_name")
        mesh_vertices, mesh_faces, obj_mesh = load_object(obj_name + '.obj', common_objects_data_path)
        is_watertight = len(trimesh.repair.broken_faces(obj_mesh)) == 0
        if not is_watertight:
            print(f"Skipping {obj_name}.obj because it is not watertight")
        else:
            main(mesh_vertices, mesh_faces, obj_mesh, obj_name, data_save_dir)
