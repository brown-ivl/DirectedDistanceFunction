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
import argparse
from data import DepthData
import odf_utils
import sampling
from torch.utils.data import DataLoader

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

# %%
# data loading

#common_objects_data_path = '/home/ubuntu/'
# common_objects_data_path = 'F:\\ivl-data\\common-3d-test-models\\data\\'
common_objects_data_path = '/gpfs/data/ssrinath/neural-odf/data/common-3d-test-models/data/'

def load_object(obj_name, data_path=common_objects_data_path):
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


def main(verts, faces, obj_mesh, obj_name, data_save_dir=None, train_size=2, val_size=10, camcenters=None, lookvecs=None, show=False):  

    
    vert_noise = 0.01
    tan_noise = 0.01
    sampling_methods = [sampling.sample_uniform_ray_space, 
                        sampling.sampling_preset_noise(sampling.sample_vertex_noise, 
                                                       vert_noise),
                        sampling.sampling_preset_noise(sampling.sample_vertex_all_directions, 
                                                       vert_noise),
                        sampling.sampling_preset_noise(sampling.sample_vertex_tangential, 
                                                       tan_noise)]
    sampling_frequency = [0.0, 0.0, 1.0, 0.0]
    test_sampling_frequency = [1., 0., 0., 0.]
    radius = 1.25
    pool_size = 65536
    train_data = DepthData(faces,
                           verts,
                           radius,
                           sampling_methods,
                           sampling_frequency,
                           size=pool_size*train_size)
    test_data = DepthData(faces,
                          verts,
                          radius,
                          sampling_methods,
                          test_sampling_frequency,
                          size=pool_size*val_size)
    train_loader = DataLoader(train_data, 
                              batch_size=pool_size, 
                              shuffle=False, 
                              drop_last=False)
    test_loader = DataLoader(test_data, 
                             batch_size=pool_size, 
                             shuffle=False, 
                             drop_last=False)

    for i, (data, target) in enumerate(train_loader):
        #print("data", data)
        data_split = "train"
        depth_map = target[1].cpu().numpy().reshape(-1, 1)
        camera_viewpoint_pose = None
        camera_projection_matrix = None
        viewpoint = data.cpu().numpy().reshape(-1, 6)[:, :3]
        unprojected_normalized_pts = data.cpu().numpy().reshape(-1, 6)[:, 3:]
        invalid_depth_mask = target[0].to(bool).cpu().numpy().reshape(-1, 1)
        #print(depth_map)
        #print(viewpoint)
        #print(unprojected_normalized_pts)
        #print(invalid_depth_mask)
        #print("target", target)
        data_dict = {}
        data_dict['depth_map'] = depth_map
        data_dict['camera_viewpoint_pose'] = camera_viewpoint_pose
        data_dict['camera_projection_matrix'] = camera_projection_matrix
        data_dict['viewpoint'] = viewpoint
        data_dict['unprojected_normalized_pts'] = unprojected_normalized_pts
        data_dict['invalid_depth_mask'] = invalid_depth_mask
        if not os.path.exists(os.path.join(data_save_dir, obj_name, 'depth', data_split)):
            os.makedirs(os.path.join(data_save_dir, obj_name, 'depth', data_split))
        print(os.path.join(data_save_dir, obj_name, 'depth', data_split, 'data_{}.npy'.format(str(i).zfill(4))))
        np.save(os.path.join(data_save_dir, obj_name, 'depth', data_split, 'data_{}.npy'.format(str(i).zfill(4))), data_dict)


    for i, (data, target) in enumerate(test_loader):
        #print("data", data)
        data_split = "val"
        depth_map = target[1].cpu().numpy().reshape(-1, 1)
        camera_viewpoint_pose = None
        camera_projection_matrix = None
        viewpoint = data.cpu().numpy().reshape(-1, 6)[:, :3]
        unprojected_normalized_pts = data.cpu().numpy().reshape(-1, 6)[:, 3:]
        invalid_depth_mask = target[0].to(bool).cpu().numpy().reshape(-1, 1)
        #print(depth_map)
        #print(viewpoint)
        #print(unprojected_normalized_pts)
        #print(invalid_depth_mask)
        #print("target", target)
        data_dict = {}
        data_dict['depth_map'] = depth_map
        data_dict['camera_viewpoint_pose'] = camera_viewpoint_pose
        data_dict['camera_projection_matrix'] = camera_projection_matrix
        data_dict['viewpoint'] = viewpoint
        data_dict['unprojected_normalized_pts'] = unprojected_normalized_pts
        data_dict['invalid_depth_mask'] = invalid_depth_mask
        if not os.path.exists(os.path.join(data_save_dir, obj_name, 'depth', data_split)):
            os.makedirs(os.path.join(data_save_dir, obj_name, 'depth', data_split))
        print(os.path.join(data_save_dir, obj_name, 'depth', data_split, 'data_{}.npy'.format(str(i+train_size).zfill(4))))
        np.save(os.path.join(data_save_dir, obj_name, 'depth', data_split, 'data_{}.npy'.format(str(i+train_size).zfill(4))), data_dict)



all_obj_files = glob.glob(os.path.join(common_objects_data_path, '*.obj'))
# data_save_dir = '/gpfs/data/ssrinath/human-modeling/datasets/common-3d-test-models/rendered_data_new/'
#common_objects_data_path = '/home/johnny/Documents/DDF/DirectedDistanceFunction/common-3d-test-models/data/'
data_save_dir = '/gpfs/data/ssrinath/neural-odf/data/common-3d-test-models/rendered-data-5d'
#data_save_dir = '/home/ubuntu/data/new/'
for obj_file in all_obj_files:
    obj_name = os.path.basename(obj_file).split('.')[0]
    if obj_name in ["shirt"]:
        print(obj_name, "obj_name")
        mesh_vertices, mesh_faces, obj_mesh = load_object(obj_name + '.obj')
        #is_watertight = len(trimesh.repair.broken_faces(obj_mesh)) == 0
        #if not is_watertight:
        #    print(f"Skipping {obj_name}.obj because it is not watertight")
        #else:
        main(mesh_vertices, mesh_faces, obj_mesh, obj_name, data_save_dir)
