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

# common_objects_data_path = '/gpfs/data/ssrinath/human-modeling/datasets/common-3d-test-models/data/'
common_objects_data_path = 'F:\\ivl-data\\common-3d-test-models\\data\\'


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


def main(mesh_vertices, mesh_faces, obj_mesh, obj_name, data_save_dir=None, val_start_counter=1000, num_viewpoints=1200):  
    
    # sample viewpoints on sphere
    sampled_camcenters = sphere_interior_sampler(num_viewpoints)
    sampled_lookvecs = sphere_surface_sampler(num_viewpoints)
    # sampled_camcenters = np.array([[0.7,0.7,0.7],[0.7,0.7,0.7]])
    # sampled_camcenters = np.zeros((2,3))
    # sampled_lookvecs = np.array([[1.0,1.0,1.0],[-1.0,-1.0,-1.0]])
    # sampled_lookvecs /= np.linalg.norm(sampled_lookvecs, axis=1, keepdims=True)


    opengl_camera_pose_matrix, reverse_camera_pose_matrix, viewpoint_samples, look_samples = get_viewpoint_samples_5d(sampled_camcenters, sampled_lookvecs, selected_samples_count=sampled_camcenters.shape[0])
    # reverse_camera_pose_matrix[:,:,1] *= -1. #TODO: Figure this out. Seems like if the up vec is flipped the image should just be flipped but it isn't.

    inside_mesh = obj_mesh.contains(viewpoint_samples)
    #reverse matrix just has negative u,v,w
    

    # pyrender camera looks at negative z direction.
    # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
    trimesh_mesh = trimesh.base.Trimesh(mesh_vertices, mesh_faces)
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)


    line_points = np.asarray([[0., 0., 0.]])
    line_indices = np.asarray([[0, 1]])

    if not os.path.exists(data_save_dir):
        os.makedirs(os.path.join(data_save_dir))
    
    data_split = 'train'
    # start_counter=50
    for iter in range(0, min(num_viewpoints, opengl_camera_pose_matrix.shape[0])): 
        print("iter: {}/{}".format(iter, min(num_viewpoints, opengl_camera_pose_matrix.shape[0])))
        if iter >= val_start_counter:
            data_split = 'val'
        camera_viewpoint_pose = opengl_camera_pose_matrix[iter]
        # if inside_mesh[iter]:
        #     print("INSIDE - ", viewpoint_samples[iter])
        camera_functional_pose = (opengl_camera_pose_matrix[iter] if not inside_mesh[iter] else reverse_camera_pose_matrix[iter]) #reverse camera to get negative depths if we're inside the object
        # print(viewpoint_samples[iter], "viewpoint")
        opengl_camera = pyrender.PerspectiveCamera(yfov=np.pi/2.0, aspectRatio=1, znear=0.00001, zfar=3)
        scene = pyrender.Scene()
        scene.add_node(pyrender.Node(mesh=mesh))
        scene.add(opengl_camera, pose=camera_functional_pose)
        r = pyrender.OffscreenRenderer(256, 256)
        _, depth_map = r.render(scene, flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES)

        if inside_mesh[iter]:
            depth_map = -1. * depth_map
            # flip to account for not negating the right vector in the reverse camera matrix
            depth_map = np.flip(depth_map, axis=1)

        # invalid depth map and unprojected skinning weights have -1 values
        unprojected_3D_points, invalid_depth_mask = unproject(np.copy(depth_map), camera_viewpoint_pose, opengl_camera.get_projection_matrix(), z_near=0.00001)

        # plt.imshow(depth_map)
        # plt.show()

        if inside_mesh[iter]:
            zeros_mask = invalid_depth_mask.reshape(depth_map.shape)
            depth_map[zeros_mask] = 0.


        #if iter == 0:
        #    all_unprojected_points = unprojected_3D_points
        #else:
        #    all_unprojected_points = np.concatenate((all_unprojected_points, unprojected_3D_points), axis=0)

        
        data_dict = {}
        data_dict['depth_map'] = depth_map
        data_dict['camera_viewpoint_pose'] = camera_viewpoint_pose
        data_dict['camera_projection_matrix'] = opengl_camera.get_projection_matrix()
        data_dict['viewpoint'] = viewpoint_samples[iter]
        data_dict['unprojected_normalized_pts'] = unprojected_3D_points
        data_dict['invalid_depth_mask'] = invalid_depth_mask
        if not os.path.exists(os.path.join(data_save_dir, obj_name, 'depth', data_split)):
            os.makedirs(os.path.join(data_save_dir, obj_name, 'depth', data_split))
        print(os.path.join(data_save_dir, obj_name, 'depth', data_split, 'data_{}.npy'.format(str(iter).zfill(4))))
        np.save(os.path.join(data_save_dir, obj_name, 'depth', data_split, 'data_{}.npy'.format(str(iter).zfill(4))), data_dict)

        #     os.system('mkdir -p ' + os.path.join(renderings_save_dir, str(iter).zfill(4)))
        #     np.save(os.path.join(renderings_save_dir, str(iter).zfill(4), 'depth_map.npy'), depth)
        #     np.save(os.path.join(renderings_save_dir, str(iter).zfill(4), 'skinning_map.npy'), skinning_map)
        #     np.save(os.path.join(renderings_save_dir, str(iter).zfill(4), 'camera_pose_data.npy'), camera_pose)
        #     if not os.path.exists(os.path.join(base_save_dir, 'opengl_projection_matrix.npy')):
        #         np.save(os.path.join(base_save_dir, 'opengl_projection_matrix.npy'), camera.get_projection_matrix())


        # line_points = np.concatenate((line_points, camera_pose[:3,3][None]), axis=0)
        # line_indices = np.concatenate((line_indices, np.asarray([[0, line_indices.shape[0]]])), axis=0)

        # o3d.visualization.draw_geometries([
        #     make_pcd(mesh_vertices, color=[1,0.706,0]),
        #     make_line_set(line_points, line_indices),#, per_line_color=np.array([1,0,0])[None]),

        # ])
        # o3d.visualization.draw_geometries([
        #    make_pcd(mesh_vertices, color=[1,0.706, 0]),
        #    make_pcd(all_unprojected_points[all_unprojected_points[:,2]!=-1], color=[0., 0., 1.])
        # ])

# sampled_points = sphere_interior_sampler(10000000, radius=1.00)
# radii = np.linalg.norm(sampled_points, axis=1)
# print(f"Fraction in half-radius sphere: {np.sum(radii < 0.5)/10000000}")

# sampled_points = sphere_surface_sampler(10000000, radius=1.00)
# positive = np.all(sampled_points > 0., axis=1)
# print(f"Fraction in positive octant: {np.sum(positive)/10000000}")


all_obj_files = glob.glob(os.path.join(common_objects_data_path, '*.obj'))
# data_save_dir = '/gpfs/data/ssrinath/human-modeling/datasets/common-3d-test-models/rendered_data_new/'
data_save_dir = 'F:\\ivl-data\\common-3d-test-models\\rendered-data-5d\\'
for obj_file in all_obj_files:
    obj_name = os.path.basename(obj_file).split('.')[0]
    print(obj_name, "obj_name")
    mesh_vertices, mesh_faces, obj_mesh = load_object(obj_name + '.obj')
    is_watertight = len(trimesh.repair.broken_faces(obj_mesh)) == 0
    if not is_watertight:
        print(f"Skipping {obj_name}.obj because it is not watertight")
    else:
        main(mesh_vertices, mesh_faces, obj_mesh, obj_name, data_save_dir)


# import numpy as np
# data = np.load('data_0000.npy', allow_pickle=True).item()
# print(data.keys())
# print(data['unprojected_normalized_pts'].shape)
# print(unprojected_normalized_pts[unprojected_normalized_pts[:,2]!=-1].shape)


