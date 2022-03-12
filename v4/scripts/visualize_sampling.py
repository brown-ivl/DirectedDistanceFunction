import numpy as np
import argparse
import trimesh
import open3d as o3d
import os, sys
import glob
import torch
import math
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion

FileDirPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(FileDirPath)
sys.path.append(os.path.join(FileDirPath, 'loaders'))


import v3_utils
from depth_sampler_5d import DepthMapSampler


dataset_dir = "F:\\ivl-data\\common-3d-test-models\\rendered-data-5d"



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

def get_intersecting_rays(sampler, surface_point, multiplier=3):
    coordinates, _, _, _ = sampler.additional_positive_intersections(torch.tensor(surface_point[None,:]), multiplier=multiplier, max_perturbation=0.4)
    startpoints = coordinates[:,:3]
    endpoints = np.vstack([surface_point[None,:],]*startpoints.shape[0])
    return startpoints, endpoints

def get_surface_rays(sampler, ray_endpoints, ray_startpoints):
    coordinates, _, depths, _ = sampler.additional_near_surface_points(ray_endpoints, ray_startpoints, max_offset=0.01)
    startpoints = coordinates[:,:3]
    endpoints = startpoints + np.hstack([depths[:,None],]*3)*coordinates[:,3:]
    return endpoints, startpoints


def get_graph_elements_for_samples(img_data, additional_intersections=0, near_surface_threshold=-1., tangent_ratio=0.0):
    sampler = DepthMapSampler(img_data, 10,  NearSurfacePointsOffset=near_surface_threshold, TangentRaysRatio=tangent_ratio, AdditionalIntersections=additional_intersections)
    print("Coordinates")
    print(sampler.Coordinates)
    start_pts = sampler.Coordinates[:,:3].numpy()
    print("Start Points")
    print(start_pts)
    depths = sampler.Depths.numpy()
    depths[sampler.Intersects == 0.0] = 1.
    print("Depths")
    print(depths)
    view_dirs = sampler.Coordinates[:,3:].numpy()
    print("View Dirs")
    print(view_dirs)
    end_pts = start_pts + np.hstack([depths,]*3) * view_dirs
    print("End Points")
    print(end_pts)
    print(np.linalg.norm(end_pts-start_pts, axis=-1))
    intersections = sampler.Intersects.numpy()
    
    vertices = []
    edges = []
    colors = []

    curr_index = 0
    for i in range(start_pts.shape[0]):
        vertices.append(start_pts[i])
        vertices.append(end_pts[i])
        # print(end_pts[i])
        edges.append([curr_index, curr_index+1])
        if intersections[i] == 1:
            colors.append([1.,0.2,0.2])
        else:
            colors.append([0.5,0.5,0.5])
        curr_index += 2
        
    # print(vertices)
    graph_elements = [make_line_set(vertices, edges, colors=colors)]
    graph_elements.append(make_point_cloud(end_pts, colors=[[1.,0.,0.]]*len(end_pts)))
    graph_elements.append(make_point_cloud(start_pts, colors=[[0.,0.,1.]]*len(start_pts)))

    return graph_elements



def get_graph_elements_for_image(img_data, extra_intersecting=False, near_surface=False):
    cam_pose_ray_length = 0.1

    cam_center = img_data["viewpoint"]
    look = img_data["camera_viewpoint_pose"][0:3,2]
    look = look / np.linalg.norm(look) * cam_pose_ray_length
    up = img_data["camera_viewpoint_pose"][0:3,1]
    up = up / np.linalg.norm(up) * cam_pose_ray_length
    right = img_data["camera_viewpoint_pose"][0:3,0]
    right = right / np.linalg.norm(right) * cam_pose_ray_length

    vertices = [cam_center, cam_center+look, cam_center+right, cam_center+up]
    edges = [[0,1], [0,2], [0,3]]
    colors = [[0.,1.,0.], [1.,1.,0.], [0.,0.,1.]]

    intersection_points = []
    sampler = DepthMapSampler(img_data, 100)

    n_shown = 8
    depths = np.array(sampler.Depths[sampler.ValidDepthMask.to(torch.bool).flatten()][:n_shown,...])
    directions = np.array(sampler.Coordinates[sampler.ValidDepthMask.to(torch.bool).flatten()][:n_shown,3:,...])
    unprojected_pts = directions * np.hstack([depths,]*3) + cam_center

    intersecting_color = [1.,0.5,0.5]

    curr_vert_index = 4
    for i in range(unprojected_pts.shape[0]):
        intersection_points.append(unprojected_pts[i])
        vertices.append(unprojected_pts[i])
        edges.append([0, curr_vert_index])
        colors.append(intersecting_color)
        curr_vert_index += 1

    if extra_intersecting:
        extra_intersecting_color = [0.2, 1.0, 0.2]
        multiplier = 3
        for i in range(unprojected_pts.shape[0]):
            startpoints, endpoints = get_intersecting_rays(sampler, unprojected_pts[i], multiplier=multiplier)
            for j in range(startpoints.shape[0]):
                vertices.append(startpoints[j])
                vertices.append(endpoints[j])
                edges.append([curr_vert_index, curr_vert_index+1])
                colors.append(extra_intersecting_color)
                curr_vert_index += 2

    if near_surface:
        near_surface_color = [0.2,0.2,1.0]
        ray_endpoints = unprojected_pts
        ray_startpoints = np.vstack([cam_center[None,:],]*ray_endpoints.shape[0])
        endpoints, startpoints = get_surface_rays(sampler, ray_endpoints, ray_startpoints)
        for i in range(endpoints.shape[0]):
            vertices.append(startpoints[i])
            vertices.append(endpoints[i])
            edges.append([curr_vert_index, curr_vert_index+1])
            colors.append(near_surface_color)
            curr_vert_index += 2

    mask_dim = int(math.sqrt(sampler.NPData['invalid_depth_mask'].shape[0]))
    depth_mask = np.logical_not(np.reshape(sampler.NPData['invalid_depth_mask'], (mask_dim, mask_dim)))
    depth_mask = np.logical_xor(binary_dilation(depth_mask, iterations=3, border_value=0),binary_erosion(depth_mask, iterations=3, border_value=1))
    # depth_mask = binary_dilation(depth_mask, iterations=3, border_value=0)
    ax = plt.subplot()
    ax.imshow(depth_mask)
    plt.show()


    
    graph_elements = [make_line_set(vertices, edges, colors=colors)]
    if len(intersection_points) > 0:
        graph_elements.append(make_point_cloud(intersection_points, colors=[[1.,0.,0.]]*len(intersection_points)))
    

    return graph_elements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show the camera position and object intersections of each depth image")
    parser.add_argument("--data-dir", "-d", type=str, default="F:\\ivl-data\\common-3d-test-models", help="Data directory")
    parser.add_argument("--object", "-o", type=str, default="armadillo", help="Object to view")
    parser.add_argument("--skip-nonintersecting", action="store_true", help="Skip depth images that don't intersect the object")
    parser.add_argument("--interior", action="store_true", help="Only show depth images from the object interior")
    parser.add_argument('--additional-intersections', type=int, default=0, help="Generate extra intersecting rays for each original intersecting ray")
    parser.add_argument('--near-surface-threshold', type=float, default=-1., help="Sample an additional near-surface (within threshold) point for each intersecting ray. No sampling if negative.")
    parser.add_argument('--tangent-ratio', type=float, default=0., help="The proportion of sampled rays that should be roughly tangent to the object.")
    args = parser.parse_args()

    #3D data and rendered data
    mesh_dir = "data"
    rendered_dir = "rendered-data-5d"

    mesh_vertices, _, obj_mesh = v3_utils.load_object(args.object, os.path.join(args.data_dir, mesh_dir))
    
    wireframe = make_line_set(obj_mesh.vertices, obj_mesh.edges, colors=[[0.,0.,0.]]*obj_mesh.edges.shape[0])
    mesh = make_mesh(obj_mesh.vertices, obj_mesh.faces)
    data_files = glob.glob(os.path.join(args.data_dir, rendered_dir, args.object, "depth", "train", '*.npy'))
    data_files.sort()

    for df in data_files:
        depth_data = np.load(df, allow_pickle=True).item()

        if (not args.skip_nonintersecting or np.min(depth_data["invalid_depth_mask"]) < 1.) and (not args.interior or np.min(depth_data["depth_map"]) < 0.):
            # graph_elements = get_graph_elements_for_image(depth_data, extra_intersecting=args.extra_intersecting, near_surface=args.near_surface)
            graph_elements = get_graph_elements_for_samples(depth_data, additional_intersections=args.additional_intersections, near_surface_threshold=args.near_surface_threshold, tangent_ratio=args.tangent_ratio)
            print(graph_elements)
            o3d.visualization.draw_geometries([mesh] + graph_elements)




    # o3d.visualization.draw_geometries(to_show)

