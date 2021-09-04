'''
Sampling functions

TODO:
Types of ray sampling:
Add skinning weights along with depth/occ!
CHECK OTHER PAPERS for how to sample lightfields

Also - add additional training examples by adding d to start point and d to depth
'''
import rasterization
import utils

import numpy as np
import argparse
import os
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform


def sample_uniform_ray_space(radius, **kwargs):
    '''
    Returns a ray that has been sampled uniformly from the 5D ray space of x,y,z,theta,phi
    Sampling Procedure:
        1) Choose a start point uniformly at random within the bounding sphere
        2) Choose a direction on the unit sphere
        3) The end point is the start point plus the direction
    '''
    start_point = utils.random_within_sphere(radius)
    end_point = start_point + utils.random_on_sphere(0.5)
    return start_point, end_point, None

def sample_vertex(radius, verts=None, **kwargs):
    '''
    Randomly selects a ray that starts a point chosen uniformly at random within the unit sphere and ends at
    a randomly chosen mesh vertex. It is recommended to use sample_vertex_noise instead so that the ray end point
    won't be exactly on the vertex
    Sampling Procedure:
        1) Choose a vertex uniformly at random as the endpoint
        2) Choose a start point uniformly at random within the bounding sphere
    '''
    assert(verts is not None)
    start_point = utils.random_within_sphere(radius)
    v = np.random.randint(0, high=verts.shape[0])
    end_point = verts[v]
    return start_point, end_point, v

def sample_vertex_noise(radius, verts=None, noise = 0.01, **kwargs):
    '''
    Returns a ray that has an endpoint near a vertex, and a start point that is uniformly chosen from within a sphere 
    Compared to sample_vertex, this has the advantage of being unlikely to intersect multiple faces, as well as providing training
    examples close to the edge of the object, and not directly on the edge.
    Sampling Procedure:
        1) Choose a vertex uniformly at random as the endpoint
        2) Modify the endpoint by adding gaussian noise with sigma defined by the noise parameter
        3) Choose a start point uniformly at random within the bounding sphere
    '''
    assert(verts is not None)
    start_point = utils.random_within_sphere(radius)
    v = np.random.randint(0, high=verts.shape[0])
    end_point = verts[v] + norm.rvs(scale=noise, size=3)
    return start_point, end_point, None


def sample_vertex_all_directions(radius, verts=None, noise = 0.01, v=None, **kwargs):
    '''
    Like sample_vertex_noise, but samples uniformly over viewing direction, which is more uniform over the 4d lightfield.
    Sampling Procedure:
        1) Choose a vertex uniformly at random as the endpoint
        2) Modify the endpoint by adding gaussian noise with sigma defined by the noise parameter
        3) Choose a direction on the unit sphere
        4) Find the ray that goes in the chosen direction from the endpoint, and find where it intersects the bounding sphere
        5) Uniformly at random choose a point between the two sphere intersections and make that the start point
    v can be passed to fix the end vertex for visualization purposes
    '''
    assert(verts is not None)
    if v is None:
        v = np.random.randint(0, high=verts.shape[0])
    end_point = verts[v] + norm.rvs(scale=noise, size=3)
    direction = utils.random_on_sphere(1.0)
    bound1, bound2 = utils.get_sphere_intersections(end_point, direction, radius)
    position = uniform.rvs()
    start_point = bound1* position + (1.-position) * bound2
    return start_point, end_point, None

def sample_vertex_tangential(radius, verts=None, noise=0.01, vert_normals=None, v=None):
    '''
    Returns a ray that has an endpoint near a mesh vertex, and has a start point that is orthogonal to the 
    vertex normal (tangential)
    Sampling Procedure:
        1) Choose a vertex uniformly at random as the endpoint
        2) Modify the endpoint by adding gaussian noise with sigma defined by the noise parameter
        3) Choose a direction on the unit sphere
        4) Cross the direction with the vertex normal to get a direction tangential to the vertex normal
        5) Find the ray that goes in the chosen tangent direction from the endpoint, and find where it intersects the bounding sphere
        6) Uniformly at random choose a point between the two sphere intersections and make that the start point
    v can be passed to fix the end vertex for visualization purposes
    '''
    assert(vert_normals is not None and verts is not None)
    if v is None:
        v = np.random.randint(0, high=verts.shape[0])
    end_point = verts[v] + norm.rvs(scale=noise, size=3)
    v_normal = vert_normals[v]
    direction = np.cross(v_normal, utils.random_on_sphere(1.0))
    bound1, bound2 = utils.get_sphere_intersections(end_point, direction, radius)
    position = uniform.rvs()
    start_point = bound1*position + (1.-position)*bound2
    # start_point = verts[v] + direction + norm.rvs(scale=noise, size=3)
    return start_point, end_point, None
    # return bound1, bound2


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Demonstrate ray sampling or run data generation speed tests")
    parser.add_argument("-v", "--viz", action="store_true", help="visualize randomly sampled rays")
    parser.add_argument("-s", "--speed", action="store_true", help="show speed benchmarks for randomly generated rays")
    parser.add_argument("-d", "--depthmap", action="store_true", help="show a depth map image of the mesh")
    args = parser.parse_args()

    
    # base_path = os.path.join("~", "Brown","ivl-research","large_files","sample_data")
    base_path = "C:\\Users\\Trevor\\Brown\\ivl-research\\large_files\\sample_data"
    instance = "50002_hips_poses_0694"
    gender = "male"
    smpl_data_path = os.path.join(base_path, f"{instance}_smpl.npy")
    faces_path = os.path.join(base_path, f"{gender}_template_mesh_faces.npy")

    smpl_data = np.load(smpl_data_path, allow_pickle=True).item()
    verts = np.array(smpl_data["smpl_mesh_v"])
    faces = np.array(np.load(faces_path, allow_pickle=True))
    verts = utils.mesh_normalize(verts)
    radius = 1.25
    fixed_endpoint = 700



    # threshold for how far away a face can be from the ray before it gets culled in rasterization
    near_face_threshold = rasterization.max_edge(verts, faces)
    vert_normals = utils.get_vertex_normals(verts, faces)

    sampling_methods = [sample_uniform_ray_space, sample_vertex_noise, sample_vertex_all_directions, sample_vertex_tangential]
    method_names = ["sample_uniform_ray_space", "sample_vertex_noise", "sample_vertex_all_directions", "sample_vertex_tangential"]

    if args.viz:
        import visualization
        lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)
        for i, sampling_method in enumerate(sampling_methods):
            visualizer = visualization.RayVisualizer(verts, lines)
            print(method_names[i])
            for _ in range(1):
                # Sample a ray
                ray_start, ray_end, v = sampling_method(radius, verts=verts, vert_normals=vert_normals, v=fixed_endpoint)
                # rotate and compute depth/occupancy
                rot_verts = rasterization.rotate_mesh(verts, ray_start, ray_end)
                occ, depth, intersected_faces = rasterization.ray_occ_depth_visual(faces, rot_verts, ray_start_depth=np.linalg.norm(ray_end-ray_start), near_face_threshold=near_face_threshold, v=v)
                # update visualizer
                visualizer.add_sample(ray_start, ray_end, occ, depth, intersected_faces)
            visualizer.add_point([1.,0.,0.], [1.,0.,0.])
            visualizer.add_point([0.,1.,0.], [0.,1.,0.])
            visualizer.add_point([0.,0.,1.], [0.,0.,1.])
            visualizer.display()
        
    if args.speed:
        n_samples = 1000
        print(f"Generating {n_samples} samples per test")
        for i, sampling_method in enumerate(sampling_methods):
            print(method_names[i])
            start = datetime.datetime.now()
            for _ in range(n_samples):
                ray_start, ray_end, v = sampling_method(radius, verts=verts, vert_normals=vert_normals, v=fixed_endpoint)
                rot_verts = rasterization.rotate_mesh(verts, ray_start, ray_end)
                occ, depth = rasterization.ray_occ_depth(faces, rot_verts, ray_start_depth=np.linalg.norm(ray_end-ray_start), near_face_threshold=near_face_threshold, v=v)
            end = datetime.datetime.now()
            secs = (end-start).total_seconds()
            print(f"\t{n_samples/secs :.0f} rays per second")

    if args.depthmap:
        cam_center = [1.0,0.0,1.0]
        direction = [-1.0,-0.0,-1.0]
        focal_length = 1.5
        sensor_size = [1.0,1.0]
        resolution = [250,250]

        # rays = utils.camera_view_rays([1.3,1.3,0.1], [-1.3,-1.3,-0.1], 1.5, [1.0,1.0], resolution)
        # depths = np.array([rasterization.ray_occ_depth(faces, rasterization.rotate_mesh(verts, ray[0], ray[1]), ray_start_depth=np.linalg.norm(ray[1]-ray[0]), near_face_threshold=near_face_threshold)[1] for ray in rays])
        # depths = np.reshape(depths, tuple(resolution))
        intersection, depth = rasterization.camera_ray_depth(verts, faces, cam_center, direction, focal_length, sensor_size, resolution, near_face_threshold=near_face_threshold)
        plt.imshow(depth)
        plt.show()
        plt.imshow(intersection)
        plt.show()


