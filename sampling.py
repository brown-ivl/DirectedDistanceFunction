'''
Sampling functions

TODO:
Types of ray sampling:
Add skinning weights along with depth/occ!
CHECK OTHER PAPERS for how to sample lightfields

Also - add additional training examples by adding d to start point and d to depth
'''
from numpy import random
import rasterization
import utils

import numpy as np
import argparse
import os
import datetime
import matplotlib.pyplot as plt
import trimesh
from scipy.stats import norm, uniform
from tqdm import tqdm


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
    return start_point, end_point, None

def sampling_preset_noise(sampling_method, noise):
    '''
    Defines a new version of one of the sampling functions with a different noise value set
    '''
    def preset_noise(radius, verts=None, vert_normals=None, v=None, **kwargs):
        return sampling_method(radius, verts=verts, noise=noise, vert_normals=vert_normals, v=None, kwargs=kwargs)
    return preset_noise

def uniform_ray_space_equal_intersections():
    '''
    Samples a ray uniformly at random from ray space (two points on surface of sphere). Then, all of the intersections along the ray are found,
    and one of the rays is sampled
    '''
    pass

def sphere_surface_endpoints(radius, n_samples=1):
    '''
    Returns ray endpoints both sampled uniformly from a sphere surface
    '''
    start_point = utils.random_on_sphere(radius)
    end_point = utils.random_on_sphere(radius)
    return start_point, end_point

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Demonstrate ray sampling or run data generation speed tests")
    parser.add_argument("-v", "--viz", action="store_true", help="visualize randomly sampled rays")
    parser.add_argument("-s", "--speed", action="store_true", help="show speed benchmarks for randomly generated rays")
    parser.add_argument("-d", "--depthmap", action="store_true", help="show a depth map image of the mesh")
    parser.add_argument("-c", "--coverage", action="store_true", help="show the intersected vertices of the mesh")
    parser.add_argument("--mesh_file", default="F:\\ivl-data\\sample_data\\stanford_bunny.obj", help="Source of mesh file")
    args = parser.parse_args()

    
    # # base_path = os.path.join("~", "Brown","ivl-research","large_files","sample_data")
    # base_path = "C:\\Users\\Trevor\\Brown\\ivl-research\\large_files\\sample_data"
    # instance = "50002_hips_poses_0694"
    # gender = "male"
    # smpl_data_path = os.path.join(base_path, f"{instance}_smpl.npy")
    # faces_path = os.path.join(base_path, f"{gender}_template_mesh_faces.npy")
    # smpl_data = np.load(smpl_data_path, allow_pickle=True).item()
    # verts = np.array(smpl_data["smpl_mesh_v"])
    # faces = np.array(np.load(faces_path, allow_pickle=True))

    mesh = trimesh.load(args.mesh_file)
    faces = mesh.faces
    verts = mesh.vertices
    
    verts = utils.mesh_normalize(verts)
    radius = 1.25
    fixed_endpoint = 700



    # threshold for how far away a face can be from the ray before it gets culled in rasterization
    near_face_threshold = rasterization.max_edge(verts, faces)
    vert_normals = utils.get_vertex_normals(verts, faces)

    sampling_methods = [sample_uniform_ray_space, sample_vertex_noise, sample_vertex_all_directions, sample_vertex_tangential]
    method_names = ["sample_uniform_ray_space", "sample_vertex_noise", "sample_vertex_all_directions", "sample_vertex_tangential"]
    # sampling_methods = [sample_vertex_all_directions, sample_vertex_tangential]
    # method_names = ["sample_vertex_all_directions", "sample_vertex_tangential"]

    if args.viz:
        import visualization
        lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)
        for i, sampling_method in enumerate(sampling_methods):
            visualizer = visualization.RayVisualizer(verts, lines)
            print(method_names[i])
            for _ in range(100):
                # Sample a ray
                ray_start, ray_end, v = sampling_method(radius, verts=verts, vert_normals=vert_normals, v=fixed_endpoint)
                # rotate and compute depth/occupancy
                rot_verts = rasterization.rotate_mesh(verts, ray_start, ray_end)
                occ, depth, intersected_faces = rasterization.ray_occ_depth_visual(faces, rot_verts, ray_start_depth=np.linalg.norm(ray_end-ray_start), near_face_threshold=near_face_threshold, v=v)
                # update visualizer
                visualizer.add_sample(ray_start, ray_end, occ, depth, faces[intersected_faces] if intersected_faces.shape[0] > 0 else [])
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
                ray_start, ray_end, v = sampling_method(radius, verts=verts, vert_normals=vert_normals, v=None)
                rot_verts = rasterization.rotate_mesh(verts, ray_start, ray_end)
                occ, depth = rasterization.ray_occ_depth(faces, rot_verts, ray_start_depth=np.linalg.norm(ray_end-ray_start), near_face_threshold=near_face_threshold, v=v)
            end = datetime.datetime.now()
            secs = (end-start).total_seconds()
            print(f"\t{n_samples/secs :.0f} rays per second")

    if args.depthmap:
        cam_center = [0.,1.0,1.]
        direction = [0.,-1.0,-1.]
        focal_length = 1.0
        sensor_size = [1.0,1.0]
        resolution = [100,100]

        # uncomment to show u and v vectors
        # direction /= np.linalg.norm(direction)
        # if direction[0] == 0. and direction[2] == 0.:
        #     u_direction = np.array([1.,0.,0.])
        #     v_direction = np.array([0.,0.,1.])*(-1. if direction[1] > 0. else 1.)
        # else:
        #     u_direction = np.cross(direction, np.array([0.,1.,0.]))
        #     v_direction = np.cross(direction, u_direction)
        #     v_direction /= np.linalg.norm(v_direction)
        #     u_direction /= np.linalg.norm(u_direction)

        # lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)
        # visualizer = visualization.RayVisualizer(verts, lines)
        # visualizer.add_point([1.,0.,0.], [1.,0.,0.])
        # visualizer.add_point([0.,1.,0.], [0.,1.,0.])
        # visualizer.add_point([0.,0.,1.], [0.,0.,1.])
        # visualizer.add_ray([cam_center, cam_center+direction/np.linalg.norm(direction)*0.1], np.array([1.,0.,0.]))
        # visualizer.add_ray([cam_center, cam_center+u_direction*0.1], np.array([0.,1.,0.]))
        # visualizer.add_ray([cam_center, cam_center+v_direction*0.1], np.array([0.,0.,1.]))
        # visualizer.display()
        # TODO: use Camera object
        cam = visualization.Camera(center=cam_center, direction=direction, focal_length=focal_length, sensor_size=sensor_size, sensor_resolution=resolution)
        intersection, depth = cam.mesh_depthmap(cam.rays_on_sphere(cam.generate_rays(), radius), verts, faces)
        plt.imshow(depth)
        plt.show()
        plt.imshow(intersection)
        plt.show()

    if args.coverage:
        lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)
        print(f"There are {faces.shape[0]} faces in the mesh")
        print(f"Sampling 10*{faces.shape[0]} rays per method")
        for i, sampling_method in enumerate(sampling_methods):
            visualizer = visualization.RayVisualizer(verts, lines)
            print(method_names[i])
            face_counts = np.zeros(faces.shape[0]).astype(float)
            # Sample 10 rays per face (on average)
            for _ in tqdm(range(10*faces.shape[0])):
                # Sample a ray
                ray_start, ray_end, v = sampling_method(radius, verts=verts, vert_normals=vert_normals, v=None)
                # rotate and compute depth/occupancy
                rot_verts = rasterization.rotate_mesh(verts, ray_start, ray_end)
                occ, depth, intersected_faces = rasterization.ray_occ_depth_visual(faces, rot_verts, ray_start_depth=np.linalg.norm(ray_end-ray_start), near_face_threshold=near_face_threshold, v=None)
                face_counts[intersected_faces.astype(int)] += 1.
            upper_limit = 20.
            upper_color = np.array([0.,1.,0.])
            lower_color = np.array([1.,0.,0.])
            pick_face_color = lambda x: np.array([1.,1.,1.]) if face_counts[x] == 0. else ((min(face_counts[x]/upper_limit, 1.))) * upper_color + (1. - min(face_counts[x]/upper_limit, 1.)) * lower_color
            mesh_verts = np.vstack(verts[faces[i]] for i in range(faces.shape[0]))
            mesh_faces = np.arange(mesh_verts.shape[0]).reshape((-1,3))
            mesh_vert_colors = np.vstack([np.vstack([pick_face_color(x)[np.newaxis, :]]*3) for x in range(faces.shape[0])])
            visualizer.add_colored_mesh(mesh_verts, mesh_faces, mesh_vert_colors)
            visualizer.display()


