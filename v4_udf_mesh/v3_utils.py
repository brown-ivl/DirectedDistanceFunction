import argparse
from ftplib import all_errors
import torch
import numpy as np
import os
import glob
import sys
import requests
import matplotlib.pyplot as plt
import math
import trimesh
import random
from scipy.stats import norm, uniform

#####################################################
###################### SETUP ########################
#####################################################

INTERSECTION_MASK_THRESHOLD = 0.5

BaselineParser = argparse.ArgumentParser(description='Parser for NeuralODFs.')
BaselineParser.add_argument('--expt-name', help='Provide a name for this experiment.')
BaselineParser.add_argument('--input-dir', help='Provide the input directory where datasets are stored.')
BaselineParser.add_argument('--dataset', help='The dataset')
BaselineParser.add_argument('--output-dir', help='Provide the *absolute* output directory where checkpoints, logs, and other output will be stored (under expt_name).')
BaselineParser.add_argument('--arch', help='Architecture to use.', choices=['standard', 'constant', 'SH', 'SH_constant'], default='standard')
BaselineParser.add_argument('--use-posenc', help='Choose to use positional encoding.', action='store_true', required=False)
BaselineParser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
BaselineParser.add_argument('--degrees', help='degree for [depth, intersect] or [depth, intersect, const, const mask]', type=lambda ds:[int(d) for d in ds.split(',')], required=False, default=[2, 2])


def seedRandom(seed):
    # NOTE: This gets us very close to deterministic but there are some small differences in 1e-4 and smaller scales
    print('[ INFO ]: Seeding RNGs with {}'.format(seed))
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#####################################################
################# LOADING/SAVING ####################
#####################################################
    

def checkpoint_filename(name, epoch):
    '''
    Returns the filename of a checkpoint for a specific epoch
    '''
    filename = f"{name}_checkpoint_{epoch:06}.tar"
    return filename

def load_checkpoint(save_dir, name, device="cpu", load_best=False):
    '''
    Load a model checkpoint
    if load_best is True, the best model checkpoint will be loaded instead of 
    '''
    if not load_best:
        checkpoint_dir = os.path.join(save_dir, name, "checkpoints")
        all_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.tar"))
        if len(all_checkpoints) == 0:
            raise Exception(f"There are no saved checkpoints for {name}")
        all_checkpoints.sort()
        print(f"Loading checkpoint {os.path.basename(all_checkpoints[-1])}")
        return torch.load(all_checkpoints[-1], map_location=device)
    else:
        checkpoint_dir = os.path.join(save_dir, name, "checkpoints")
        best_file = os.path.join(save_dir, name, "best_checkpoint.txt")
        if not os.path.exists(best_file):
            raise Exception(f"Could not identify the best checkpoint. {best_file} was not found.")
        f = open(best_file, "r")
        best_epoch = int(f.read().split("$")[0])
        f.close()
        checkpoint_file = os.path.join(checkpoint_dir, checkpoint_filename(name, best_epoch))
        print(f"Loading checkpoint {os.path.basename(checkpoint_file)}")
        return torch.load(checkpoint_file, map_location=device)

def save_checkpoint(save_dir, checkpoint_dict):
    assert("name" in checkpoint_dict)
    assert("epoch" in checkpoint_dict)
    name = checkpoint_dict["name"]
    epoch = checkpoint_dict["epoch"]
    if not os.path.exists(os.path.join(save_dir, name)):
        os.mkdir(os.path.join(save_dir, name))
    if not os.path.exists(os.path.join(save_dir, name, "checkpoints")):
        os.mkdir(os.path.join(save_dir, name, "checkpoints"))
    out_file = os.path.join(save_dir, name, "checkpoints", checkpoint_filename(name, epoch))
    torch.save(checkpoint_dict, out_file)

    # check if this is the best checkpointt
    best_file = os.path.join(save_dir, name, "best_checkpoint.txt")
    is_best = False
    if "val" in checkpoint_dict["loss_history"]:
        if not os.path.exists(best_file):
            is_best = True
        else:
            f = open(best_file, "r+")
            best_val = float(f.read().split("$")[1])
            if checkpoint_dict["loss_history"]["val"][-1] < best_val:
                is_best = True
            f.close()
    if is_best:
        f = open(best_file, "w")
        f.write(f"{epoch}${checkpoint_dict['loss_history']['val'][-1]}")
        f.close()

def build_checkpoint(model, name, epoch, optimizer, loss_history):
    checkpoint_dict = {
        'name': name,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
    }
    return checkpoint_dict

def checkpoint(model, save_dir, name, epoch, optimizer, loss_history):
    checkpoint_dict = build_checkpoint(model, name, epoch, optimizer, loss_history)
    save_checkpoint(save_dir, checkpoint_dict)

def expandTilde(Path):
    if '~' == Path[0]:
        return os.path.expanduser(Path)

    return Path

def downloadFile(url, filename, verify=True):
    with open(expandTilde(filename), 'wb') as f:
        response = requests.get(url, stream=True, verify=verify)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}>{}]'.format('=' * done, '-' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')


#####################################################
#################### SAMPLING #######################
#####################################################

def sphere_interior_sampler(num_points, radius=1.25):
    '''
    Uniform sampling of sphere volume: https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
    '''
    phi = np.random.uniform(0,2*np.pi,size=num_points)
    costheta = np.random.uniform(-1,1,size=num_points)
    theta = np.arccos(costheta)
    u = np.random.uniform(0,1,size=num_points)
    r = radius * np.cbrt(u)

    x = r * np.sin( theta) * np.cos( phi )
    y = r * np.sin( theta) * np.sin( phi )
    z = r * np.cos( theta )
    return np.concatenate((x[:,None], y[:,None], z[:,None]), axis=-1)

def sphere_surface_sampler(num_points, radius=1.00):
    '''
    uniform sampling from sphere surface by drawing from normal distribution
    '''
    normal_samples = np.random.normal(size=(num_points,3))
    surface_samples = normal_samples / np.stack((np.linalg.norm(normal_samples,axis=1),)*3, axis=1) * radius
    return surface_samples

def odf_domain_sampler(n_points, radius=1.25):
    '''
    Samples points uniformly at random from an ODF input domain
    '''
    # sample viewpoints on sphere
    sampled_positions = sphere_interior_sampler(n_points, radius=radius)
    sampled_directions = sphere_surface_sampler(n_points)

    coords = np.concatenate([sampled_positions, sampled_directions], axis=-1)
    return coords

#####################################################
###################### MESHES #######################
#####################################################

def load_object(obj_name, data_path):
    obj_file = os.path.join(data_path, f"{obj_name}.obj")

    obj_mesh = trimesh.load(obj_file)
    # obj_mesh.show()

    ## deepsdf normalization
    mesh_vertices = obj_mesh.vertices
    mesh_faces = obj_mesh.faces
    center = (mesh_vertices.max(axis=0) + mesh_vertices.min(axis=0))/2.0
    max_dist = np.linalg.norm(mesh_vertices - center, axis=1).max()
    max_dist = max_dist #* 1.03
    mesh_vertices = (mesh_vertices - center) / max_dist
    obj_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
    
    return mesh_vertices, mesh_faces, obj_mesh

#####################################################
###################### OTHER ########################
#####################################################

def positional_encoding(val, L=10):
    '''
    val - the value to apply the encoding to
    L   - controls the size of the encoding (size = 2*L  - see paper for details)
    Implements the positional encoding described in section 5.1 of NeRF
    https://arxiv.org/pdf/2003.08934.pdf
    '''
    return [x for i in range(L) for x in [math.sin(2**(i)*math.pi*val), math.cos(2**(i)*math.pi*val)]]


def positional_encoding_tensor(coords, L=10):
    assert(len(coords.shape)==2)
    columns = []
    for i in range(coords.shape[1]):
        columns += [coords[:,i]]+[x for j in range(L) for x in [torch.sin(2**i*coords[:,i]), torch.cos(2**(i)*coords[:,i])]]
    pos_encodings = torch.stack(columns, dim=-1)
    return pos_encodings

def sendToDevice(TupleOrTensor, Device):
    '''
    Send tensor or tuple to specified device
    '''
    if isinstance(TupleOrTensor, torch.Tensor):
        TupleOrTensorD = TupleOrTensor.to(Device)
    else:
        TupleOrTensorD = [None]*len(TupleOrTensor)
        for Ctr in range(len(TupleOrTensor)):
            TupleOrTensorD[Ctr] = sendToDevice(TupleOrTensor[Ctr], Device)
        if isinstance(TupleOrTensor, tuple):
            TupleOrTensorD = tuple(TupleOrTensorD)

    return TupleOrTensorD

def plotLosses(loss_history, save_dir, name):
    plt.clf()
    for loss in loss_history:
        if loss == "train" or loss == "val":
            plt.plot(loss_history[loss], linestyle='-', label=loss)
        else:
            plt.plot(loss_history[loss], linestyle="--", label=loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(name)
    plt.savefig(os.path.join(save_dir, name, "losses_curve.png"))


# Ref: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
def sample_hemisphere(points):
    # sample from a sphere
    normal_samples = np.random.normal(size=(len(points),3))
    # sphere to positive x hemisphere
    normal_samples[:, 0] = np.abs(normal_samples[:, 0])
    surface_samples = normal_samples / np.stack((np.linalg.norm(normal_samples,axis=1),)*3, axis=1)
    
    vec1 = np.array([[1., 0., 0.]])
    vec2 = np.copy(points)

    vec2 /= np.linalg.norm(vec2, axis=-1)[:,np.newaxis]

    v = np.cross(vec1, vec2)
    c = np.inner(vec1, vec2).reshape(-1)[:, np.newaxis, np.newaxis]
    s = np.linalg.norm(v, axis=-1)[:, np.newaxis, np.newaxis]

    kmat = np.zeros((len(v), 3, 3))
    kmat[:,0,1] = -v[:,2]
    kmat[:,0,2] = v[:,1]
    kmat[:,1,0] = v[:,2]
    kmat[:,1,2] = -v[:,0]
    kmat[:,2,0] = -v[:,1]
    kmat[:,2,1] = v[:,0]
    
    rotation_matrix = np.eye(3)[np.newaxis, :, :] + kmat + kmat@kmat * ((1 - c) / (s ** 2))
    surface_samples = rotation_matrix@surface_samples[:,:,np.newaxis]
    return surface_samples.squeeze(-1)


def random_on_sphere(radius):
    '''
    Returns a random point on the surface of the sphere centered at the origin with given radius
    '''
    r_x = norm.rvs()
    r_y = norm.rvs()
    r_z = norm.rvs()
    normalizer = radius / ((r_x**2 + r_y**2 + r_z**2)**0.5)
    return np.array([r_x*normalizer, r_y*normalizer, r_z*normalizer])


def random_within_sphere(radius):
    '''
    Returns a point sampled randomly from the volume of a sphere centered at the origin with given radius
    '''
     # choose a point on the surface of the sphere
    initial_point = random_on_sphere(radius)
    # choose a radius towards the surface point
    magnitude = uniform.rvs() ** (1./3.)
    return initial_point * magnitude


def sample_uniform_ray_space(radius, **kwargs):
    '''
    Returns a ray that has been sampled uniformly from the 5D ray space of x,y,z,theta,phi
    Sampling Procedure:
        1) Choose a start point uniformly at random within the bounding sphere
        2) Choose a direction on the unit sphere
        3) The end point is the start point plus the direction
    '''
    start_point = random_within_sphere(radius)
    end_point = start_point + random_on_sphere(0.5)
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
    start_point = random_within_sphere(radius)
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
    start_point = random_within_sphere(radius)
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
    direction = random_on_sphere(1.0)
    bound1, bound2 = get_sphere_intersections(end_point, direction, radius)
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
    direction = np.cross(v_normal, random_on_sphere(1.0))
    bound1, bound2 = get_sphere_intersections(end_point, direction, radius)
    position = uniform.rvs()
    start_point = bound1*position + (1.-position)*bound2
    return start_point, end_point, None


def get_sphere_intersections(p0, v, radius):
    '''
    Returns the two points where the line p = p0 + t*v intersects the origin centered sphere with given radius
    Returns None if there is no intersection or if the intersections lie in the negative v direction
    The intersection closest to p0 is returned first
    This involves solving the following for t:
    radius = ||p0 + t*v||
    '''
    # use quadratic equation to solve
    # 0 = t^2 (v dot v) + t (2 p0 dot v) + p0 dot p0 - r^2
    a = np.dot(v, v)
    b = 2 * np.dot(p0, v)
    c = np.dot(p0, p0) - radius**2
    inner_term = b**2 - 4*a*c
    if inner_term < 0.:
        print(np.linalg.norm(p0))
        print(f"{p0},   {v},   {radius}")
        return None
    partial = np.sqrt(inner_term)
    x1 = (-b - partial) / (2*a)
    x2 = (-b + partial) / (2*a)
    if x1 < 0. and x2 < 0.:
        print(f"{x1},{x2}")
        print(np.linalg.norm(p0))
        print(f"{p0},   {v},   {radius}")
        return None
    return p0 + x1*v, p0 + x2*v


def get_vertex_normals(verts, faces):
    '''
    Given an array of n vertices and an array of face indices, returns an nx3 array containing the vertex normals. 
    The normals are calculated as the average of the face normal for each face containing the vertex.
    '''
    a = verts[faces.astype(int)][:,0]
    b = verts[faces.astype(int)][:,1]
    c = verts[faces.astype(int)][:,2]

    e1 = b-a
    e2 = c-a

    face_normals = np.cross(e1, e2)
    face_normals_magnitude = np.linalg.norm(face_normals, axis=1)
    for i in range(faces.shape[0]):
        if face_normals_magnitude[i] == 0.:
            print("Face normal is zero", faces[i])
    # print(f"FACE NORMS IS ZERO MAG: {face_normals.shape[0] - np.nonzero(np.linalg.norm(face_normals, axis=1).shape[0])}")
    # print(face_normals_magnitude[0:5])
    face_normals = (face_normals / np.hstack([face_normals_magnitude[:,np.newaxis]]*3))
    # print(np.linalg.norm(face_normals, axis=1)[0:5])
    vert_normals = np.zeros((verts.shape[0], 3))
    vert_face_count = np.zeros((verts.shape[0]))
    for i in range(faces.shape[0]):
        for j in range(faces.shape[1]):
            vert_face_count[faces[i][j]] += 1
            vert_normals[faces[i][j]] += face_normals[i]
    vert_normals = vert_normals / np.hstack([np.linalg.norm(vert_normals, axis=1)[:,np.newaxis]]*3)
    return vert_normals


def max_edge(verts, faces, padding=1.001):
    '''
    Finds the maximum edge length within the mesh
    '''
    a = verts[faces][:,0]
    b = verts[faces][:,1]
    c = verts[faces][:,2]
    e1 = b-a
    e2 = c-b
    e3 = a-c
    all_edges = np.vstack([e1,e2,e3])
    max_edge_length = np.max(np.sqrt(np.sum(np.square(all_edges), axis=1)))
    return max_edge_length * padding


def ray_occ_depth(faces, verts, ray_start_depth=1., near_face_threshold=0.08, v=None):
    '''
    Takes in faces and verts which define a mesh that has been rotated so that the ray end point is at the origin, and
    the ray start point lies on the z axis at a distance of ray_start_depth
    
    This function returns
        occ, a boolean indicating whether or not the start of the ray lies within the mesh
        depth, the depth to the first intersection, or np.inf if there are no intersections in the positive direction
    
    v can be passed if the the ray endpoint is a mesh vertex. This prevents each face with vertex v from being counted as a separate intersection
    '''
     # Prune faces far from the ray
    near_verts, near_faces, near_vert_indices,_ = prune_mesh(verts, faces, near_face_threshold)

    # Remove the faces that contain the ray endpoint vertex
    if v is not None:
        near_faces = near_faces[np.all(near_faces != near_vert_indices[v], axis=1)]

    wgts = get_weights(near_faces, near_verts)
    wgt_verts = near_verts[near_faces].reshape((-1,3))[:,:2]
    # a face is only intersected if all three of its halfspaces have positive outputs
    halfspace_outputs = (np.sum(np.multiply(wgts, -wgt_verts[:,:2]), axis=1) >= 0.).reshape((-1,3))

    intersected_faces = near_faces[np.all(halfspace_outputs, axis=1)]
    intersections = get_intersection_depths(near_verts[intersected_faces])
    # check if the ray origin is inside the mesh
    intersections_behind_origin = (intersections - ray_start_depth) >= 0
    n_behind_origin = np.sum(intersections_behind_origin)
    if v is not None:
        n_behind_origin += near_verts[near_vert_indices[v]][2] >= ray_start_depth
    occ = (n_behind_origin) % 2 != 0
    # case where there are no intersections
    if intersections.shape[0] == 0 and v is None:
        return occ, np.inf
    # case where sole intersection is ray endpoint
    if intersections.shape[0] == 0 and v is not None:
        return occ, ray_start_depth

    # get ray depths
    intersections[(intersections - ray_start_depth) >= 0] = np.NINF
    ray_depth = np.max(intersections - ray_start_depth) * -1
    if v is not None:
        # account for the intersection at 0 that we removed
        if ray_start_depth > 0 and ray_start_depth < ray_depth:
            return occ, ray_start_depth 


def sampling_preset_noise(sampling_method, noise):
    '''
    Defines a new version of one of the sampling functions with a different noise value set
    '''
    def preset_noise(radius, verts=None, vert_normals=None, v=None, **kwargs):
        return sampling_method(radius, verts=verts, noise=noise, vert_normals=vert_normals, v=None, **kwargs)
    return preset_noise


def load_object_shapenet(object_path):
    obj_file = os.path.join(object_path, "models", "model_normalized.obj")

    obj_mesh = trimesh.load_mesh(obj_file)
    obj_mesh = as_mesh(obj_mesh)
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


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def rotate_mesh(verts, ray_start, ray_end):
    '''
    Rotates mesh vertices so that the ray endpoint is at the origin and the ray start lies on the z-axis
    '''
    vec = ray_start - ray_end
    rot_mat = z_align_vector(vec)
    translation = -1 * ray_end
    rot_verts = np.matmul(verts + translation, rot_mat)
    return rot_verts


def prune_mesh(verts, faces, radius):
    '''
    Prunes vertices and faces that are more than radius away from the z axis
    Returns a new set of vertices and faces, and an array to convert old vert indices to pruned vert indices, and an array to convert to original face indices
    '''
    # take only faces near the ray
    near_verts_i = np.sqrt(np.sum(np.square(verts[:,:2]), axis=1)) < radius
    near_faces_i = np.all(near_verts_i[faces], axis=1)
    new_vert_indices = np.cumsum(near_verts_i) - 1

    near_faces = new_vert_indices[faces[near_faces_i]]
    near_verts = verts[near_verts_i]
    original_faces = np.arange(faces.shape[0])[near_faces_i]

    return near_verts, near_faces, new_vert_indices, original_faces


def z_align_vector(vec):
    '''
    Finds a rotation matrix that aligns vec with the z axis
    '''

    v_mag = np.sqrt(np.sum(np.square(vec)))
    z_vec = np.array([0.,0.,1.]) * v_mag
    axis_of_rotation = np.cross(z_vec, vec)
    if np.linalg.norm(axis_of_rotation) != 0.0:
        axis_of_rotation /= np.linalg.norm(axis_of_rotation)
    else:
        # In the event that vec is on the z axis, we will choose the y axis as our rotation axis
        axis_of_rotation = np.array([0.0,1.0,0.0])
    A = get_xprod_mat(axis_of_rotation)

    # theta is angle between vec and z axis
    cos_theta = np.sum(vec * z_vec) / (v_mag * v_mag)
    sin_theta = np.sqrt(1. - np.square(cos_theta))

    # calculate rotation matrix
    # https://people.eecs.berkeley.edu/~ug/slide/pipeline/assignments/as5/rotation.html
    rot_mat = np.eye(3) + A * sin_theta + np.matmul(A, A) * (1-cos_theta)
    return rot_mat


def get_xprod_mat(vec):
    '''
             | 0   -az ay  |   | vx |
    a x v =  | az  0   -ax | x | vy | = Av
             | -ay ax   0  |   | vz |

    Returns matrix A, given vector a. This matrix can be used
    to compute a cross product with another vector, v
    '''

    return np.array([[0., -vec[2], vec[1]],
                    [vec[2], 0., -vec[0]],
                    [-vec[1], vec[0], 0.]])


def get_weights(faces, verts):
    '''
    Returns weights that define halfspaces along triangle edges in the x-y plane
    '''
    a = verts[faces][:,0]
    b = verts[faces][:,1]
    c = verts[faces][:,2]
    e1 = (b-a)[:,:2]
    e2 = (c-b)[:,:2]
    e3 = (a-c)[:,:2]
    wgts_a = np.hstack([e1[:,1,np.newaxis], -1 * e1[:,0, np.newaxis]])
    wgts_b = np.hstack([e2[:,1,np.newaxis], -1 * e2[:,0, np.newaxis]])
    wgts_c = np.hstack([e3[:,1,np.newaxis], -1 * e3[:,0, np.newaxis]])

    wgt_a_correction = np.ones(wgts_a.shape)
    wgt_b_correction = np.ones(wgts_b.shape)
    wgt_c_correction = np.ones(wgts_c.shape)
    # correct_wgts = np.ones(wgts_a.shape)

    wgt_a_correction[np.sum(wgts_a*e3, axis=1) > 0] = -1.
    wgt_b_correction[np.sum(wgts_b*e1, axis=1) > 0] = -1.
    wgt_c_correction[np.sum(wgts_c*e2, axis=1) > 0] = -1.

    wgts_a = wgts_a * wgt_a_correction
    wgts_b = wgts_b * wgt_b_correction
    wgts_c = wgts_c * wgt_c_correction

    wgts = np.hstack([wgts_a[:,np.newaxis], wgts_b[:,np.newaxis],wgts_c[:,np.newaxis]]).reshape((-1,2))
    return wgts


def get_intersection_depths(occupied_faces):
    '''
    Computes the depths, d, of the points (0,0,d) where a ray along the z-axis intersects each of the provided faces
    '''
    vec1 = occupied_faces[:,1,:] - occupied_faces[:,0,:]
    vec2 = occupied_faces[:,2,:] - occupied_faces[:,0,:]
    w = np.cross(vec1 ,vec2)
    a = occupied_faces[:,0,:]
    d = np.sum(np.multiply(w,a), axis=1) / w[:,2]
    return d


def mesh_normalize(verts):
    '''
    Translates and rescales mesh vertices so that they are tightly bounded within the unit sphere
    '''
    translation = (np.max(verts, axis=0) + np.min(verts, axis=0))/2.
    verts = verts - translation
    scale = np.max(np.linalg.norm(verts, axis=1))
    verts = verts / scale
    return verts
