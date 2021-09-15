'''
General utility functions
'''
import math
import numpy as np
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt

# def deepsdf_undo_preprocess(smpl_vertices, points):
#         '''
#         Undo the DeepSDF preprocessing that has been applied to points
#         '''
#         translation = (np.max(smpl_vertices, axis=0)[0] + np.min(smpl_vertices, axis=0)[0])/2.
#         centered_mesh = smpl_vertices - translation
#         scale = np.max(np.sqrt(np.sum(np.square(centered_mesh), axis=1)))
#         # DeepSDF uses this 1.03 constant
#         points = scale *1.03 * points
#         points += translation
#         return points

def mesh_normalize(verts):
    '''
    Translates and rescales mesh vertices so that they are tightly bounded within the unit sphere
    '''
    translation = (np.max(verts, axis=0) + np.min(verts, axis=0))/2.
    verts = verts - translation
    scale = np.max(np.linalg.norm(verts, axis=1))
    verts = verts / scale
    return verts


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


def get_vertex_normals(verts, faces):
    '''
    Given an array of n vertices and an array of face indices, returns an nx3 array containing the vertex normals. 
    The normals are calculated as the average of the face normal for each face containing the vertex.
    '''
    a = verts[faces][:,0]
    b = verts[faces][:,1]
    c = verts[faces][:,2]

    e1 = b-a
    e2 = c-a

    face_normals = np.cross(e1, e2)
    face_normals_magnitude = np.linalg.norm(face_normals, axis=1)
    # print(face_normals_magnitude[0:5])
    face_normals = (face_normals / np.hstack([face_normals_magnitude[:,np.newaxis]]*3)) * 0.1
    # print(np.linalg.norm(face_normals, axis=1)[0:5])
    vert_normals = np.zeros((verts.shape[0], 3))
    vert_face_count = np.zeros((verts.shape[0]))
    for i in range(faces.shape[0]):
        for j in range(faces.shape[1]):
            vert_face_count[faces[i][j]] += 1
            vert_normals[faces[i][j]] += face_normals[i]
    vert_normals = vert_normals / np.hstack([vert_face_count[:,np.newaxis]]*3)
    return vert_normals

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
        return None
    partial = np.sqrt(inner_term)
    x1 = (-b - partial) / (2*a)
    x2 = (-b + partial) / (2*a)
    if x1 < 0. and x2 < 0.:
        return None
    return p0 + x1*v, p0 + x2*v

def vector_to_angles(vector):
    '''
    Given a vector, returns the angles theta and phi from it's spherical coordinates (ISO convention)
    '''
    phi = np.arctan(vector[1]/vector[0])
    xy = np.sqrt(np.square(vector[0]) + np.square(vector[1]))
    theta = np.arctan(xy/vector[2])
    return theta, phi

def positional_encoding(val, L=10):
    '''
    val - the value to apply the encoding to
    L   - controls the size of the encoding (size = 2*L  - see paper for details)
    Implements the positional encoding described in section 5.1 of NeRF
    https://arxiv.org/pdf/2003.08934.pdf
    '''
    return [x for i in range(L) for x in [math.sin(2**(i)*math.pi*val), math.cos(2**(i)*math.pi*val)]]


def saveLossesCurve(*args, **kwargs):
    '''
    Copied from Beacon - wanted log scale for loss
    '''
    plt.clf()
    ylim = 0
    for Ctr, arg in enumerate(args, 0):
        if len(arg) <= 0:
            continue
        if Ctr >= 1: # For sublosses
            plt.plot(arg, linestyle='--')
        else:
            plt.plot(arg, linestyle='-')
        ylim = ylim + np.median(np.asarray(arg))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])
    if 'legend' in kwargs:
        if len(kwargs['legend']) > 0:
            plt.legend(kwargs['legend'])
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    if 'log' in kwargs and kwargs['log']:
        plt.yscale("log")
    if ylim > 0:
        plt.ylim([0.0, ylim])
    if 'out_path' in kwargs:
        plt.savefig(kwargs['out_path'])
    else:
        print('[ WARN ]: No output path (out_path) specified. beacon.utils.saveLossesCurve()')
