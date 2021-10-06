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

def color_difference(numerical_difference, gt_mask, learned_mask, scale_cap=0.15):
    '''
    Shows the differences between the learned and ground truth predictions.
    numerical_difference should be (learned - gt)
    Color scale extends between -cap and cap (by default depth disparities larger than 0.15 will appear the same)
    '''
    depth_difference = np.zeros(gt_mask.shape + (3,))
    # set colors for positive difference
    depth_difference[:,:,1] = np.where(numerical_difference > 0., np.min(np.stack([numerical_difference/scale_cap, np.ones(numerical_difference.shape)], axis=-1), axis=-1), 0.)
    depth_difference[:,:,0] = np.where(numerical_difference > 0., np.min(np.stack([numerical_difference/scale_cap, np.ones(numerical_difference.shape)], axis=-1)*0.4, axis=-1), 0.)
    depth_difference[:,:,2] = np.where(numerical_difference > 0., np.min(np.stack([numerical_difference/scale_cap, np.ones(numerical_difference.shape)], axis=-1)*0.4, axis=-1), 0.)
    # set colors for negative difference
    depth_difference[:,:,2] = np.where(numerical_difference < 0., np.min(np.stack([numerical_difference/(-scale_cap), np.ones(numerical_difference.shape)], axis=-1), axis=-1), depth_difference[:,:,2])
    depth_difference[:,:,0] = np.where(numerical_difference < 0., np.min(np.stack([numerical_difference/(-scale_cap), np.ones(numerical_difference.shape)], axis=-1)*0.4, axis=-1), depth_difference[:,:,0])
    depth_difference[:,:,1] = np.where(numerical_difference < 0., np.min(np.stack([numerical_difference/(-scale_cap), np.ones(numerical_difference.shape)], axis=-1)*0.4, axis=-1), depth_difference[:,:,1])
    depth_difference[np.logical_not(np.logical_or(gt_mask, learned_mask))] = np.array([1.,1.,1.])
    # XOR operation. Colors image red where the masks don't align
    depth_difference[np.logical_and(np.logical_or(gt_mask, learned_mask), np.logical_not(np.logical_and(gt_mask, learned_mask)))] = np.array([1.,0.4,0.4])
    return depth_difference


def show_depth_data(gt_intersect, gt_depth, learned_intersect, learned_depth, all_axes, vmin, vmax):
    '''
    gt_intersect      - a ground truth intersection mask
    gt_depth          - a ground truth depth map
    learned_intersect - learned intersection mask
    learned_depth     - learned depth map 
    all_axes          - the matplotlib axes to write the data to
    vmin, vmax        - the minimum and maximum value for depth normalization
    '''
    scale_cap = 0.15
    ax1,ax2,ax3,ax4,ax5,ax6=all_axes
    depth_learned_mask = np.where(learned_intersect, learned_depth, np.inf)
    numerical_difference = depth_learned_mask - gt_depth
    depth_difference = color_difference(numerical_difference, gt_intersect > 0.5, learned_intersect > 0.5, scale_cap=scale_cap)
    for ax in all_axes:
        ax.clear()
    ax1.imshow(gt_intersect)
    ax1.set_title("GT Intersect")
    ax2.imshow(gt_depth, vmin=vmin, vmax=vmax)
    ax2.set_title("GT Depth")
    ax3.imshow(depth_difference)
    ax3.set_title("Depth Difference")
    ax4.imshow(learned_intersect)
    ax4.set_title("Intersect")
    ax5.imshow(depth_learned_mask, vmin=vmin, vmax=vmax)
    ax5.set_title("Depth (Masked)")
    ax6.imshow(learned_depth, vmin=vmin, vmax=vmax)
    ax6.set_title("Depth")

def show_depth_data_4D(gt_n_ints, gt_depth, learned_n_ints, learned_depth, all_axes, vmin, vmax, max_n_ints):
    '''
    gt_n_ints           - a ground truth intersection mask
    gt_first_depth      - a ground truth depth map 
    learned_n_ints      - learned intersection mask
    learned_first_depth - learned depth map
    all_axes            - the matplotlib axes to write the data to
    vmin, vmax          - the minimum and maximum value for depth normalization
    '''
    scale_cap = 0.15
    ax1, ax2, ax3, ax4, ax5, ax6, ax7 = all_axes
    depth_learned_mask = np.where(learned_n_ints > 0.5, learned_depth, np.inf)
    numerical_depth_difference = depth_learned_mask - gt_depth
    depth_difference = color_difference(numerical_depth_difference, gt_n_ints > 0.5, learned_n_ints > 0.5, scale_cap=scale_cap)
    numerical_int_difference = learned_n_ints - gt_n_ints
    int_difference = color_difference(numerical_int_difference, gt_n_ints > 0.5, learned_n_ints > 0.5, scale_cap=max_n_ints)
    for ax in all_axes:
        ax.clear()
    ax1.imshow(gt_n_ints, vmin=0, vmax=max_n_ints)
    ax1.set_title("GT Intersection Count")
    ax2.imshow(int_difference)
    ax2.set_title("Intersection Difference")
    ax3.imshow(gt_depth, vmin=vmin, vmax=vmax)
    ax3.set_title("GT Depth")
    ax4.imshow(learned_n_ints, vmin=0, vmax=max_n_ints)
    ax4.set_title("Intersection Count")
    ax5.imshow(depth_difference)
    ax5.set_title("Depth Difference")
    ax6.imshow(depth_learned_mask, vmin=vmin, vmax=vmax)
    ax6.set_title("Depth (Masked)")
    ax7.imshow(learned_depth, vmin=vmin, vmax=vmax)
    ax7.set_title("Depth")
    

