'''
Utility functions that can be used to generate 3D occupancy data, as well as ray surface depth
and skinning weights data. 
'''

import numpy as np
import utils
from tqdm import tqdm
import trimesh

# np.random.seed(20210804)

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

def max_edge(verts, faces, padding=1.001):
    '''
    Finds the minimum edge length within the mesh
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

def get_barycentric_coordinates(intersection, vertices):
    '''
    Given a point on a triangle, and an array with 3 triangle vertices (in 3D), this returns the barycentric coordinates (weights for each 
    triangle vertex) for the point of intersection
    '''
    pass

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

    return occ, ray_depth

def ray_occ_depth_visual(faces, verts, ray_start_depth=1., v=None, near_face_threshold=0.08):
    '''
    Same functionality as ray_occ_depth, except that the face indices corresponding to the first ray 
    intersection(s) are also returned. This function is about 30% slower than the original so it should 
    only be used for visualization, not data generation.
    '''
     # Prune faces far from the ray
    near_verts, near_faces, near_vert_indices, original_faces = prune_mesh(verts, faces, near_face_threshold)

    # Remove the faces that contain the ray endpoint vertex
    if v is not None:
        removed_faces = original_faces[np.any(near_faces == near_vert_indices[v], axis=1)]
        # change removed faces back to original vertex indices for visualization
        # removed_faces = np.array([[get_original_index(removed_faces[i][j]) for j in range(removed_faces.shape[1])] for i in range(removed_faces.shape[0])])
        original_faces = original_faces[np.all(near_faces != near_vert_indices[v], axis=1)]
        near_faces = near_faces[np.all(near_faces != near_vert_indices[v], axis=1)]

    wgts = get_weights(near_faces, near_verts)
    wgt_verts = near_verts[near_faces].reshape((-1,3))[:,:2]

    halfspace_outputs = (np.sum(np.multiply(wgts, -wgt_verts[:,:2]), axis=1) >= 0.).reshape((-1,3))

    intersected_face_inds = np.all(halfspace_outputs, axis=1)
    intersected_faces = near_faces[intersected_face_inds]
    original_faces = original_faces[intersected_face_inds]
    intersections = get_intersection_depths(near_verts[intersected_faces])
    # change back to original vertex indices for visualization
    # intersected_faces = np.array([[get_original_index(intersected_faces[i][j]) for j in range(intersected_faces.shape[1])] for i in range(intersected_faces.shape[0])])

    # check if the ray origin is inside the mesh
    intersections_behind_origin = (intersections - ray_start_depth) >= 0
    n_behind_origin = np.sum(intersections_behind_origin)
    if v is not None:
        n_behind_origin += near_verts[near_vert_indices[v]][2] >= ray_start_depth
    occ = (n_behind_origin) % 2 != 0

    # case where there are no intersections
    if intersections.shape[0] == 0 and v is None:
        return occ, np.inf, np.array([])
    # case where sole intersection is ray endpoint
    if intersections.shape[0] == 0 and v is not None:
        return occ, ray_start_depth, removed_faces

    # get ray depths
    intersections[(intersections - ray_start_depth) >= 0] = np.NINF
    ray_depth = np.max(intersections - ray_start_depth) * -1
    # intersected_faces = intersected_faces[np.argmax(intersections)][np.newaxis, :]
    original_faces = original_faces[np.argmax(intersections)][np.newaxis] if ray_depth < np.inf else np.array([])
    if v is not None:
        # account for the intersection at 0 that we removed
        if ray_start_depth > 0 and ray_start_depth < ray_depth:
            return occ, ray_start_depth, removed_faces 
    return occ, ray_depth, original_faces

def ray_all_depths(faces, verts, near_face_threshold=0.08, ray_start_depth=1., return_faces=False):
    '''
    Takes in faces and verts which define a mesh that has been rotated so that the ray end point is at the origin, and
    the ray start point lies on the z axis at a distance of ray_start_depth
    
    This function returns
        intersections, a list of depths to the nth intersection
        original_faces, the face indices that are intersected (only returned if return_faces is True)
    '''
     # Prune faces far from the ray
    near_verts, near_faces, _, original_faces = prune_mesh(verts, faces, near_face_threshold)

    # weights for face halfspaces
    wgts = get_weights(near_faces, near_verts)
    wgt_verts = near_verts[near_faces].reshape((-1,3))[:,:2]

    # a face is only intersected if all three of its halfspaces have positive outputs
    halfspace_outputs = (np.sum(np.multiply(wgts, -wgt_verts[:,:2]), axis=1) >= 0.).reshape((-1,3))
    
    intersected_faces_i = np.all(halfspace_outputs, axis=1)
    intersected_faces = near_faces[intersected_faces_i]
    intersections = get_intersection_depths(near_verts[intersected_faces])
    intersections = ray_start_depth - intersections
    # remove negative intersections and sort the intersections by depth
    intersections = [i for i in intersections if i > 0.]
    intersections.sort()

    if return_faces:
        return intersections, original_faces[intersected_faces_i]
    else:
        return intersections
