from typing import final
from matplotlib.pyplot import connect
import numpy as np
import trimesh
import argparse
import torch

import rasterization
import odf_utils

#Icosahedron taken from https://people.sc.fsu.edu/~jburkardt/data/obj/icosahedron.obj
#Icosahedron sphere connectivity https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.90.6202&rep=rep1&type=pdf

icosahedron_verts = [
[0., -0.525731, 0.850651],
[0.850651, 0., 0.525731],
[0.850651, 0., -0.525731],
[-0.850651, 0., -0.525731],
[-0.850651, 0., 0.525731],
[-0.525731, 0.850651, 0.],
[0.525731, 0.850651, 0.],
[0.525731, -0.850651, 0.],
[-0.525731, -0.850651, 0.],
[0., -0.525731, -0.850651],
[0., 0.525731, -0.850651],
[0., 0.525731, 0.850651]
]

icosahedron_faces = [
[2, 3, 7],
[2, 8, 3],
[4, 5, 6],
[5, 4, 9],
[7, 6, 12],
[6, 7, 11],
[10, 11, 3],
[11, 10, 4],
[8, 9, 10],
[9, 8, 1],
[12, 1, 2],
[1, 12, 5],
[7, 3, 11],
[2, 7, 12],
[4, 6, 11],
[6, 5, 12],
[3, 8, 10],
[8, 2, 1],
[4, 10, 9],
[5, 9, 1]
]

class MeshODF():

    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.radius = radius

    def query_rays(self, points, directions):
        '''
        Queries the surface depth from the provided points in the provided directions
        This only uses the first intersection
        '''
        points = np.array(points)
        directions = np.array(directions)
        intersect = []
        depths = []
        n_ints = [1] * points.shape[0]
        near_face_threshold = rasterization.max_edge(self.vertices, self.faces)
        for i in range(points.shape[0]):
            # if np.linalg.norm(points[i]) > self.radius:
            if odf_utils.get_sphere_intersections(points[i], directions[i], self.radius) is None:
                lines = np.concatenate([self.faces[:,:2], self.faces[:,1:], self.faces[:,[0,2]]], axis=0)
                visualizer = visualization.RayVisualizer(self.vertices, lines)
                visualizer.add_point(points[i], [1.0,0.0,0.0])
                visualizer.display()
            start_point, end_point = odf_utils.get_sphere_intersections(points[i], directions[i], self.radius)
            ray_length = np.linalg.norm(end_point-start_point)
            rot_verts = rasterization.rotate_mesh(self.vertices, start_point, end_point)
            _, depth = rasterization.ray_occ_depth(self.faces, rot_verts, ray_start_depth=ray_length, near_face_threshold=near_face_threshold)
            depth -= np.linalg.norm(points[i] - start_point)
            depth = depth if depth > 0. else np.inf
            intersect.append(depth < np.inf)
            depths.append([depth])
        # return torch tensors just so the output is exactly the same as the learned NN
        return torch.tensor(np.array(intersect)), torch.tensor(np.array(depths)), torch.tensor(np.array(n_ints))


def sphere_subdivision(verts, faces, radius=1.0):
    '''
    Verts - a list of numpy arrays defining the current vertices
    Faces - a list of lists defining the connections between the vertices
    '''
    output_verts = [v for v in verts]
    output_faces = []
    # maps an edge (two vertex indices) to the new intermediate vertex index
    new_vertex_indices = {}

    for f in faces:

        # Calculate the three new vertices, use existing vertices if they have already been added
        if (f[0], f[1]) in new_vertex_indices:
            v0v1_index = new_vertex_indices[(f[0], f[1])]
        else:
            v0v1 = (verts[f[0]] + verts[f[1]])/2.
            v0v1 = v0v1 / np.linalg.norm(v0v1) * radius
            v0v1_index = len(output_verts)
            output_verts.append(v0v1)
            # add both edge permutations to dict
            new_vertex_indices[(f[0], f[1])] = v0v1_index
            new_vertex_indices[(f[1], f[0])] = v0v1_index
        
        if (f[1], f[2]) in new_vertex_indices:
            v1v2_index = new_vertex_indices[(f[1], f[2])]
        else:
            v1v2 = (verts[f[1]] + verts[f[2]])/2.
            v1v2 = v1v2 / np.linalg.norm(v1v2) * radius
            v1v2_index = len(output_verts)
            output_verts.append(v1v2)
            # add both edge permutations to dict
            new_vertex_indices[(f[1], f[2])] = v1v2_index
            new_vertex_indices[(f[2], f[1])] = v1v2_index

        if (f[2], f[0]) in new_vertex_indices:
            v2v0_index = new_vertex_indices[(f[2], f[0])]
        else:
            v2v0 = (verts[f[2]] + verts[f[0]])/2.
            v2v0 = v2v0 / np.linalg.norm(v2v0) * radius
            v2v0_index = len(output_verts)
            output_verts.append(v2v0)
            # add both edge permutations to dict
            new_vertex_indices[(f[2], f[0])] = v2v0_index
            new_vertex_indices[(f[0], f[2])] = v2v0_index

        #Add the four new faces to the output - index order matters so we don't flip the normal
        output_faces.append([f[0], v0v1_index, v2v0_index])
        output_faces.append([v0v1_index, f[1], v1v2_index])
        output_faces.append([v2v0_index, v1v2_index, f[2]])
        output_faces.append([v0v1_index, v1v2_index, v2v0_index])

    return output_verts, output_faces

def large_edge_subdivision(verts, faces, edge_threshold=0.05):
    '''
    Verts - a list of numpy arrays defining the current vertices
    Faces - a list of lists defining the connections between the vertices
    '''
    output_verts = [v for v in verts]
    output_faces = []
    output_probes = []
    # TODO: use multiple probes from different angles
    # maps an edge (two vertex indices) to the new intermediate vertex index
    new_vertex_indices = {}

    # divide all edges over a certain threshold
    np_faces = np.array(faces)
    edges = np.concatenate([np_faces[:,:2], np_faces[:,1:], np_faces[:,[0,2]]], axis=0)
    for e in range(edges.shape[0]):
        if (edges[e][0], edges[e][1]) in new_vertex_indices:
            continue
        elif np.linalg.norm(verts[edges[e,1]]-verts[edges[e,0]]) > edge_threshold:
                new_vert = (verts[edges[e,0]]+verts[edges[e,1]])/2.
                new_vert_index = len(output_verts)
                output_verts.append(new_vert)
                output_probes.append(new_vert)
                # add both edge permutations to dict
                new_vertex_indices[(edges[e,0], edges[e,1])] = new_vert_index
                new_vertex_indices[(edges[e,1], edges[e,0])] = new_vert_index
    
    # Create new faces
    for f in faces:
        # get the intermediate vertices for all the edges that were subdivided
        intermediate_vertices = []
        case_number = 0
        if (f[0], f[1]) in new_vertex_indices:
            intermediate_vertices.append(new_vertex_indices[(f[0], f[1])])
            case_number += 1
        if (f[1], f[2]) in new_vertex_indices:
            intermediate_vertices.append(new_vertex_indices[(f[1], f[2])])
            case_number += 2
        if (f[2], f[0]) in new_vertex_indices:
            intermediate_vertices.append(new_vertex_indices[(f[2], f[0])])
            case_number += 4
        
        # Handle the 8 different cases for triangle face subdivision

        # No edges were divided
        if case_number == 0:
            output_faces.append(f)
        
        # Only the first edge was divided
        if case_number == 1:
            output_faces.append([f[0], intermediate_vertices[0], f[2]])
            output_faces.append([intermediate_vertices[0], f[1], f[2]])

        # Only the second edge was divided
        if case_number == 2:
            output_faces.append([f[0], f[1], intermediate_vertices[0]])
            output_faces.append([intermediate_vertices[0], f[2], f[0]])
            
        #Only the third edge was divided
        if case_number == 4:
            output_faces.append([f[0], f[1], intermediate_vertices[0]])
            output_faces.append([intermediate_vertices[0], f[1], f[2]])

        #The first and second edge were divided
        if case_number == 3:
            output_faces.append([f[1], intermediate_vertices[1], intermediate_vertices[0]])
            if np.linalg.norm(output_verts[intermediate_vertices[0]]-output_verts[f[2]]) < np.linalg.norm(output_verts[intermediate_vertices[1]] - output_verts[f[0]]):
                output_faces.append([f[0], intermediate_vertices[0], f[2]])
                output_faces.append([intermediate_vertices[0], intermediate_vertices[1], f[2]])
            else:
                output_faces.append([f[0], intermediate_vertices[0], intermediate_vertices[1]])
                output_faces.append([f[0], intermediate_vertices[1], f[2]])
        
        #The first and third edge were divided
        if case_number == 5:
            output_faces.append([f[0], intermediate_vertices[0], intermediate_vertices[1]])
            if np.linalg.norm(output_verts[intermediate_vertices[0]]-output_verts[f[2]]) < np.linalg.norm(output_verts[intermediate_vertices[1]] - output_verts[f[1]]):
                output_faces.append([intermediate_vertices[1], intermediate_vertices[0], f[2]])
                output_faces.append([intermediate_vertices[0], f[1], f[2]])
            else:
                output_faces.append([intermediate_vertices[0], f[1], intermediate_vertices[1]])
                output_faces.append([intermediate_vertices[1], f[1], f[2]])

        #The second and third edge were divided
        if case_number == 6:
            output_faces.append([intermediate_vertices[1], intermediate_vertices[0], f[2]])
            if np.linalg.norm(output_verts[f[0]]-output_verts[intermediate_vertices[0]]) < np.linalg.norm(output_verts[intermediate_vertices[1]] - output_verts[f[1]]):
                output_faces.append([f[0], intermediate_vertices[0], intermediate_vertices[1]])
                output_faces.append([f[0], f[1], intermediate_vertices[0]])
            else:
                output_faces.append([f[0], f[1], intermediate_vertices[1]])
                output_faces.append([intermediate_vertices[1], f[1], intermediate_vertices[0]])

        #All three edges were divided
        if case_number == 7:
            output_faces.append([f[0], intermediate_vertices[0], intermediate_vertices[2]])
            output_faces.append([intermediate_vertices[0], f[1], intermediate_vertices[1]])
            output_faces.append([intermediate_vertices[2], intermediate_vertices[1], f[2]])
            output_faces.append([intermediate_vertices[0], intermediate_vertices[1], intermediate_vertices[2]])

    return output_verts, output_faces, output_probes


def icosahedron_sphere_tessalation(radius=1., subdivisions=0):
    '''
    Returns the vertices and faces of a tessalated sphere, generated by subdividing an icosahedron
    radius - the radius of the sphere that is generated
    subdivisions - the number of times to subdivide the original icosahedron connectivity
    '''
    vertices = [np.array(v)/np.linalg.norm(v)*radius for v in icosahedron_verts]
    # the obj file wasn't zero indexed so subtract 1
    faces = [[i-1 for i in f] for f in icosahedron_faces]

    for i in range(subdivisions):
        vertices, faces = sphere_subdivision(vertices, faces, radius=radius)
    
    return vertices, faces

def vertex_dists(vertices, focal_point=[0.,0.,0.]):
    '''
    Returns the unit direction vector pointing from each vertex to the focal point
    '''
    focal_point = np.array(focal_point)
    directions = [(focal_point-v) / np.linalg.norm(focal_point-v) for v in vertices]
    return directions

def sphere_surface_to_point_cloud(obj_verts, obj_faces, sphere_vertices, focal_point=[0., 0., 0.]):
    '''
    Returns the 1st, 2nd, 3rd, and 4th+ intersection point clouds produced by shooting rays from the 
    sphere vertices in the specified directions
    '''
    focal_point = np.array(focal_point)
    near_face_threshold = rasterization.max_edge(obj_verts, obj_faces)
    pointclouds = [[],[],[],[]]

    for v in sphere_vertices:
        ray_direction = (focal_point-v) / np.linalg.norm(focal_point-v)
        rot_verts = rasterization.rotate_mesh(obj_verts, v, focal_point)
        int_depths = rasterization.ray_all_depths(obj_faces, rot_verts,near_face_threshold=near_face_threshold, ray_start_depth=np.linalg.norm(focal_point - v))
        for i, d in enumerate(int_depths):
            new_point = v + ray_direction*d
            pointclouds[min(i, 3)].append(new_point)

    return pointclouds

def sphere_surface_to_mesh(obj_verts, obj_faces, sphere_vertices, focal_point=[0.,0.,0.]):
    '''
    Returns a mesh produced by shooting rays from the sphere vertices in the specified directions
    '''

    focal_point = np.array(focal_point)
    near_face_threshold = rasterization.max_edge(obj_verts, obj_faces)
    mesh_verts = []

    for v in sphere_vertices:
        ray_direction = (focal_point-v) / np.linalg.norm(focal_point-v)
        rot_verts = rasterization.rotate_mesh(obj_verts, v, focal_point)
        _, depth = rasterization.ray_occ_depth(obj_faces, rot_verts,near_face_threshold=near_face_threshold, ray_start_depth=np.linalg.norm(focal_point - v))
        new_point = v + ray_direction * (depth if depth is not np.inf else 0.0)
        mesh_verts.append(new_point)
    
    return mesh_verts



def show_subdivisions_and_probes(vertices, probes, inf_mask, directions, faces, delta, cluster=[], c_verts=[], c_faces=[], show_mesh_faces=True):
    '''
    For visualization purposes.
    Shows which edges have been subdivided and where the probe locations are.
    '''
    mesh_file = "F:\\ivl-data\\sample_data\\stanford_bunny_watertight.obj"
    mesh = trimesh.load(mesh_file)
    gt_faces = mesh.faces
    gt_verts = mesh.vertices
    gt_verts = odf_utils.mesh_normalize(gt_verts)
    gt_lines = np.concatenate([gt_faces[:,:2], gt_faces[:,1:], gt_faces[:,[0,2]]], axis=0)


    vertices = np.array(vertices)
    probes = np.array(probes)
    inf_mask = np.array(inf_mask)
    directions = np.array(directions)
    faces = np.array(faces)
    lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)

    import visualization
    visualizer = visualization.RayVisualizer(gt_verts, gt_lines)
    if show_mesh_faces:
        visualizer.add_colored_mesh(vertices, np.array(list(faces)), np.array([[0.5,0.5,0.5]]*faces.shape[0]))
    
    probes_offset = vertices.shape[0] - probes.shape[0]
    # directions = directions[probes_offset:]
    new_lines = lines[np.any(lines >= probes_offset, axis=1)]
    new_lines = vertices[new_lines]
    
    for i in range(new_lines.shape[0]):
        visualizer.add_ray([new_lines[i,0,:], new_lines[i,1,:]], np.array([1.,0.,0.]))
    for i in range(probes.shape[0]):
        visualizer.add_ray([probes[i], probes[i]+delta*directions[i]], np.array([0.,0.,1.]))
        if inf_mask[i]:
            if i in cluster:
                visualizer.add_point(probes[i], np.array([1.,0.,1.]))
            else:
                visualizer.add_point(probes[i], np.array([1.,0.,0.]))
        else:
            visualizer.add_point(probes[i], np.array([0.,1.,0.]))

    # Visualize the expanded cluster of infinite depth vertices and their neghboring faces
    for v in c_verts:
        visualizer.add_point(vertices[v], [1.,0.,1.])
    for f in c_faces:
        visualizer.add_colored_mesh(vertices[faces[f]], np.array([[0,1,2]]), np.array([[1.,0.2,1.]]))

    visualizer.display(show_wireframe=True)

def get_depths(model, probes, directions, delta):
    '''
    Gets the ODF predicted distance for each probe in the provided direction. Each probe is offset from the estimated surface point by delta.
    If depth is infinity, checks the opposite direction. If the opposite direction doesn't confirm infinity, then depth is set to be delta.
    model      - the ODF
    probes     - the 3D probe location
    directions - vectors defining the query direction for each probe
    delta      - how far set back from the surface each probe point is
    Returns a depth for each probe
    '''

    with torch.no_grad():
        # pass in surface point, direction
        _, depths, _ = model.query_rays(torch.tensor(probes, dtype=torch.float32), torch.tensor(directions, dtype=torch.float32))
    model_depths = depths.cpu()
    model_depths = torch.min(model_depths, dim=1)[0]
    model_depths = model_depths.numpy()

    # TODO: the sphere intersection radius should be lower than the camera radius (1.25) but larger than 1+delta
    probe_radius = 1.15
    opposing_points = [odf_utils.get_sphere_intersections(probes[i], directions[i], probe_radius)[1] for i in range(len(probes)) if model_depths[i]==np.inf]
    opposing_points_stack = [x for x in opposing_points]
    opposing_points_stack.reverse()
    # this is the distance to the probe from the opposite point
    opposing_vert_depth = [np.linalg.norm(opposing_points_stack.pop()-probes[i]) for i in range(len(probes)) if model_depths[i] == np.inf]
    # opposite query direction
    opposing_directions = [-1*directions[i] for i in range(len(probes)) if model_depths[i]==np.inf]

    has_inf_depths = len(opposing_points) > 0
    # check whether the infinite depth rays are truly infinite or possibly just occluded by a another face (in which case depth estimate should just be 0.0)
    if has_inf_depths:
        with torch.no_grad():
            _, opposing_model_depths, _ = model.query_rays(torch.tensor(opposing_points, dtype=torch.float32), torch.tensor(opposing_directions, dtype=torch.float32))
        opposing_model_depths = opposing_model_depths.cpu()
        opposing_model_depths = torch.min(opposing_model_depths, dim=1)[0]
        opposing_model_depths = list(opposing_model_depths.numpy())
        confirmed_inf = [opposing_model_depths[i] > opposing_vert_depth[i] for i in range(len(opposing_model_depths))]
        # reverse so we can pop the elements off in order
        confirmed_inf.reverse()
        # update the depths of the inf rays depending on whether or not they were confirmed
        model_depths = [x if x != np.inf else (np.inf if confirmed_inf.pop() else 0.0) for x in model_depths]

    return model_depths

def pull_back_probes(probes, directions, delta):
    '''
    Pulls probes back from the object surface. Move in the opposite direction from direction, with a magnitude of delta.
    '''
    directions = directions[-len(probes):]
    probes = [probes[i]- delta*directions[i] for i in range(len(probes))]
    return probes

# #####################################################################################
#                  PROBE CLUSTERING, TUNNELING, AND HOLE REPAIR
# #####################################################################################
def edges_adjacent_to_faces(lines, faces, face_indices, used_lines):
    face_vertices = set(list(faces[face_indices].flatten()))
    all_masks = [np.any(lines==x, axis=1)[:,np.newaxis] for x in face_vertices]
    final_mask = np.any(np.hstack(all_masks), axis=1)
    adj_lines = list(lines[final_mask])
    return [x for x in adj_lines if x not in used_lines]

def faces_adjacent_to_edges(faces, edges, used_faces):
    all_masks = [np.any(np.hstack([np.any(faces==x[0], axis=1)[:, np.newaxis], np.any(faces==x[1], axis=1)[:, np.newaxis]]))[:, np.newaxis] for x in edges]
    new_face_inds = [i for i in range(faces.shape[0]) if all_masks[i] and i not in used_faces]
    return new_face_inds

def face_plane_intersection(faces, intersected_face1, intersected_face2, vert1, direction1, vert2, direction2):
    '''
    Returns all the faces intersected by the plane connecting two probe rays
    '''

    plane_verts, plane_faces = adjacent_rays_plane(vert1, direction1, vert2, direction2)
    intersected_faces = [intersected_face1, intersected_face2]
    
    faces = np.array(faces)
    lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)
    used_lines = np.concatenate([faces[intersected_faces,:2], faces[intersected_faces,1:], faces[intersected_faces,[0,2]]], axis=0)
    used_lines = list(used_lines)

    latest_faces = [x for x in intersected_faces]

    while len(latest_faces) > 0:
        # get new neighboring lines
        neighbor_lines = edges_adjacent_to_faces(lines, faces, latest_faces, used_lines)
        # select only the neighboring lines that intersect our plane
        intersecting_neighbor_lines = []
        near_face_threshold = rasterization.max_edge(plane_verts, plane_faces)
        for l in neighbor_lines:
            rot_verts = rasterization.rotate_mesh(plane_verts, l[0], l[1])
            _, depth, _ = rasterization.ray_occ_depth(plane_faces, rot_verts, ray_start_depth=np.linalg.norm(l[1]-l[0]), near_face_threshold=near_face_threshold)
            if depth < np.inf:
                intersecting_neighbor_lines.append(l)

        #get new faces that border the new lines
        latest_faces = faces_adjacent_to_edges(faces, intersecting_neighbor_lines, intersected_faces)

        # add new faces and lines to respective lists
        used_lines += intersecting_neighbor_lines
        faces += latest_faces

    return intersected_faces

def probe_edges_all_intersections(vertices, faces, directions, probe_edges):
    '''
    Returns all the faces that are intersected by the probe planes defined by each edge.
    Returns faces as a set of indices
    '''
    intersected_faces = set()
    near_face_threshold = rasterization.max_edge(vertices, faces)
    for e in probe_edges:
        intersection1 = single_vert_intersection(vertices, faces, vertices[e[0]], directions[e[0]], near_face_threshold)
        intersection2 = single_vert_intersection(vertices, faces, vertices[e[0]], directions[e[0]], near_face_threshold)
        intersected_faces = set.union(intersected_faces, face_plane_intersection(faces, intersection1, intersection2, vertices[e[0]], directions[e[0]], vertices[e[1]], directions[e[1]]))
    return intersected_faces
    

def adjacent_rays_plane(vert1, direction1, vert2, direction2):
    '''
    Given two vertices and their probe_directions, returns the adjacent triangle planes that join the two sampling rays
    '''

    ray1_sphere_intersect = odf_utils.get_sphere_intersections(vert1, direction1, 1.0)[1]
    ray2_sphere_intersect = odf_utils.get_sphere_intersections(vert2, direction2, 1.0)[1]

    if np.linalg.norm(ray1_sphere_intersect-vert2) < np.linalg.norm(ray2_sphere_intersect-vert1):
        return np.array([vert1, vert2, ray1_sphere_intersect, ray2_sphere_intersect]), np.array([[0,2,1], [1,2,3]])
    else:
        return np.array([vert1, vert2, ray1_sphere_intersect, ray2_sphere_intersect]), np.array([[0,2,3], [1,0,3]])

def single_vert_intersection(vertices, faces, vert, direction, near_face_threshold):
    '''
    Returns the face index of the first self intersection from the vert and the specified direction
    '''
    vertices = np.array(vertices)
    faces = np.array(faces)

    # epsilon value prevents intersecting the face we start at (same idea as avoiding shadow acne in ray tracing)
    epsilon = 0.000001
    ray_length = np.linalg.norm(direction) - epsilon
    rot_verts = rasterization.rotate_mesh(vertices, vert, vert + direction)
    _, _, intersected_face = rasterization.ray_occ_depth_visual(faces, rot_verts, ray_start_depth=ray_length, near_face_threshold=near_face_threshold)
    return intersected_face

def vert_cluster_intersections(vertices, faces, probe_cluster, directions):
    '''
    Returns the indices of faces that are intersected by the probes in the cluster
    '''

    near_face_threshold = rasterization.max_edge(np.array(vertices), np.array(faces))
    
    intersected_faces = []
    for i, probe in enumerate(probe_cluster):
        # print(single_vert_intersection(vertices, faces, probe, directions[i], delta, near_face_threshold))
        # print("SHOWING FAILED VERTEX")

        # _, depth = single_vert_intersection(vertices, faces, vertices[probe], directions[i], delta, near_face_threshold)
        # print(depth)

        # show_subdivisions_and_probes(np.array(vertices), np.array([vertices[probe], vertices[probe] + depth*directions[i]]), [True, True], [directions[i], directions[i]], faces, delta)
        intersected_faces.append(single_vert_intersection(vertices, faces, vertices[probe], directions[i], near_face_threshold)[0])
    return list(set(intersected_faces))

# def vert_neighbors(faces, face_cluster, inf_vertices):
#     '''
#     Given a set of faces, returns the vertices neighboring the faces that have infinite depth
#     '''
#     face_cluster = np.array(face_cluster)
#     faces = np.array(faces)
#     neighboring_vert_indices = list(set(list(faces[face_cluster].flatten())))
#     inf_vertices = set(inf_vertices)
#     # TODO: don't just check if it's in inf_vertices, check if there is a self intersection before the predicted depth.
#     neighbors = [x for x in neighboring_vert_indices if x in inf_vertices]
#     return neighbors

# def face_neighbors(faces, vert_cluster):
#     '''
#     Given a set of vertices, returns the faces neighboring the vertices
#     '''
#     faces = np.array(faces)
#     face_masks = []
#     for v in vert_cluster:
#         face_masks.append(np.any(faces==v, axis=1)[:, np.newaxis])
#     final_mask = np.any(np.hstack(face_masks), axis=1)

#     face_indices = [i for i in range(faces.shape[0]) if final_mask[i]]
#     return face_indices

def new_edges(all_vertices, new_vertices, vert2vert):
    '''
    Returns all edges (pairs of vertices) that contain one element of all_vertices and one element of new_vertices. 
    Edges are only valid if the two vertices are neighbors in the adjacency mapping (vert2vert)
    Note that new_vertices is a subset of all_vertices
    '''
    new_edges = []
    for v1 in new_vertices:
        neighbors = vert2vert[v1]
        for v2 in all_vertices:
            if v2 in neighbors:
                new_edges.append((v1,v2))
    return new_edges


def cluster_expansion(vertices, faces, cluster, inf_verts, directions, odf_depths, vert2vert, vert2face, face2vert):
    '''
    Takes a cluster of infinite depth vertices.
    Returns two clusters, each of both faces and vertices.
    Cluster A contains the vertices and neighboring faces around the input cluster.
    Cluster B is the faces intersected by the first cluster and their neighboring vertices.
    Directions should be the directions for ALL vertices!
    NOTE: faces and vertices are both returned as indices into their respective arrays
    '''
    near_face_threshold = rasterization.max_edge(vertices, faces)

    # TODO: make sure that inf_verts is a set
    # The two clusters (A&B)
    A_vertices = set([x for x in cluster])
    A_faces = set()
    B_vertices = set()
    B_faces = set()

    #items added to the clusters in the most recent iteration
    av_new = set([x for x in A_vertices])
    af_new = set()
    bv_new = set()
    bf_new = set()

    # Dictionary to store the self intersection depth of individual vertices so they don't get queried twice
    self_intersection_before_depth = {}

    #iterate while new things are still being added to the clusters
    while((len(av_new) + len(af_new) + len(bv_new) + len(bf_new)) > 0):
        # af_next = vert_cluster_intersections(vertices, faces, bv_new, [directions[i] for i in bv_new]) if len(bv_new) > 0 else []
        # af_next += face_neighbors(faces, av_new) if len(av_new) > 0 else []
        # af_next = list(set(af_next))

        # bf_next = vert_cluster_intersections(vertices, faces, av_new, [directions[i] for i in av_new]) if len(av_new) > 0 else []
        # bf_next += face_neighbors(faces, bv_new) if len(bv_new) > 0 else []
        # bf_next = list(set(bf_next))

        #UPDATE FACES

        # Add Face Condition 1 : Intersected by probe plane
        af_next = probe_edges_all_intersections(vertices, faces, directions, new_edges(B_vertices, bv_new, vert2vert))
        # Add Face Condition 2 : Adjacent to an added vertex
        for v in av_new:
            af_next = set.union(af_next, vert2face[v])

        # Add Face Condition 1 : Intersected by probe plane
        af_next = probe_edges_all_intersections(vertices, faces, directions, new_edges(B_vertices, bv_new, vert2vert))
        # Add Face Condition 2 : Adjacent to an added vertex
        for v in av_new:
            af_next = set.union(af_next, vert2face[v])


        #UPDATE VERTICES

        def new_vertices(new_faces, existing_vertices):
            #Add Vertex Condition 1: Neighboring added plane 
            neighbors = set()
            for f in new_faces:
                neighbors = set.union(neighbors, face2vert[f])
            #1.1 Neighbors added if they are inf depth (and haven't been added)
            new_verts = set.union(neighbors, inf_verts) - existing_vertices
            
            #1.2 TODO: Map inf verts to cluster sets and add all verts from the cluster

            #1.3 Neighbors added if their self intersection is closer than their odf depth
            for v in neighbors - existing_vertices - inf_verts:
                # check if this value has already been computed
                if v in self_intersection_before_depth:
                    if self_intersection_before_depth[v]:
                        new_verts = set.union(new_verts, v)
                else:
                    # Compute the depth of the first self intersection
                    # Use epsilon to avoid intersecting the current face
                    epsilon = 0.000001
                    ray_start = vertices[v] + epsilon * directions[v]
                    ray_end = vertices[v] + epsilon * directions[v]
                    rot_verts = rasterization.rotate_mesh(vertices, ray_start, ray_end)
                    _, self_intersection_depth = rasterization.ray_occ_depth(faces, rot_verts, ray_start_depth=np.linalg.norm(ray_end - ray_start), near_face_threshold=near_face_threshold)
                    intersect_before_odf_depth = self_intersection_depth < odf_depths[v]
                    self_intersection_before_depth[v] = intersect_before_odf_depth
                    if intersect_before_odf_depth:
                        new_verts.add(v)
            return new_verts        

        av_new = new_vertices(af_new, A_vertices)
        bv_new = new_vertices(bf_new, B_vertices)   

        # check to see which ones are actually new
        af_next = [x for x in af_next if x not in A_faces]
        bf_next = [x for x in bf_next if x not in B_faces]
        av_next = [x for x in av_next if x not in A_vertices]
        bv_next = [x for x in bv_next if x not in B_vertices]
        A_faces += af_next
        B_faces += bf_next
        A_vertices += av_next
        B_vertices += bv_next

        # update the new additions for the next iteration
        af_new = af_next
        bf_new = bf_next
        av_new = av_next
        bv_new = bv_next

    print("FINISHED CLUSTER EXPANSION")
    print(A_vertices)
    print(A_faces)
    print(B_vertices)
    print(B_faces)

    return A_vertices, A_faces, B_vertices, B_faces

def cluster_inf_probes(inf_indices, faces, vert2vert):
    '''
    Returns a list of lists of the provided indices. Each sublist represents a set of infinite depth vertices that are connected by mesh edges.
    '''
    clusters = []
    faces = np.array(faces)
    while len(inf_indices) != 0:
        # start with one of the inf vertices
        new_cluster = [inf_indices.pop()]
        new_indices = [new_cluster[0]]

        # as long as new neighbors are being added, keep iterating
        while len(new_indices) != 0:
            # find all the vertices neighboring our new indices
            neighbors = set()
            for x in new_indices:
                neighbors = set.union(neighbors, vert2vert[x])

            # update new indices for next round by assigning all neighbors that have inf depth to it
            new_indices = []
            for x in inf_indices:
                if x in neighbors:
                    new_indices.append(x)
                    new_cluster.append(x)

            # remove elements from inf indices if they were added to the current cluster
            inf_indices = [x for x in inf_indices if x not in neighbors]

        # add the new cluster to the list 
        clusters.append(new_cluster)
    return clusters
        

def update_step(model, vertices, faces, probes, directions, radius, delta):
    '''
    Updates vertex locations and fixes any self intersections in the mesh. Returns new vertices and faces.
    '''
    # we only care about the probe directions
    # directions = [x / np.linalg.norm(x) for x in directions]
    probe_directions = directions[-len(probes):]

    probe_depths = get_depths(model, probes, probe_directions, delta)
    inf_mask = [probe_depths[i] == np.inf for i in range(len(probe_depths))]
    show_subdivisions_and_probes(vertices, probes, inf_mask, probe_directions, faces, delta)

    # Cluster inf depth vertices together
    probes_offset = len(vertices) - len(probes)
    inf_indices = [i + probes_offset for i in range(len(probe_depths)) if probe_depths[i]==np.inf]
    vert2vert, vert2face, face2vert, face2edge, edge2face = odf_utils.mesh_adjacency_dictionaries(vertices, faces)
    clusters = cluster_inf_probes(inf_indices, faces, vert2vert)
    for c in clusters:
        c_probe = [x-probes_offset for x in c]
        A_vertices, A_faces, B_vertices, B_faces = cluster_expansion(vertices, faces, c, inf_indices, directions, delta)
        show_subdivisions_and_probes(vertices, probes, inf_mask, probe_directions, faces, delta, cluster=c_probe, c_verts=A_vertices, c_faces=A_faces, show_mesh_faces=False)
        show_subdivisions_and_probes(vertices, probes, inf_mask, probe_directions, faces, delta, cluster=c_probe, c_verts=B_vertices, c_faces=B_faces, show_mesh_faces=False)

    # TODO: hole repair

    new_vertices = [vertices[i] for i in range(probes_offset)] + [probes[i] + probe_depths[i]*probe_directions[i] if probe_depths[i] != np.inf else vertices[probes_offset + i] for i in range(len(probes))]

    return new_vertices, faces



def make_model_mesh(model, initial_tessalation_factor=3, radius=1.25, focal_point=[0.,0.,0.], show=True, iterations = 3, delta=0.08):
    focal_point = np.array(focal_point)
    vertices, faces = icosahedron_sphere_tessalation(radius, subdivisions=initial_tessalation_factor)
    faces = np.array(faces)
    ray_directions = [(focal_point-v) / np.linalg.norm(focal_point-v) for v in vertices]
    
    # if show:
    #     show_subdivisions_and_probes(vertices, vertices, ray_directions, faces, delta)
    vertices, faces = update_step(model, vertices, faces, vertices, ray_directions, radius, delta)

    if show:
        # can't import visualization on OSCAR because it uses Open3D and OpenGL
        import visualization
        import open3d as o3d
        o3d.visualization.draw_geometries([visualization.make_mesh(np.array(vertices), faces)])
    
    for i in range(iterations - 1):
        vertices, faces, probes = large_edge_subdivision(vertices, faces)
        directions = -1. * odf_utils.get_vertex_normals(np.array(vertices), np.array(faces))
        
        # if show:
        #     show_subdivisions_and_probes(vertices, probes, directions, faces, delta)

        probes = pull_back_probes(probes, directions, delta)
        vertices, faces = update_step(model, vertices, faces, probes, directions, radius, delta)

        if show:
            # can't import visualization on OSCAR because it uses Open3D and OpenGL
            import visualization
            import open3d as o3d
            o3d.visualization.draw_geometries([visualization.make_mesh(np.array(vertices), faces)])

    
    # TODO: save to file
    return vertices, faces
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View visualizations of the 3D meshing algorithm")
    parser.add_argument("-i", "--icosahedron", action="store_true", help="visualize an icosahedron-based sphere tessalation at various levels of subdivision")
    parser.add_argument("-p", "--pointcloud", action="store_true", help="visualize the pointcloud generated from the sphere looking inwards")
    parser.add_argument("-m", "--mesh", action="store_true", help="visualize the mesh generated from the sphere looking inwards")
    parser.add_argument("-r", "--repair", action="store_true", help="visualize mesh hole repair")
    parser.add_argument("--mesh_file", default="F:\\ivl-data\\sample_data\\stanford_bunny_watertight.obj", help="Source of mesh file")
    args = parser.parse_args()

    if args.icosahedron:
        import visualization
        radius = 1.25
        subdivisions = 5
        for i in range(subdivisions+1):
            vertices, faces = icosahedron_sphere_tessalation(radius=radius, subdivisions=i)
            print(f"Showing sphere after {i} subdivisions")
            print(f"Vertices: {len(vertices)}, Faces: {len(faces)}")
            viz = visualization.RayVisualizer(vertices, [])
            viz.add_mesh_faces(faces)
            viz.display()
    if args.pointcloud:
        import visualization
        # hyperparameters
        radius = 1.25
        subdivisions = 4

        # setup
        mesh = trimesh.load(args.mesh_file)
        faces = mesh.faces
        verts = mesh.vertices
        verts = odf_utils.mesh_normalize(verts)
        lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)
        visualizer = visualization.RayVisualizer(verts, lines)

        # Generate pointclouds
        sphere_vertices, _ = icosahedron_sphere_tessalation(radius=radius, subdivisions=subdivisions)
        pointclouds = sphere_surface_to_point_cloud(verts, faces, sphere_vertices)
        depth_layer_colors = [[52./255., 88./255., 235./255.], [51./255., 224./255., 25./255.], [223./255., 48./255., 242./255.], [46./255., 29./255., 22./255.]]
        for pc, color in zip(pointclouds, depth_layer_colors):
            for point in pc:
                visualizer.add_point(point, color)
            visualizer.display(show_wireframe=False)
    
    if args.mesh:
        import visualization
        import open3d as o3d
        # hyperparameters
        radius = 1.25
        subdivisions = 4

        # setup
        mesh = trimesh.load(args.mesh_file)
        faces = mesh.faces
        verts = mesh.vertices
        verts = odf_utils.mesh_normalize(verts)
        lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)

        # # Generate mesh
        # sphere_vertices, sphere_faces = icosahedron_sphere_tessalation(radius=radius, subdivisions=subdivisions)
        # mesh_vertices = sphere_surface_to_mesh(verts, faces, sphere_vertices)
        # # o3d.visualization.RenderOption(mesh_shade_option=1)
        # o3d.visualization.draw_geometries([visualization.make_mesh(mesh_vertices, sphere_faces, color=np.array([[1.,0.,0.]]))])


        gt_model = MeshODF(verts, faces)
        make_model_mesh(gt_model, initial_tessalation_factor=3)

    if args.repair:
        radius = 1.25
        subdivisions = 2
        vertices, faces = icosahedron_sphere_tessalation(radius=radius, subdivisions=subdivisions)
        inf_vert_index = 1
        direction = -1. * vertices[inf_vert_index] / np.linalg.norm(vertices[inf_vert_index])
        direction += np.array([0.01,0.005,0.012])
        non_intersecting_vertices = [inf_vert_index]
        # can't import visualization on OSCAR because it uses Open3D and OpenGL
        import visualization
        import open3d as o3d
        o3d.visualization.draw_geometries([visualization.make_mesh(np.array(vertices), faces)])
        vertices, faces, _, _ = recompute_mesh_connectivity(vertices, faces, inf_vert_index, direction, non_intersecting_vertices)
        o3d.visualization.draw_geometries([visualization.make_mesh(np.array(vertices), faces)])