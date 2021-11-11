from os import remove
from typing import final
from matplotlib.pyplot import connect
import numpy as np
import trimesh
import argparse
import torch

import rasterization
import odf_utils
from visualization import RayVisualizer

# TODO: selectively import this so this file can be used on Oscar
import visualization
import open3d as o3d
from datetime import datetime

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

# mesh_file = "F:\\ivl-data\\sample_data\\stanford_bunny_watertight.obj"
mesh_file = "F:\\ivl-data\\sample_data\\simple_car_fixed.obj"
mesh = trimesh.load(mesh_file)

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
    print(f"Edges shape: {edges.shape}")
    print(f"Verts shape: {verts.shape}")
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



def show_subdivisions_and_probes(vertices, probes, inf_mask, directions, faces, delta, cluster={}, show_mesh_faces=True, show_new_lines=True, show_probes=True):
    '''
    For visualization purposes.
    Shows which edges have been subdivided and where the probe locations are.
    '''
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
    
    if show_new_lines:
        for i in range(new_lines.shape[0]):
            visualizer.add_ray([new_lines[i,0,:], new_lines[i,1,:]], np.array([1.,0.,0.]))

    if show_probes:
        for i in range(probes.shape[0]):
            visualizer.add_ray([probes[i], probes[i]+delta*directions[i]], np.array([0.,0.,1.]))
            if inf_mask[i]:
                if i+probes_offset in cluster:
                    visualizer.add_point(probes[i], np.array([1.,0.,1.]))
                else:
                    visualizer.add_point(probes[i], np.array([1.,0.,0.]))
            else:
                visualizer.add_point(probes[i], np.array([0.,1.,0.]))
    
    # for x in {898, 899, 901, 902, 903, 1825, 1826, 1830, 807, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1227, 1228, 1230, 1232, 1235, 1236, 1631, 1632, 1634, 1637, 1638, 1639, 1641, 1643, 1646, 885, 888, 890, 894, 895}:
    #     print(x)
    #     visualizer.add_point(vertices[x], [1.,0.,0.])
    visualizer.display(show_wireframe=True)

def get_depths(model, probes, directions, delta=0.08):
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
        model_depths = [x if x != np.inf else (np.inf if confirmed_inf.pop() else delta) for x in model_depths]
        # model_depths = [x if x != np.inf else np.inf for x in model_depths]

    return model_depths

def pull_back_probes(probes, directions, delta):
    '''
    Pulls probes back from the object surface. Move in the opposite direction from direction, with a magnitude of delta.
    '''
    directions = directions[-len(probes):]
    probes = [probes[i]- delta*directions[i] for i in range(len(probes))]
    return probes

def get_cluster_bounding_edges(faces, cluster):
    '''
    Given mesh faces and a set of vertices in the cluster, returns a set of directed edges that minimally encircle the specified cluster
    '''
    faces = np.array(faces)
    faces_in_cluster = np.any(np.stack([faces==v for v in cluster]), axis=0)
    relevant_indices = np.sum(faces_in_cluster, axis=1) == 1
    relevant_faces = faces[relevant_indices]
    
    edges = []
    for f in range(relevant_faces.shape[0]):
        cluster_index = 0
        if relevant_faces[f,1] in cluster:
            cluster_index = 1
        if relevant_faces[f,2] in cluster:
            cluster_index = 2
        edges.append((relevant_faces[f,(cluster_index+1)%3], relevant_faces[f, (cluster_index+2)%3]))
    return edges
    

def fix_inconsistent_vertices(model, inconsistent_vertices, vertices, faces, vert2vert, new_positions, show_clusters=False):
    '''
    For each inconsistent vertex, finds the new probe direction, queries it, and updates the vertex in new_points
    '''


    if show_clusters:
        gt_faces = mesh.faces
        gt_verts = mesh.vertices
        gt_verts = odf_utils.mesh_normalize(gt_verts)
        gt_lines = np.concatenate([gt_faces[:,:2], gt_faces[:,1:], gt_faces[:,[0,2]]], axis=0)


    all_probe_directions = {}
    # cluster the inconsistent vertices
    clusters = cluster_inconsistent_probes(inconsistent_vertices, vert2vert)
    # get the anchors and calculate the probe directions for each cluster
    anchors = []
    for c in clusters:
        new_anchors, anchor_sum, vert_anchors = cluster_weighted_anchors(c, vert2vert, new_positions)
        anchors.append({k:v/anchor_sum for k,v in new_anchors.items()})
        # new_probe_directions, new_probe_looks = new_cluster_probes(c, anchors, vert2vert, vertices, new_positions)
        # all_probe_directions.update(new_probe_directions)

        

        # visualize the cluster, anchors, and new probe directions
        # if show_clusters:
        #     assert(faces is not None)
        #     faces = np.array(faces)
        #     lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)
        #     visualizer = visualization.RayVisualizer(gt_verts, gt_lines)

        #     for v in new_probe_looks:
        #         visualizer.add_ray([vertices[v], new_probe_looks[v]], [0.,0.,1.])
        #     for a in anchors:
        #         visualizer.add_point(new_positions[a], anchors[a]/anchor_sum*np.array([0.,1.,1.]))
        #         visualizer.add_ray([vertices[a], new_positions[a]], [1., 0., 0.])
        #         visualizer.add_ray([vertices[a], new_positions[a]], [1., 0., 0.])
        #     for v in vert_anchors:
        #         visualizer.add_ray([vertices[v],vertices[vert_anchors[v]]], [0.,1.,0.])

        #     # show the bounding edges of the cluster
        #     bounding_edges = get_cluster_bounding_edges(faces, c)
        #     for edge in bounding_edges:
        #         visualizer.add_ray([vertices[edge[0]], vertices[edge[1]]], [1.,0.,1.])
        #     visualizer.display()

    vertices, faces, new_probe_indices, new_looks, new_positions = cluster_collapse(vertices, faces, clusters, anchors, new_positions)

    if show_clusters:
        faces = np.array(faces)
        lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)
        visualizer = visualization.RayVisualizer(gt_verts, gt_lines)

        new_looks = np.array(new_looks)

        for i in range(len(new_probe_indices)):
            visualizer.add_ray([vertices[new_probe_indices[i]], vertices[new_probe_indices[i]]+new_looks[i]], [0.,0.,1.])
        
        visualizer.display()




    # query the ODF for the depth from the new probe and direction
    # ordered_inconsistent_verts = [x for x in inconsistent_vertices]
    # probe_locations = np.array([vertices[i] for i in ordered_inconsistent_verts])
    # probe_directions = np.array([all_probe_directions[i] for i in ordered_inconsistent_verts])


    depths = get_depths(model, np.array([vertices[i] for i in new_probe_indices]), np.array(new_looks))
    depth_dict = {new_probe_indices[i]: depths[i] for i in range(len(new_probe_indices))}
    dir_dict = {new_probe_indices[i]: new_looks[i] for i in range(len(new_probe_indices))}
    pos_dict = {i: vertices[i] + depth_dict[i] * dir_dict[i] for i in new_probe_indices if depth_dict[i] != np.inf}
    new_positions = [new_positions[i] if i not in pos_dict else pos_dict[i] for i in range(len(new_positions))]

    # can't import visualization on OSCAR because it uses Open3D and OpenGL
    # o3d.visualization.draw_geometries([visualization.make_mesh(np.array(new_positions), faces)])

    # TODO: set inconsistent verts not in pos_dict to be linear interpolations of neighbors (weighted?)
    # TODO: requery the non-updated points from a slightly different viewing direction (maintain non-intersecting mesh)
    print(np.array(new_positions).shape)
    return np.array(new_positions), np.array(faces, dtype=int)

def cluster_collapse(vertices, faces, clusters, anchors, new_positions):
    '''
    Collapses mesh clusters into single points based on the average position of the points in the cluster
    Returns the new faces and vertices, as well as the probe positions and look vectors for the collapsed clusters
    '''
    vertices = np.array(vertices)
    faces = np.array(faces)
    new_positions = np.array(new_positions)
    bounding_edges = [get_cluster_bounding_edges(faces, c) for c in clusters]
    removed_indices = np.zeros(vertices.shape[0])
    all_cluster_verts = [v for c in clusters for v in c]
    removed_indices[all_cluster_verts] = 1
    
    #Convert old vertex indices to new indices. Undefined values for vertices which were deleted
    old_indices_to_new = np.arange(vertices.shape[0]) - np.cumsum(removed_indices)

    # remove faces and vertices
    new_vertices = vertices[np.logical_not(removed_indices)]
    new_positions = new_positions[np.logical_not(removed_indices)]
    new_faces = faces[np.logical_not(np.any(np.any(np.stack([faces==v for v in all_cluster_verts]), axis=0), axis=1))]
    new_faces = old_indices_to_new[new_faces]

    # add cluster centers as vertices and connect them to their bounding edges
    new_cluster_vertices = []
    new_cluster_faces = []
    new_probe_indices = []
    new_looks = []
    for i in range(len(clusters)):
        new_center = np.mean(vertices[list(clusters[i])], axis=0)
        new_cluster_vertices.append(new_center)
        new_probe_indices.append(i + len(new_vertices))
        new_look_point = np.array([0.,0.,0.])
        for p, w in anchors[i].items():
            new_look_point += w*vertices[p]
        new_looks.append((new_look_point-new_center)/np.linalg.norm(new_look_point-new_center))
        for edge in bounding_edges[i]:
            new_cluster_faces.append([old_indices_to_new[edge[0]], old_indices_to_new[edge[1]], new_vertices.shape[0]+i])
    
    new_vertices = np.concatenate([new_vertices, new_cluster_vertices])
    new_faces = np.concatenate([new_faces, new_cluster_faces])
    new_positions = np.concatenate([new_positions, new_cluster_vertices])



    return new_vertices, new_faces, new_probe_indices, new_looks, new_positions






def new_cluster_probes(cluster, anchors, vert2vert, vertices, new_points):
    '''
    Returns new probe directions for points in the cluster, based on the anchor points and vertex distances
    '''
    # TODO figure out how to weight the anchors according to both their graph distances and euclidean distance

    anchor_dists = {v: {} for v in cluster}
    
    for v in cluster:
        # Run Dijkstra's for each element in the cluster to get the distance to all anchor points
        
        dijkstra_queue = {x for x in cluster}
        dist = {x: np.inf for x in dijkstra_queue}
        dist[v] = 0

        while len(dijkstra_queue) > 0:
            # Find the next point to look at
            lowest_dist = np.inf
            next_point = None
            for x in dijkstra_queue:
                if lowest_dist == np.inf:
                    next_point = x
                    lowest_dist = dist[x]
                elif dist[x] < lowest_dist:
                    next_point = x
                    lowest_dist = dist[x]
            dijkstra_queue -= {next_point}

            # update distance to neighbors
            for x in set.intersection(vert2vert[next_point], cluster):
                if dist[next_point] + 1 < dist[x]:
                    dist[x] = dist[next_point] + 1
        
        # Calculate the distance from v to each anchor point
        for point in cluster:
            for anchor in set.intersection(vert2vert[point], set(anchors)):
                if anchor not in anchor_dists[v]:
                    anchor_dists[v][anchor] = dist[point] + 1
                elif dist[point] + 1 < anchor_dists[v][anchor]:
                    anchor_dists[v][anchor] = dist[point] + 1

    # calculate the look direction for each of the points in the cluster (using the anchor weights, distances, and positions)
    new_probe_directions = {}
    new_probe_look_points = {}
    for v in cluster:
        anchor_weight_sum = 0.0
        for a in anchors:
            anchor_weight_sum += (1./anchor_dists[v][a])*anchors[a]
        look_point = np.array([0.,0.,0.])
        for a in anchors:
            look_point += new_points[a]*(1./anchor_dists[v][a])*anchors[a]/anchor_weight_sum
        new_probe_look_points[v] = look_point
        new_probe_directions[v] = look_point - vertices[v]
        new_probe_directions[v] /= np.linalg.norm(new_probe_directions[v])
    
    return new_probe_directions, new_probe_look_points


def cluster_weighted_anchors(cluster, vert2vert, new_positions):
    '''
    Returns a dictionary that maps vertex indices to weights
    the weights correspond to how strongly that vertex's new position pulls the cluster probes towards it
    '''
    anchors = {}
    # factor to accentuate the importance of max min cos
    K = 2.
    anchor_sum = 0.

    # map cluster vertices to their anchors
    vert_to_anchor_edges = {}

    # each vertex in the cluster can add one anchor
    for v in cluster:
        new_anchor = None
        max_min_cos = -2.0
        # check all of the neighboring vertices as potential anchor candidates
        for neighbor in vert2vert[v] - cluster:
            min_cos = 2.0
            neighbor_vec = new_positions[neighbor] - new_positions[v]
            # check all other neighbors to see what the minimum cosine is for this candidate
            for other_neighbor in vert2vert[v] - cluster - {neighbor}:
                other_neighbor_vec = new_positions[other_neighbor] - new_positions[v]
                new_cos = np.dot(neighbor_vec, other_neighbor_vec) / (np.linalg.norm(neighbor_vec) * np.linalg.norm(other_neighbor_vec))
                if new_cos < min_cos:
                    min_cos = new_cos
            if min_cos < 2.0 and min_cos > max_min_cos:
                max_min_cos = min_cos
                new_anchor = neighbor
        if new_anchor is not None:
            anchor_sum += ((1. + max_min_cos) / 2.0) ** K
            if new_anchor not in anchors:
                anchors[new_anchor] = ((1. + max_min_cos) / 2.0) ** K
            else:
                anchors[new_anchor] += ((1. + max_min_cos) / 2.0) ** K

            vert_to_anchor_edges[v] = new_anchor
    
    return anchors, anchor_sum, vert_to_anchor_edges

def cluster_inconsistent_probes(inconsistent_indices, vert2vert):
    '''
    Returns a list of sets of the provided indices. Each sublist represents a set of infinite depth vertices that are connected by mesh edges.
    '''
    # don't want to change the actual data structure that was passed in
    inconsistent_copy = [x for x in inconsistent_indices]
    clusters = []
    while len(inconsistent_copy) != 0:
        # start with one of the inf vertices
        new_cluster = [inconsistent_copy.pop()]
        new_indices = [new_cluster[0]]

        # as long as new neighbors are being added, keep iterating
        while len(new_indices) != 0:
            # find all the vertices neighboring our new indices
            neighbors = set()
            for x in new_indices:
                neighbors = set.union(neighbors, vert2vert[x])

            # update new indices for next round by assigning all neighbors that have inf depth to it
            new_indices = []
            for x in inconsistent_copy:
                if x in neighbors:
                    new_indices.append(x)
                    new_cluster.append(x)

            # remove elements from inf indices if they were added to the current cluster
            inconsistent_copy = [x for x in inconsistent_copy if x not in neighbors]

        # add the new cluster to the list 
        clusters.append(set(new_cluster))
    return clusters

def get_inconsistent_probes(vertices, faces, probes, probe_directions, odf_depths, probes_offset):
    '''
    Returns probes that overlap with their neighbors, leading to self intersections.
    '''
    epsilon = 0.000001
    vertices = np.array(vertices)
    faces = np.array(faces)
    rho = 0.1
    near_face_threshold = rasterization.max_edge(vertices, faces)
    # Currently: return any probes with infinite depth
    inf_depth_probes = set([i + probes_offset for i in range(len(probe_directions)) if odf_depths[i] == np.inf])
    # TODO: also return any probes that overlap with their neighbors

    self_intersecting_probes = set()
    for i in range(len(probe_directions)):
        if odf_depths[i] > rho:
            vi = i+probes_offset
            rot_verts = rasterization.rotate_mesh(vertices, vertices[i+probes_offset] + epsilon*probe_directions[i], vertices[i+probes_offset] + probe_directions[i])
            _, depth = rasterization.ray_occ_depth(faces, rot_verts, ray_start_depth=np.linalg.norm(probe_directions[i])-epsilon, near_face_threshold=near_face_threshold)
            if depth < odf_depths[i]:
                self_intersecting_probes.add(vi)
        
    inconsistent_probes = set.union(inf_depth_probes, self_intersecting_probes)
    # inconsistent_probes = inf_depth_probes

    return inconsistent_probes



        

def update_step(model, vertices, faces, probes, directions, delta, show_clusters=False):
    '''
    Updates vertex locations and fixes any self intersections in the mesh. Returns new vertices and faces.
    '''
    # Hyperparameters
    REFINEMENT_RATE = 0.5
    K = 2

    # we only care about the probe directions
    # directions = [x / np.linalg.norm(x) for x in directions]
    probe_directions = directions[-len(probes):]
    probes_offset = len(vertices) - len(probes)

    # graph properties that will be used during probe refinement
    vert2vert = odf_utils.get_vertex_adjacencies(vertices, faces)
    # neighbor_weights = get_neighbor_weights(vertices, vert2vert, probes_offset, K)


    probe_depths = get_depths(model, probes, probe_directions)

    # STEP 1: Identify inconsistent probes
    inconsistent_probes = get_inconsistent_probes(vertices, faces, probes, probe_directions, probe_depths, probes_offset)
    inconsistent_mask = [True if i+probes_offset in inconsistent_probes else False for i in range(len(probes))]

    # TODO: Gradually phase out update probes based on how close they are to the originally inconsistent ones
    # update_probes = [x for x in inconsistent_probes]

    new_positions = [vertices[i] for i in range(probes_offset)] + [probes[i-probes_offset] + probe_depths[i-probes_offset]*probe_directions[i-probes_offset] if not i in inconsistent_probes else vertices[i] for i in range(probes_offset, len(vertices))]
    print(f"Initial new positions: {len(new_positions)}")    

    if show_clusters:
        clusters = cluster_inconsistent_probes(inconsistent_probes, vert2vert)
        for c in clusters:
        #     anchors, anchor_sum, vert_anchors = cluster_weighted_anchors(c, vert2vert, new_positions)
        #     dirs, looks = new_cluster_probes(c, anchors, vert2vert, vertices, new_positions)
        #     faces = np.array(faces)
        #     lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)
        #     visualizer = visualization.RayVisualizer(vertices, lines)
        #     for v in looks:
        #         visualizer.add_ray([vertices[v], looks[v]], [0.,0.,1.])
        #     for a in anchors:
        #         visualizer.add_point(new_positions[a], anchors[a]/anchor_sum*np.array([0.,1.,1.]))
        #     visualizer.display()
            show_subdivisions_and_probes(vertices, probes, inconsistent_mask, probe_directions, faces, delta, cluster=c)

    print(np.array(new_positions).shape)
    new_positions, faces = fix_inconsistent_vertices(model, inconsistent_probes, vertices, faces, vert2vert, new_positions, show_clusters=show_clusters) if len(inconsistent_probes) > 0 else (new_positions, faces)
    print(f"FIX INCONSISTENT RESULTS-")
    print(f"use fix_inc {len(inconsistent_probes) > 0}")
    print(np.array(new_positions).shape)

    return np.array(new_positions), faces



def make_model_mesh(model, initial_tessalation_factor=3, radius=1.25, focal_point=[0.,0.,0.], show=True, iterations = 4, delta=0.08):
    start_time = datetime.now()
    focal_point = np.array(focal_point)
    vertices, faces = icosahedron_sphere_tessalation(radius, subdivisions=initial_tessalation_factor)
    faces = np.array(faces)
    ray_directions = [(focal_point-v) / np.linalg.norm(focal_point-v) for v in vertices]
    
    # if show:
    #     show_subdivisions_and_probes(vertices, vertices, ray_directions, faces, delta)
    vertices, faces = update_step(model, vertices, faces, vertices, ray_directions, delta, show_clusters=show)

    if show:
        # can't import visualization on OSCAR because it uses Open3D and OpenGL
        o3d.visualization.draw_geometries([visualization.make_mesh(np.array(vertices), faces)])
    
    print(f"iterations: {iterations}")
    for i in range(iterations - 1):
        vertices, faces, probes = large_edge_subdivision(vertices, faces)
        directions = -1. * odf_utils.get_vertex_normals(np.array(vertices), np.array(faces))
        
        # if show:
        #     show_subdivisions_and_probes(vertices, probes, directions, faces, delta)

        probes = pull_back_probes(probes, directions, delta)
        vertices, faces = update_step(model, vertices, faces, probes, directions, delta, show_clusters=show)

        # if show:
        #     # can't import visualization on OSCAR because it uses Open3D and OpenGL
        #     o3d.visualization.draw_geometries([visualization.make_mesh(np.array(vertices), faces)])
        print(f"{i+1}/{iterations-1}")
    
    end_time = datetime.now()

    elapsed_time = end_time - start_time
    print(f'elapsed time: {elapsed_time}')

    o3d.visualization.draw_geometries([visualization.make_mesh(np.array(vertices), faces)])

    
    # TODO: save to file
    return vertices, faces
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View visualizations of the 3D meshing algorithm")
    parser.add_argument("-i", "--icosahedron", action="store_true", help="visualize an icosahedron-based sphere tessalation at various levels of subdivision")
    parser.add_argument("-p", "--pointcloud", action="store_true", help="visualize the pointcloud generated from the sphere looking inwards")
    parser.add_argument("-m", "--mesh", action="store_true", help="visualize the mesh generated from the sphere looking inwards")
    parser.add_argument("-r", "--repair", action="store_true", help="visualize mesh hole repair")

    # mesh extraction parameters
    parser.add_argument("-t", "--tesselation_subdivisions", type=int, default=3, help="The number of times to subdivide the initial sphere tesselation")
    parser.add_argument("-c", "--update_cycles", type=int, default=3, help="The number of times to update the mesh vertices")
    parser.add_argument("--delta", type=float, default=0.08, help="The amount to pull the probe points back by")
    parser.add_argument("--rho", type=float, default=0.10, help="The depth threshold above which points should be checked for self intersection")
    parser.add_argument("--edge_subdivision_threshold", type=float, default=0.05, help="The edge length threshold, above which an edge will be subdivided")

    parser.add_argument("-s", "--show", action="store_true", help="visualize the mesh creation process")
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
        # mesh = trimesh.load(args.mesh_file)
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
        make_model_mesh(gt_model, initial_tessalation_factor=args.tesselation_subdivisions, iterations=args.update_cycles, show=args.show)