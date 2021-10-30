from typing import final
from matplotlib.pyplot import connect
import numpy as np
import trimesh
import argparse
import torch

import rasterization
import utils

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
            if utils.get_sphere_intersections(points[i], directions[i], self.radius) is None:
                lines = np.concatenate([self.faces[:,:2], self.faces[:,1:], self.faces[:,[0,2]]], axis=0)
                visualizer = visualization.RayVisualizer(self.vertices, lines)
                visualizer.add_point(points[i], [1.0,0.0,0.0])
                visualizer.display()
            start_point, end_point = utils.get_sphere_intersections(points[i], directions[i], self.radius)
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

# ################### RECONNECTING MESH ###################

def neighboring_vert_indices(original_index, lines):
    neighboring_verts = []
    for i in range(lines.shape[0]):
        if lines[i][0] == original_index:
            neighboring_verts.append(lines[i][1])
        elif lines[i][1] == original_index:
            neighboring_verts.append(lines[i][0])
    neighboring_verts = set(neighboring_verts)
    return neighboring_verts

def vector_angle_3d_plane(u_vector, v_vector, vec):
    '''
    Returns the angle between u_vector and vec from 0 to 2*PI
    '''
    # scalar projection of vec onto the u-axis of the plane
    u_component = np.dot(vec, u_vector) / np.linalg.norm(u_vector)
    # scalar projection of vec onto the v-axis of the plane
    v_component = np.dot(vec, v_vector) / np.linalg.norm(v_vector)
    # print("ANGLE VEC COMPONENTS")
    # print(u_component)
    # print(v_component)

    # get the angle (in the UV plane) between vec and the u-axis
    return np.arctan2(v_component, u_component)

def reconnect_neighbors(vertices, faces, inf_vert_index, inf_vert_direction):
    '''
    Takes a mesh, a vertex that had an infinite depth sample, and the sample direction
    Returns an updated list of faces where some of the neighboring vertices in the mesh have been reconnected to avoid self intersection.
    Also returns and ordered list of a subset of the neighboring vertices
    '''
    neighboring_faces = faces[np.any(faces==inf_vert_index, axis=1)]
    neighboring_edges = neighboring_faces[neighboring_faces!=inf_vert_index].reshape((-1,2))
    
    root_point = vertices[inf_vert_index]
    scaled_direction = inf_vert_direction * (np.linalg.norm(inf_vert_direction)**2)
    # returns the component of the vector perpendicular to the inf_vert_direction vector
    def get_perpendicular_vector(point):
        return point - np.dot(point - root_point, inf_vert_direction) * scaled_direction

    root_angle_vector = get_perpendicular_vector(vertices[neighboring_edges[0,0]])
    # returns the angle between the perpendicular vector fo the provided point, and the root angle vector
    def get_angle(point):
        point_perp_vec = get_perpendicular_vector(point)
        theta = np.arccos()
    # TODO: finish method

    return None

def connect_holes(vertices, faces, inf_vert_index, inf_vert_direction, intersected_face):
    '''
    Connects the neighboring vertices of the inf_vert_index to the vertices of the intersected face
    Removes faces containing inf_vert_index as well as the intersected face
    Returns the updated faces and vertices, and the original vertices index of any extra vertex that might have had to be removed
    '''

    # TODO: call reconnect_neighbors

    if inf_vert_index in list(faces[intersected_face]):
        print("BAD FACE CONNECTION")
        print(f"VERT: {inf_vert_index}")
        print(f"FACE: {faces[intersected_face]}")

        # import visualization
        # lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)
        # visualizer = visualization.RayVisualizer(vertices, lines)
        # visualizer.add_point(vertices[inf_vert_index], [1.0,0.0,0.0])
        # visualizer.add_colored_mesh(vertices[faces[intersected_face]], np.array([[0,1,2]]), np.array([[0.0,1.0,0.0]]))
        # visualizer.display()


    # faces that border the point being removed
    neighboring_faces = faces[np.any(faces==inf_vert_index, axis=1)]
    # The number of vertices on the intersected face that also neighbor the vertex being removed
    bordering_intersected = np.any(neighboring_faces==faces[intersected_face, 0], axis=1).astype(int) \
                            + np.any(neighboring_faces==faces[intersected_face, 1], axis=1).astype(int) \
                            + np.any(neighboring_faces==faces[intersected_face, 2], axis=1).astype(int)
    # This is a mask over the faces neighboring inf_vert_index that indicates whether the face shares an edge with the intersected face
    bordering_intersected = bordering_intersected > 1
    print(f"BORDERING INTERSECTED: {bordering_intersected}")

    # handle the case where the intersected face borders 1 or 2 of the faces that border the non-intersecting vertex
    n_shared_edges = np.sum(bordering_intersected)
    if n_shared_edges > 0:
        print("CASE 1")
        # edges of removed faces that are still part of other faces and border the gap left by the removal
        gap_edges = []
        # add edges from the neighboring faces
        for i in range(neighboring_faces.shape[0]):
            if not bordering_intersected[i]:
                # make sure that the edge is added in the correct order so that the new faces will have the right normal
                if neighboring_faces[i,0] == inf_vert_index:
                    gap_edges.append([neighboring_faces[i,1], neighboring_faces[i,2]])
                if neighboring_faces[i,1] == inf_vert_index:
                    gap_edges.append([neighboring_faces[i,2], neighboring_faces[i,0]])
                if neighboring_faces[i,2] == inf_vert_index:
                    gap_edges.append([neighboring_faces[i,0], neighboring_faces[i,1]])
        
        # second removed vert is one of the vertices of the intersected face when the intersected face shares two borders with the neighboring faces
        second_removed_vert = None
        gap_edge_verts = set(np.array(gap_edges).flatten())
        # add edge(s) from the intersected face (only ones that don't border the neighboring faces)        
        for i in range(3):
            if n_shared_edges == 2 and (not faces[intersected_face, i] in gap_edge_verts):
                second_removed_vert = faces[intersected_face, i]
            edge = [faces[intersected_face, i], faces[intersected_face, (i+1)%3]]
            # the edge in the intersected face will either share 1 or two vertices with the existing gap edges
            # we want 1 vertex if there is 1 shared edge, 2 vertices if there are two shared edges
            n_shared_vertices = sum([x in gap_edge_verts for x in edge])
            if n_shared_edges == n_shared_vertices:
                gap_edges.append(edge)

        # remove the intersected face and faces that border the non intersecting vertex
        faces = np.vstack([faces[:intersected_face,:], faces[intersected_face+1:,:]])
        faces = faces[np.logical_not(np.any(faces == inf_vert_index, axis=1))]

        # ------- Remove vertex and add new faces --------------
 
        gap_edge_verts = set(np.array(gap_edges).flatten())
        # take the vertices around the missing faces, average them to form a new vertex, and connect the edges around the missing faces to this new vertex
        new_vertex = np.mean(vertices[list(gap_edge_verts)], axis=0)
        new_vertex_index = vertices.shape[0]
        vertices = np.vstack([vertices, [new_vertex]])
        new_faces = [[new_vertex_index] + edge for edge in gap_edges]
        # for edge in gap_edges:
        #     if new_vertex_index in edge:
        #         print("++++++++++ Linear Face Added! +++++++++++++")
        #         print(gap_edges)
        #         print(inf_vert_index)
        faces = np.vstack([faces, np.array(new_faces)])

        # remove the non intersecting vertex and update faces
        vertices = np.vstack([vertices[:inf_vert_index, :], vertices[inf_vert_index+1:,:]])
        faces[faces > inf_vert_index] -= 1

        # remove the second vertex if necessary
        if second_removed_vert is not None:
            # adjust the second vert index if it is greater than the previously removed index
            adjusted_srv = second_removed_vert if second_removed_vert < inf_vert_index else second_removed_vert - 1
            vertices = np.vstack([vertices[:adjusted_srv, :], vertices[adjusted_srv+1:, :]])
            faces[faces > adjusted_srv] -= 1
        
        return vertices, faces, second_removed_vert

    # the case where the intersected face and the neighboring faces of the non-intersecting vertex do not share any edges
    else:
        print("CASE 2")
        # edges of removed faces that are still part of other faces and border the gap left by the removal
        gap_edges = []
        # add edges from the neighboring faces
        for i in range(neighboring_faces.shape[0]):
            # make sure that the edge is added in the correct order so that the new faces will have the right normal
            if neighboring_faces[i,0] == inf_vert_index:
                gap_edges.append([int(neighboring_faces[i,1]), int(neighboring_faces[i,2])])
            if neighboring_faces[i,1] == inf_vert_index:
                gap_edges.append([int(neighboring_faces[i,2]), int(neighboring_faces[i,0])])
            if neighboring_faces[i,2] == inf_vert_index:
                gap_edges.append([int(neighboring_faces[i,0]), int(neighboring_faces[i,1])])
        
        # intersected_edges = []
        # for i in range(3):
        #     intersected_edges.append([faces[intersected_face, i], faces[intersected_face, (i+1)%3]])
        intersected_edges = [[int(faces[intersected_face, i]), int(faces[intersected_face, (i+1)%3])] for i in range(3)]
        print("GAP EDGES")
        print(gap_edges)
        print("INTERSECTED EDGES")
        print(intersected_edges)

        # choose arbitrary gap edge vertex to be the zero angle, and take the perpendicular component of it
        u_vector = vertices[gap_edges[0][0]] - vertices[inf_vert_index]
        u_vector -= np.dot(u_vector, inf_vert_direction) * inf_vert_direction / (np.linalg.norm(inf_vert_direction)**2)
        u_vector /= np.linalg.norm(u_vector)
        v_vector = np.cross(inf_vert_direction, u_vector)
        v_vector /= np.linalg.norm(v_vector)

        # store the angles of the relevant vertex indices
        vertex_angles = {int(vi): vector_angle_3d_plane(u_vector, v_vector, vertices[vi]-vertices[inf_vert_index]) for vi in list(set(list(np.array(gap_edges+intersected_edges).flatten())))}

        gap_edge_vertices = list(set(list(np.array(gap_edges).flatten())))
        intersected_face_vertices = [int(faces[intersected_face, 0]), int(faces[intersected_face, 1]), int(faces[intersected_face, 2])]
        new_faces = []

        # track the intersect vertices that have faces with the gap edge vertices
        int_vert_connections = {v: [] for v in gap_edge_vertices}

        import visualization
        import open3d as o3d
        lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)
        visualizer = visualization.RayVisualizer(vertices, lines)
        visualizer.add_mesh_faces(list(faces))
        visualizer.display()
        visualizer.add_colored_mesh(vertices[faces[intersected_face]], np.array([[0,1,2]]), np.array([[1.,0.,0.]]))
        # o3d.visualization.draw_geometries([visualization.make_mesh(np.array(vertices), faces)])
        visualizer.add_ray([vertices[inf_vert_index], vertices[inf_vert_index]+ u_vector], [0.,1.,0.])
        visualizer.add_ray([vertices[inf_vert_index], vertices[inf_vert_index]+ v_vector], [0.,0.,1.])
        d = 0.
        for i in intersected_face_vertices:
            d += 0.3
            
            u_component = np.dot(vertices[i], u_vector) / np.linalg.norm(u_vector)
            # scalar projection of vec onto the v-axis of the plane
            v_component = np.dot(vertices[i], v_vector) / np.linalg.norm(v_vector)
            offset_vec = u_component*u_vector + v_component * v_vector
            visualizer.add_ray([vertices[inf_vert_index], vertices[inf_vert_index]+ offset_vec], [1.,0+d,0.+d])
            visualizer.add_ray([vertices[inf_vert_index], vertices[i]], [1.,0.,1.])
        visualizer.add_ray([vertices[inf_vert_index], vertices[inf_vert_index] + 2.* inf_vert_direction], [0.,1.,1.])
        visualizer.display()

        # print("GAP EDGE midpoints")
        for edge in gap_edges:
            midpoint_angle = vector_angle_3d_plane(u_vector, v_vector, ((vertices[edge[0]] + vertices[edge[1]])/2.)-vertices[inf_vert_index])
            # print(midpoint_angle)
            # get the difference (in angle) between the midpoint of the edge and each vertex on the intersected face
            # also compare with 2*PI + angle if the vertex angle is larger or -2*PI + angle if the vertex angle is smaller
            angle_diffs = [min(abs(vertex_angles[intersected_face_vertices[j]]-midpoint_angle), abs(vertex_angles[intersected_face_vertices[j]] - ((-1. if vertex_angles[intersected_face_vertices[j]] < midpoint_angle else 1.) *2*np.pi + midpoint_angle))) for j in range(len(intersected_face_vertices))]
            start_vertex = intersected_face_vertices[np.argmin(angle_diffs)]
            # store the new adjacencies
            int_vert_connections[edge[0]].append(start_vertex)
            int_vert_connections[edge[1]].append(start_vertex)
            if not start_vertex in list(edge):
                new_faces.append([start_vertex, edge[0], edge[1]])
        # print("INT VERT angles")
        # for v in intersected_face_vertices:
            # print(vertex_angles[v])

        for vert in gap_edge_vertices:
            int_vert_neighbors = list(set(int_vert_connections[vert]))
            # if there are two intersected face vertices that 
            if len(int_vert_neighbors) > 1:
                assert(len(int_vert_neighbors) < 3)
                for edge in intersected_edges:
                    if int_vert_neighbors[0] in edge and int_vert_neighbors[1] in edge:
                        if not vert in list(edge):
                            new_faces.append([vert] + edge)

        # for edge in intersected_edges:
        #     midpoint_angle = vector_angle_3d_plane(u_vector, v_vector, ((vertices[edge[0]] + vertices[edge[1]])/2.)-vertices[inf_vert_index])
        #     # get the difference (in angle) between the midpoint of the edge and each vertex on on the gap edges
        #     # also compare with 2*PI + angle if the vertex angle is larger or -2*PI + angle if the vertex angle is smaller
        #     angle_diffs = [min(abs(vertex_angles[gap_edge_vertices[j]]-midpoint_angle), abs(vertex_angles[gap_edge_vertices[j]] - ((-1. if vertex_angles[gap_edge_vertices[j]] < midpoint_angle else 1.) * 2*np.pi + midpoint_angle))) for j in range(len(gap_edge_vertices))]
        #     start_vertex = gap_edge_vertices[np.argmin(angle_diffs)]
        #     new_faces.append([start_vertex, edge[0], edge[1]])
        # TODO: fix issue where triangle is added with two of the same vertices
        print(new_faces)

        # remove the intersected face and faces that border the non intersecting vertex
        faces = np.vstack([faces[:intersected_face,:], faces[intersected_face+1:,:]])
        faces = faces[np.logical_not(np.any(faces == inf_vert_index, axis=1))]
        # add new faces
        faces = np.vstack([faces, np.array(new_faces)])

        # remove the non intersecting vertex and update faces
        vertices = np.vstack([vertices[:inf_vert_index, :], vertices[inf_vert_index+1:,:]])
        faces[faces > inf_vert_index] -= 1

        return vertices, faces, None
        



def recompute_mesh_connectivity(vertices, faces, inf_vert_index, inf_vert_direction, non_intersecting_vertices):
    '''
    This function recomputes the connectivity of the mesh when one of the sampled probes has infinite depth (this indicates a hole in the mesh that needs to be reconnected)
    vertices                  - the vertices of the mesh
    faces                     - the faces of the mesh
    inf_vert_index            - the vertex that had the infinite depth
    inf_vert_direction        - the sampling direction for the infinite depth
    non_intersecting_vertices - indices into the vertices array specifying vertices that didn't have intersections

    Returns the updated vertices and faces, the updated stack of non-intersecting vertices, and the index of an additional removal from the stack/list if one occurred
    '''
    curr_vertex_index = non_intersecting_vertices[-1]
    vertices = np.array(vertices)
    faces = np.array(faces)

    # TODO: pass this value in so that it isn't another O(n^2) computation
    near_face_threshold = rasterization.max_edge(vertices, faces)
    # use a small offset when finding the next face intersection so we don't find one of the faces neighboring the starting vertex
    offset = 0.00001
    rot_verts = rasterization.rotate_mesh(vertices, vertices[inf_vert_index]+inf_vert_direction*offset, vertices[inf_vert_index]+inf_vert_direction)

    _, _, intersected_face = rasterization.ray_occ_depth_visual(faces, rot_verts, ray_start_depth=np.linalg.norm(inf_vert_direction)-offset, near_face_threshold=near_face_threshold)
    # TODO: double check to make sure there is an intersection
    if len(intersected_face) > 0:
        intersected_face = int(intersected_face[0])
    else:
        print("ISSUE: SELF INTERSECTING MESH FACES --> RAY DOES NOT SELF INTERSECT")
        print(vertices[inf_vert_index])
        non_intersecting_vertices.pop()
        return vertices, faces,non_intersecting_vertices, None

    vertices, faces, second_removed_vertex = connect_holes(vertices, faces, inf_vert_index, inf_vert_direction, intersected_face)

    # find the index of second_removed_vertex if it exists in the stack
    additional_removal = None
    if second_removed_vertex is not None:
        for i, vi in enumerate(non_intersecting_vertices):
            if vi == second_removed_vertex:
                additional_removal = i
    
    # remove the necessary vertices from non_intersecting_vertices
    non_intersecting_vertices = [x for x in non_intersecting_vertices[:-1] if x != second_removed_vertex]

    # adjust the vertices indices of the stack
    non_intersecting_vertices = [x - (x > curr_vertex_index) - (x > second_removed_vertex if second_removed_vertex is not None else 0) for x in non_intersecting_vertices]

    return vertices, faces, non_intersecting_vertices, additional_removal

def recompute_mesh_connectivity_new(vertices, faces, vert_index, old_vertex, sampled_depth, vert_direction, vert_stack):
    '''
    This function recomputes the connectivity of the mesh when one of the sampled probes has infinite depth (this indicates a hole in the mesh that needs to be reconnected)
    vertices                  - the vertices of the mesh
    faces                     - the faces of the mesh
    vert_index                - the vertex that is being checked for self intersection
    old_vertex                - the previous position of the vertex in question
    sampled_depth             - The depth from the last round of sampling for the vertex in question
    vert_direction            - the sampling direction for the vertex in question
    vert_stack                - indices into the vertices array specifying vertices that still need to be checked for self intersection

    Returns the updated vertices and faces, the updated stack of non-intersecting vertices, and the index of an additional removal from the stack/list if one occurred
    '''
    # TODO: change to edge flips so that new vertices are not added (this could be an additional source of self intersection)
    vertices = np.array(vertices)
    faces = np.array(faces)

    # TODO: pass this value in so that it isn't another O(n^2) computation
    near_face_threshold = rasterization.max_edge(vertices, faces)
    # use a small offset when finding the next face intersection so we don't find one of the faces neighboring the starting vertex
    offset = 0.000001
    rot_verts = rasterization.rotate_mesh(vertices, old_vertex, old_vertex+vert_direction)

    _, depth, intersected_face = rasterization.ray_occ_depth_visual(faces, rot_verts, ray_start_depth=np.linalg.norm(vert_direction)-offset, near_face_threshold=near_face_threshold)

    # TODO: double check to make sure there is an intersection
    if len(intersected_face) > 0:
        # check to see whether the intersected face is closer than the sampled depth
        if depth+offset*2. < sampled_depth:
            intersected_face = int(intersected_face[0])
            # TODO: ensure that the intersected face does not contain vert_index

            # print("SELF INTERSECT FACE")
            # print(sampled_depth)
            # print(depth+offset*2.0)
            # lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)
            # visualizer = visualization.RayVisualizer(vertices, lines)
            # visualizer.add_point(old_vertex, [1.0,0.0,0.0])
            # visualizer.add_point(vertices[vert_index], [0.0,1.0,0.0])
            # visualizer.add_point(old_vertex + (depth+offset*2.)*vert_direction/np.linalg.norm(vert_direction), [0.0,0.0,1.0])
            # visualizer.display()
        else:
            vert_stack.pop()
            return vertices, faces, vert_stack, None
    else:
        print("ISSUE: SELF INTERSECTING MESH FACES --> RAY DOES NOT SELF INTERSECT")
        print(vertices[vert_index])
        vert_stack.pop()
        return vertices, faces,vert_stack, None

    # if we get to this point, there is a self intersection on the mesh, that we will repair by punching a hole through the mesh and fixing the connectivity
    # This involves deleting either 1 or 2 vertices and multiple faces, and adding new faces
    vertices, faces, second_removed_vertex = connect_holes(vertices, faces, vert_index, vert_direction, intersected_face)

    # find the index of second_removed_vertex if it exists in the stack
    additional_removal = None
    if second_removed_vertex is not None:
        for i, vi in enumerate(vert_stack):
            if vi == second_removed_vertex:
                additional_removal = i
    
    # remove the necessary vertices from vert_stack
    vert_stack = [x for x in vert_stack[:-1] if x != second_removed_vertex]

    # adjust the vertices indices of the stack
    vert_stack = [x - (x > vert_index) - (x > second_removed_vertex if second_removed_vertex is not None else 0) for x in vert_stack]

    return vertices, faces, vert_stack, additional_removal


def sample_next_vertices(model, vertices, faces, probes, directions, radius, delta, first_sampling=False):
    '''
    Update mesh vertices and faces by querying an ODF
    model          - An ODF
    vertices       - the current vertices of the mesh
    faces          - triples of vertex indices representing mesh faces
    probes         - a suffix of the vertices defining which vertices still need to be sampled
    directions     - defines a sampling direction (negative of normal) for each vertex
    radius         - the radius of the sampling sphere
    delta          - how far back to move the probe points before sampling
    first_sampling - True if this is the first sampling, in which case the probes won't be moved back in the negative direction
    '''
    vertices = np.array(vertices)
    faces = np.array(faces)
    probes = np.array(probes)
    directions = np.array(directions)
    # only take the directions that correspond to a probe
    directions = directions[-probes.shape[0]:]
    # probes offset allows us to convert between probe indices and vertices/directions indices (some prefix of the vertices are not probes)
    probes_offset = vertices.shape[0] - probes.shape[0]
    # don't shift the probes back if they are already on the sphere surface
    # TODO: there could still be errors here if the object surface is close to the sampling sphere (doesn't matter if delta is less than the sampling sphere buffer)
    if not first_sampling:
        print("NOT FIRST SAMPLING")
        print(np.max(np.linalg.norm(probes, axis=1)))
        print(probes)
        probes = probes - (delta * directions)
        print(np.sum(np.isnan(probes)))
        print(np.sum(np.isnan(directions)))
        print(np.max(np.linalg.norm(probes, axis=1)))
        print(probes)
    # lines = np.concatenate([faces[:,:2], faces[:,1:], faces[:,[0,2]]], axis=0)
    with torch.no_grad():
        # pass in surface point, direction
        print("Starting")
        _, depths, n_ints = model.query_rays(torch.tensor(probes, dtype=torch.float32), torch.tensor(directions, dtype=torch.float32))
        print("Done")
    n_ints = n_ints.cpu()
    model_depths = depths.cpu()
    model_depths = torch.min(model_depths, dim=1)[0]
    model_depths = model_depths.numpy()
    # new_vertices = [vertices[i] for i in range(probes_offset)] + [probes[i-probes_offset] + directions[i]*model_depths[i-probes_offset] if model_depths[i-probes_offset] < np.inf else utils.get_sphere_intersections(vertices[i], directions[i], radius)[1] for i in range(probes_offset, vertices.shape[0])]
    new_vertices = [vertices[i] for i in range(probes_offset)] + [probes[i-probes_offset] + directions[i-probes_offset]*model_depths[i-probes_offset] if model_depths[i-probes_offset] < np.inf else vertices[i] for i in range(probes_offset, vertices.shape[0])]

    old_vertices_stack = [vertices[i] for i in range(probes_offset, vertices.shape[0])]

    vertices = np.array(new_vertices)

    check_intersection_stack = [i for i in range(probes_offset, vertices.shape[0])]
    depth_stack = [model_depths[i]-delta for i in range(vertices.shape[0]-probes_offset)]
    directions_stack = [directions[i] for i in range(vertices.shape[0]-probes_offset)]

    print(f"OLD VERTICES LEN: {len(old_vertices_stack)}")
    print(f"DIRECTIONS LEN: {len(directions)}")
    print(f"MODEL DEPTHS LEN: {len(model_depths)}")
    print(f"i range: {probes_offset} --> {vertices.shape} ")
    opposing_points = [utils.get_sphere_intersections(old_vertices_stack[i], directions[i], 1.0)[1]  if model_depths[i]==np.inf else None for i in range(vertices.shape[0]-probes_offset)]
    has_inf_depths = np.any([True if x is not None else False for x in opposing_points])
    opposing_vert_depth_stack = [np.linalg.norm(opposing_points[i-probes_offset]-vertices[i]) if opposing_points[i-probes_offset] is not None else None for i in range(probes_offset, vertices.shape[0])]
    opposing_directions = [-1*directions[i] for i in range(vertices.shape[0]-probes_offset)]

    if has_inf_depths:
        with torch.no_grad():
            _, opposing_model_depths, n_ints = model.query_rays(torch.tensor([x for x in opposing_points if x is not None], dtype=torch.float32), torch.tensor([x for x in opposing_directions if x is not None], dtype=torch.float32))
        n_ints = n_ints.cpu()
        opposing_model_depths = opposing_model_depths.cpu()
        opposing_model_depths = torch.min(opposing_model_depths, dim=1)[0] if opposing_model_depths.shape[0] > 1 else opposing_model_depths
        opposing_model_depths = list(opposing_model_depths.numpy())
        opposing_model_depths.reverse()
        # put the model depths back at their correct indices and fill the rest with Nones
        opposing_model_depth_stack = [x if x is None else opposing_model_depths.pop() for x in opposing_points]
    else:
        opposing_model_depth_stack = [None for _ in opposing_points]

    while len(check_intersection_stack) > 0:
        # check for the case where we have infinite depth that wasn't supported by the second network query
        print("vals")
        print(f"{depth_stack[-1]}, {opposing_model_depth_stack[-1]}, {opposing_vert_depth_stack[-1]}")
        if depth_stack[-1] == np.inf and not (opposing_model_depth_stack[-1] == np.inf or opposing_model_depth_stack[-1] > opposing_vert_depth_stack[-1]):
            check_intersection_stack.pop()
            old_vertices_stack.pop()
            depth_stack.pop()
            directions_stack.pop()
            opposing_vert_depth_stack.pop()
            opposing_model_depth_stack.pop()
        else:
            vertices, faces, check_intersection_stack, additional_removal = recompute_mesh_connectivity_new(vertices, faces, check_intersection_stack[-1], old_vertices_stack[-1], depth_stack[-1], directions_stack[-1], check_intersection_stack)
            old_vertices_stack.pop()
            depth_stack.pop()
            directions_stack.pop()
            opposing_vert_depth_stack.pop()
            opposing_model_depth_stack.pop()
            if additional_removal is not None:
                old_vertices_stack = old_vertices_stack[:additional_removal] + old_vertices_stack[additional_removal+1:]
                depth_stack = depth_stack[:additional_removal] + depth_stack[additional_removal+1:]
                directions_stack = directions_stack[:additional_removal] + directions_stack[additional_removal+1:]
                opposing_vert_depth_stack = opposing_vert_depth_stack[:additional_removal] + opposing_vert_depth_stack[additional_removal+1:]
                opposing_model_depth_stack = opposing_model_depth_stack[:additional_removal] + opposing_model_depth_stack[additional_removal+1:]
        # import visualization
        # import open3d as o3d
        # o3d.visualization.draw_geometries([visualization.make_mesh(np.array(vertices), faces)])

    # # points to be recalculated
    # non_intersecting_vertices = [i for i in range(probes_offset, len(vertices)) if  model_depths[i-probes_offset] == np.inf]
    # print(f"{len(non_intersecting_vertices)}/{len(new_vertices)}")

    # opposing_points = []
    # opposing_directions = []
    # opposing_vertex_depths = []
    # for i in non_intersecting_vertices:
    #     opposing_point = utils.get_sphere_intersections(vertices[i], directions[i], radius)[1]
    #     opposing_points.append(opposing_point)
    #     opposing_vertex_depths.append(np.linalg.norm(opposing_point - vertices[i]))
    #     opposing_directions.append(-1 * directions[i])


    # if len(opposing_points) > 0:
    #     with torch.no_grad():
    #         _, depths, n_ints = model.query_rays(torch.tensor(opposing_points, dtype=torch.float32), torch.tensor(opposing_directions, dtype=torch.float32))

    #     n_ints = n_ints.cpu()
    #     opposing_model_depths = depths.cpu()
    #     opposing_model_depths = torch.min(opposing_model_depths, dim=1)[0] if opposing_model_depths.shape[0] > 1 else opposing_model_depths
    #     opposing_model_depths = list(opposing_model_depths.numpy())

    #     #Using lists as stacks. Last element of list is top of stack
    #     while len(non_intersecting_vertices) > 0:
    #         # redo mesh connectivity if there is truly a hole in the mesh
    #         if opposing_model_depths[-1] == np.inf or opposing_model_depths[-1] > opposing_vertex_depths[-1]:
    #             vertices, faces, non_intersecting_vertices, additional_removal  = recompute_mesh_connectivity(vertices, faces, non_intersecting_vertices[-1], directions[non_intersecting_vertices[-1]], non_intersecting_vertices)
    #             opposing_vertex_depths.pop()
    #             opposing_model_depths.pop()
    #             if additional_removal is not None:
    #                 opposing_vertex_depths = opposing_vertex_depths[:additional_removal] + opposing_vertex_depths[additional_removal+1:]
    #                 opposing_model_depths = opposing_model_depths[:additional_removal] + opposing_model_depths[additional_removal+1:]
    #         else: 
    #             non_intersecting_vertices.pop()
    #             opposing_vertex_depths.pop()
    #             opposing_model_depths.pop()
    #         import visualization
    #         import open3d as o3d
    #         o3d.visualization.draw_geometries([visualization.make_mesh(np.array(vertices), faces)])

    #     # TODO: recompute mesh edges for the self intersecting vertices

    return vertices, faces

def show_subdivisions_and_probes(vertices, probes, inf_mask, directions, faces, delta, cluster=[], c_verts=[], c_faces=[], show_mesh_faces=True):
    '''
    For visualization purposes.
    Shows which edges have been subdivided and where the probe locations are.
    '''
    mesh_file = "F:\\ivl-data\\sample_data\\stanford_bunny_watertight.obj"
    mesh = trimesh.load(mesh_file)
    gt_faces = mesh.faces
    gt_verts = mesh.vertices
    gt_verts = utils.mesh_normalize(gt_verts)
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
    opposing_points = [utils.get_sphere_intersections(probes[i], directions[i], probe_radius)[1] for i in range(len(probes)) if model_depths[i]==np.inf]
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
    
    

def adjacent_rays_plane(vert1, direction1, vert2, direction2):
    '''
    Given two vertices and their probe_directions, returns the adjacent triangle planes that join the two sampling rays
    '''

    ray1_sphere_intersect = utils.get_sphere_intersections(vert1, direction1, 1.0)[1]
    ray2_sphere_intersect = utils.get_sphere_intersections(vert2, direction2, 1.0)[1]

    if np.linalg.norm(ray1_sphere_intersect-vert2) < np.linalg.norm(ray2_sphere_intersect-vert1):
        return np.array([vert1, vert2, ray1_sphere_intersect, ray2_sphere_intersect]), np.array([[0,2,1], [1,2,3]])
    else:
        return np.array([vert1, vert2, ray1_sphere_intersect, ray2_sphere_intersect]), np.array([[0,2,3], [1,0,3]])

def single_vert_intersection(vertices, faces, vert, direction, delta, near_face_threshold):
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

def vert_cluster_intersections(vertices, faces, probe_cluster, directions, delta):
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
        intersected_faces.append(single_vert_intersection(vertices, faces, vertices[probe], directions[i], delta, near_face_threshold)[0])
    return list(set(intersected_faces))

def vert_neighbors(faces, face_cluster, inf_vertices):
    '''
    Given a set of faces, returns the vertices neighboring the faces that have infinite depth
    '''
    face_cluster = np.array(face_cluster)
    faces = np.array(faces)
    neighboring_vert_indices = list(set(list(faces[face_cluster].flatten())))
    inf_vertices = set(inf_vertices)
    # TODO: don't just check if it's in inf_vertices, check if there is a self intersection before the predicted depth.
    neighbors = [x for x in neighboring_vert_indices if x in inf_vertices]
    return neighbors

def face_neighbors(faces, vert_cluster):
    '''
    Given a set of vertices, returns the faces neighboring the vertices
    '''
    faces = np.array(faces)
    face_masks = []
    for v in vert_cluster:
        face_masks.append(np.any(faces==v, axis=1)[:, np.newaxis])
    final_mask = np.any(np.hstack(face_masks), axis=1)

    face_indices = [i for i in range(faces.shape[0]) if final_mask[i]]
    return face_indices



def cluster_expansion(vertices, faces, cluster, inf_verts, directions, delta):
    '''
    Takes a cluster of infinite depth vertices.
    Returns two clusters, each of both faces and vertices.
    Cluster A contains the vertices and neighboring faces around the input cluster.
    Cluster B is the faces intersected by the first cluster and their neighboring vertices.
    Directions should be the directions for ALL vertices!
    NOTE: faces and vertices are both returned as indices into their respective arrays
    '''
    # The two clusters (A&B)
    A_vertices = [x for x in cluster]
    A_faces = []
    B_vertices = []
    B_faces = []

    #items added to the clusters in the most recent iteration
    av_new = [x for x in A_vertices]
    af_new = []
    bv_new = []
    bf_new = []

    #iterate while new things are still being added to the clusters
    while((len(av_new) + len(af_new) + len(bv_new) + len(bf_new)) > 0):
        af_next = vert_cluster_intersections(vertices, faces, bv_new, [directions[i] for i in bv_new], delta) if len(bv_new) > 0 else []
        af_next += face_neighbors(faces, av_new) if len(av_new) > 0 else []
        af_next = list(set(af_next))

        bf_next = vert_cluster_intersections(vertices, faces, av_new, [directions[i] for i in av_new], delta) if len(av_new) > 0 else []
        bf_next += face_neighbors(faces, bv_new) if len(bv_new) > 0 else []
        bf_next = list(set(bf_next))
        
        av_next =  vert_neighbors(faces, af_new, inf_verts) if len(af_new) > 0 else []
        bv_next =  vert_neighbors(faces, bf_new, inf_verts) if len(bf_new) > 0 else []        

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

def cluster_inf_probes_old(inf_indices, faces):
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
            neighbors = []
            for x in new_indices:
                neighbors += list(faces[np.any(faces==x, axis=1)].flatten())
            neighbors = set(neighbors)
            new_indices = []
            # TODO: remove duplicate work here
            for x in inf_indices:
                if x in neighbors:
                    new_indices.append(x)
                    new_cluster.append(x)
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
    clusters = cluster_inf_probes(inf_indices, faces)
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
        directions = -1. * utils.get_vertex_normals(np.array(vertices), np.array(faces))
        
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
        verts = utils.mesh_normalize(verts)
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
        verts = utils.mesh_normalize(verts)
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