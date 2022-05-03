from tkinter import N
from turtle import forward
import numpy as np
import random

VISUAL = True

if VISUAL:
    import open3d as o3d

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

corner_vertices = {
    0: (0.,0.,0.),
    1: (1.,0.,0.),
    2: (1.,1.,0.),
    3: (0.,1.,0.),
    4: (0.,0.,1.),
    5: (1.,0.,1.),
    6: (1.,1.,1.),
    7: (0.,1.,1.)
}

edge_vertices = {
    0: (0.5,0.,0.),
    1: (1.,0.5,0.),
    2: (0.5,1.,0.),
    3: (0.,0.5,0.),
    4: (0.5,0.,1.),
    5: (1.,0.5,1.),
    6: (0.5,1.,1.),
    7: (0.,0.5,1.),
    8: (0.,0.,0.5),
    9: (1.,0.,0.5),
    10: (1.,1.,0.5),
    11: (0.,1.,0.5)
}

cube_edges = [(0,1), (1,2), (2,3), (3,0), (0,4), (1,5), (2,6), (3,7), (4,5), (5,6), (6,7), (7,4)]

face_pairs = [
    set([4,8]), #Face 1
    set([4,9]),
    set([4,0]),
    set([8,9]),
    set([8,0]),
    set([0,9]),
    set([0,3]), #Face 2
    set([0,2]),
    set([0,1]),
    set([3,1]),
    set([2,1]),
    set([3,2]),
    set([1,9]), #Face 3
    set([1,5]),
    set([1,10]),
    set([9,10]),
    set([9,5]),
    set([5,10]),
    set([3,8]), #Face 4
    set([3,7]),
    set([3,11]),
    set([8,7]),
    set([8,11]),
    set([7,11]),
    set([2,11]), #Face 5
    set([2,6]),
    set([2,10]),
    set([11,6]),
    set([11,10]),
    set([6,10]),
    set([4,7]), #Face 6
    set([4,6]),
    set([4,5]),
    set([7,6]),
    set([7,5]),
    set([6,5])
]


z_rotation_mapping = {
        0: 8,
        8: 4,
        4: 9,
        9: 0,
        3: 7,
        7: 5,
        5: 1,
        1: 3,
        2: 11,
        11: 6,
        6: 10,
        10: 2
    }

def rotate_z(indices):
    '''
    Rotates the cube around the z-axis
    Given occupied vertex indices, returns the new occupied indices after the rotation
    Refer to the middle diagram of Figure 6 in the Marching Cubes Lewiner Paper for vertex indexing 
    '''
    return [z_rotation_mapping[i] for i in indices]


y_rotation_mapping = {
        0: 3,
        3: 2,
        2: 1,
        1: 0,
        8: 11,
        11: 10,
        10: 9,
        9: 8,
        4: 7,
        7: 6,
        6: 5,
        5: 4
    }

def rotate_y(indices):
    '''
    Rotates the cube around the y-axis
    Given occupied vertex indices, returns the new occupied indices after the rotation
    Refer to the middle diagram of Figure 6 in the Marching Cubes Lewiner Paper for vertex indexing 
    '''
    return [y_rotation_mapping[i] for i in indices]

#Defines the rotations needed to get from a given vertex to the root vertex (vertex 0)
#Returns a tuple of (# z rotations, # y rotations)
to_root={
    0: (0,0),
    9: (1,0),
    4: (2,0),
    8: (3,0),
    1: (0,1),
    5: (1,1),
    7: (2,1),
    3: (0,3),
    2: (0,2),
    10: (1,2),
    6: (2,2),
    11: (3,2)
}

#Flips the cube to the second orientation that maintains the same root vertex
flip_mapping ={
    0: 0,
    9: 3,
    4: 2,
    8: 1,
    1: 8,
    5: 11,
    7: 10,
    3: 9,
    2: 4,
    11: 5,
    6: 6,
    10: 7,
}

def flip_cube(indices):
    return [flip_mapping[i] for i in indices]

def on_same_face(indices):
    '''
    Returns true if the all of the provided vertices are in the same axis-aligned plane
    '''
    if len(indices) > 4:
        return False
    if len(indices) <= 1:
        return True
    
    x_coords = [edge_vertices[i][0] for i in indices]
    y_coords = [edge_vertices[i][1] for i in indices]
    z_coords = [edge_vertices[i][2] for i in indices]

    if max(x_coords) == (sum(x_coords)/len(x_coords)) and x_coords[0] != 0.5:
        return True
    if max(y_coords) == (sum(y_coords)/len(y_coords)) and y_coords[0] != 0.5:
        return True
    if max(z_coords) == (sum(z_coords)/len(z_coords)) and z_coords[0] != 0.5:
        return True
    return False

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def triangulate_cycle(cycle):
    '''
    Returns triangle faces for the provided cycle
    '''
    assert(len(cycle) > 2)
    # minimal case is only a single triangle
    if len(cycle) == 3:
        return [cycle]
    else:
        # at each step add the least "slivery" triangle
        best_index = None
        max_angle = None
        for i in range(len(cycle)):
            v1 = np.array(edge_vertices[cycle[i-1]]) - np.array(edge_vertices[cycle[i]])
            v2 = np.array(edge_vertices[cycle[i-2]]) - np.array(edge_vertices[cycle[i]])
            curr_angle = angle_between(v1,v2)
            if max_angle is None or curr_angle > max_angle:
                max_angle = curr_angle
                best_index = i
        # recursively triangulate the rest of the cycle
        removed_vertex = cycle[best_index-1]
        new_cycle = [i for i in cycle if i != removed_vertex]
        return [[cycle[best_index], cycle[best_index-1], cycle[best_index-2]]] + triangulate_cycle(new_cycle)


def zero_cycle_meshing(chains):
    '''
    Returns the faces of the mesh for cube cases that don't have any complete cycles
    '''
    # Subcase 1 - no vertices
    if len(chains) == 0:
        return []
    
    if len(chains) == 1:
        # Subcase 2 - single chain on one face
        if on_same_face(chains[0]):
            return []
        # Subcase 3 - single chain on multiple faces
        else:
            return triangulate_cycle(chains[0])

    if len(chains) == 2:
        # Subcase 4 - two individual points
        if len(chains[0]) == 1 and len(chains[1]) == 1:
            return []
        #Subcase 5 - two chains where at least one isn't a single point
        else:
            # figure out which vertices to connect to form a cycle (use smallest average edge length)
            dist1 = np.linalg.norm(np.array(edge_vertices[chains[0][0]])-np.array(edge_vertices[chains[1][0]]))
            dist1 += np.linalg.norm(np.array(edge_vertices[chains[0][-1]])-np.array(edge_vertices[chains[1][-1]]))
            dist2 = np.linalg.norm(np.array(edge_vertices[chains[0][0]])-np.array(edge_vertices[chains[1][-1]]))
            dist2 += np.linalg.norm(np.array(edge_vertices[chains[0][-1]])-np.array(edge_vertices[chains[1][0]]))
            if dist1 < dist2:
                chains[1].reverse()
            return triangulate_cycle(chains[0]+chains[1])
    
    # Subcase 6 - 3 individual points
    if len(chains) == 3:
        return []

class Cube():

    def __init__(self, case_number=None, assignments=None):
        assert(case_number is not None or assignments is not None)
        if case_number is not None:
            self.case = case_number
            self.edge_vertices = []
            for i in range(12):
                self.edge_vertices.append((((self.case%(2**(12-i))) // (2**(11-i)))-1)*-1)
            self.indices = [i for i,e in enumerate(self.edge_vertices) if e]
        elif assignments is not None:
            self.indices = assignments
            self.edge_vertices = [1 if x in assignments else 0 for x in range(12)]
            self.case = sum([self.edge_vertices[i]*(2**i) for i in range(12)])
        self.n_vertices = len(self.indices)
        self.indices.sort()

        # make edge dictionary
        self.adjacencies = self.edge_connections()
        self.edges = {} 
        for n in self.indices:
            self.edges[n] = []
        for pair in self.adjacencies:
            self.edges[pair[0]].append(pair[1])
            self.edges[pair[1]].append(pair[0])

        # find the faces of the mesh
        self.faces = self.get_faces()

    def zero_align(self):
        '''
        Returns a new cube instance that is obtained by rotating this cube instance so that the lowest index vertex is in the zero position.
        '''
        if self.n_vertices == 0:
            return Cube(assignments=[])
        inds = self.indices.copy()
        z_rot, y_rot = to_root[inds[0]]
        for _ in range(z_rot):
            inds = rotate_z(inds)
        for _ in range(y_rot):
            inds = rotate_y(inds)
        inds.sort()
        new_cube = Cube(assignments=inds)
        return new_cube

    def show(self, show_faces=False):
        if VISUAL:
            geometries = []
            corners = np.array([corner_vertices[i] for i in range(8)])
            edges = np.array(cube_edges)
            np_edge_verts = np.array([edge_vertices[i] for i in range(12)])
            geometries.append(make_line_set(corners, edges))
            if len(self.indices) > 0:
                geometries.append(make_point_cloud(np.array([edge_vertices[i] for i in self.indices]), (np.array([[1.,0.,0.],]*len(self.indices)))))
            if len(self.adjacencies) > 0:
                geometries.append(make_line_set(np_edge_verts, np.array(self.adjacencies), np.array([[0.,0.,1.],]*len(self.adjacencies))))
            if show_faces and len(self.faces) > 0:
                geometries.append(make_mesh(np_edge_verts, self.faces))
            o3d.visualization.draw_geometries(geometries)
        else:
            print("To show a cube case, please enable the VISUAL flag in jumping_cubes.py")

    def get_faces(self):
        n_cycles = len(self.get_maximal_cycles())
        # zero cycle case
        if n_cycles == 0:
            return zero_cycle_meshing(self.get_linear_chains())

        # one cycle case
        if n_cycles == 1:
            return []
        # two cycle case
        if n_cycles == 2:
            return []

    def edge_connections(self):
        surface_edges = []
        for i in range(len(self.indices)):
            for j in range(i+1, len(self.indices)):
                if set([self.indices[i],self.indices[j]]) in face_pairs:
                    surface_edges.append([self.indices[i],self.indices[j]])
        return surface_edges

    def dfs_cycles(self):
        '''
        Finds all of the cycles in the vertex graph using DFS
        '''
        visited = set([])
        cycles = []
        cycle_sets = []
        nodes = set(self.indices)
        
        def dfs_cycle_recur(path):
            # record visit to node
            visited.add(path[-1])
            # look for cycle
            if path[-1] in path[:-1]:
                cycle = path[path.index(path[-1]):-1]
                if len(cycle) > 2 and set(cycle) not in cycle_sets:
                    cycle_sets.append(set(cycle))
                    cycles.append(cycle)
            # if no cycle, continue DFS
            else:
                for neighbor in self.edges[path[-1]]:
                    dfs_cycle_recur(path+[neighbor])

        # run dfs on all connected components of the graph
        while len(nodes) != 0:
            root = nodes.pop()
            dfs_cycle_recur([root])
            nodes -= visited

        return cycles

    def get_maximal_cycles(self):
        '''
        Returns all cycles that do not share any edges with a larger cycle
        '''
        cycles = self.dfs_cycles()
        cycles.sort(key=len)
        cycles.reverse()

        used_edges = set()
        maximal_cycles = []

        for cycle in cycles:
            is_maximal = True
            for i in range(len(cycle)):
                if (cycle[i-1], cycle[i]) in used_edges:
                    is_maximal = False
                    break
            if not is_maximal:
                break

            # if it is a maximal cycle
            maximal_cycles.append(cycle)
            for i in range(len(cycle)):
                used_edges.add((cycle[i-1], cycle[i]))
                used_edges.add((cycle[i], cycle[i-1]))
        return maximal_cycles

    def get_linear_chains(self):
        '''
        Returns all of the linear edge chains for this cube
        Will error if the current cube has a cycle
        '''

        def chain_traversal(curr_chain, curr_link):
            curr_chain.append(curr_link)
            next_links = [link for link in self.edges[curr_link] if link not in curr_chain]
            assert(len(next_links) < 2)
            # check for end of chain
            if len(next_links) == 0:
                return curr_chain
            else:
                return chain_traversal(curr_chain, next_links[0])


        nodes = set(self.indices)
        chains = []

        while len(nodes) > 0:
            start = nodes.pop()
            curr_chain = [start]
            assert(len(self.edges[start]) <= 2)
            forward_link, backward_link = None, None
            if len(self.edges[start]) == 1:
                forward_link = self.edges[start][0]
            if len(self.edges[start]) == 2:
                forward_link = self.edges[start][0]
                backward_link = self.edges[start][1]
            
            
            #look forwards
            if forward_link is not None:
                curr_chain = chain_traversal(curr_chain, forward_link)
            if backward_link is not None:
                curr_chain.reverse()
                curr_chain = chain_traversal(curr_chain, backward_link)
            chains.append(curr_chain)
            nodes -= set(curr_chain)
        return chains


    def equivalent(self, other_cube):
        '''
        Returns true if these two cubes are rotations of the same basic case, false.
        Only works if the current cube is zero-aligned
        '''
        if not self.n_vertices == other_cube.n_vertices:
            return False

        base = self.indices.copy()
        base.sort()
        for vert in other_cube.indices:
            inds = other_cube.indices.copy()
            z_rot, y_rot = to_root[vert]
            for _ in range(z_rot):
                inds = rotate_z(inds)
            for _ in range(y_rot):
                inds = rotate_y(inds)
            inds.sort()
            if inds == base:
                return True
            else:
                inds = flip_cube(inds)
                inds.sort()
                if inds == base:
                    return True
        return False
        
marching_cubes_cases = [
    [], #Case 0
    [0,3,8], #Case 1
    [1,3,8,9], #Case 2
    [0,3,4,5,8,9], #Case 3
    [0,3,5,6,8,10], #Case 4
    [0,3,9,10,11], #Case 5
    [1,3,5,6,8,9,10], #Case 6
    [0,1,4,5,6,7,8,9,10], #Case 7
    [8,9,10,11], #Case 8
    [0,1,6,7,8,10], #Case 9
    [0,1,2,3,4,5,6,7], #Case 10
    [0,1,5,6,8,11], #Case 11
    [0,3,4,7,8,9,10,11], #Case 12
    [0,1,2,3,4,5,6,7,8,9,10,11], #Case 13
    [0,3,6,7,9,10], #Case 14
]

mc_cubes = [Cube(assignments=indices).zero_align() for indices in marching_cubes_cases]

def check_marching_cubes(indices):
    '''
    Checks if the provided indices are a marching cubes case, and returns the case number if they are
    '''
    curr_cube = Cube(assignments=indices)
    for i, mc_case in enumerate(mc_cubes):
        if mc_case.equivalent(curr_cube):
            return i
    return None

def discover_base_cases():
    bases = {i: [] for i in range(13)}
    cycle_distribution = {}
    for case in range(4096):
        c = Cube(case_number=case)
        base_class = c.n_vertices

        # check whether this case already exists
        case_exists = False
        for base in bases[base_class]:
            if base.equivalent(c):
                case_exists = True
                break
        # add the case if it doesn't exist yet
        if not case_exists:
            bases[base_class].append(c)
            n_cycles = len(c.get_maximal_cycles())
            if n_cycles == 1:
                # print(f"Found {len(c.get_linear_chains())} chains" + str(tuple([len(chain) for chain in c.get_linear_chains()])))
                print(f"Showing case {case}")
                c.show(show_faces=True)
            if not n_cycles in cycle_distribution:
                cycle_distribution[n_cycles] = 1
            else:
                cycle_distribution[n_cycles] += 1

    print(cycle_distribution)

    # count all of the cases and see which ones match the original marching cubes
    total_cases = 0
    for base_class in bases:
        # print(f"{base_class} vertices : {len(bases[base_class])} unique cases")
        total_cases += len(bases[base_class])
        # for case in bases[base_class]:
        #     mc_case = check_marching_cubes(case.indices)
        #     if mc_case is not None:
        #         print(f"Marching Cubes Case {mc_case}")
    print(f"Total cases: {total_cases}")

if __name__ == "__main__":
    # Cube(0)
    # Cube(4095)
    # Cube(1432)
    # x = Cube(assignments=[0,9,5,6,11,8]) #Cycle
    # x.show()
    # Cube(assignments=[0,3,11,6]) #Partial Cycle
    # Cube(assignments=[8,3,0,9,10]) #Cycle + Partial Cycle

    # # Can Ignore
    # Cube(case_number=7) #Cycle in Plane
    # Cube(assignments=[9,10]) #Line
    # Cube(assignments=[9]) #Point

    # #Combined
    # Cube(assignments=[8,3,0,5,10]) #Cycle + Line
    discover_base_cases()