'''
Script to visualize ODF meshing algorithm
This provides a 2D visualization using a calculated distance function
'''

import numpy as np
import matplotlib.pyplot as plt
import math

from numpy.core.fromnumeric import shape

shape_verts = np.array([
    [-0.5, 0.1],
    [-0.5, 0.2],
    [-0.2, 0.5],
    [-0.2, 0.7],
    [0.0, 0.8],
    [0.6, 0.7],
    [0.7, 0.6],
    [0.7, 0.3],
    [0.6, 0.15],
    [0.4, 0.1],
    [0.25, 0.2],
    [0.25, 0.3],
    [0.4, 0.3],
    [0.4, 0.5],
    [0.1, 0.5],
    [0.1, 0.0],
    [0.3, -0.2],
    [0.45, -0.55],
    [0.25, -0.5],
    [-0.25, -0.5],
    [-0.3, -0.4],
    [-0.35, -0.2],
    [-0.25, -0.2],
    [-0.25, -0.4],
    [0.1, -0.4],
    [0.1, -0.2],
    [-0.3, 0.1]
])

shape_edges = np.array([[i, i+1] for i in range(len(shape_verts)-1)] + [[len(shape_verts)-1, 0]])

class AlgVisualizer2D():

    def __init__(self, obj_verts, obj_edges):
        super().__init__()
        
        # the object being fitted
        self.obj_verts = np.array(obj_verts)
        self.obj_edges = np.array(obj_edges)
        # the shrinkwrapped polygon being refined to approximate the mesh
        self.wrap_verts = []
        self.wrap_edges = []
        
        self.probe_points = None
        self.probe_segments = None
        self.curr_segments = None


    def draw_obj(self):
        self.ax.scatter([0.], [0.], color="green")
        for edge in self.obj_edges:
            self.ax.plot(self.obj_verts[edge,0], self.obj_verts[edge,1], color="black")

    def draw_wrap(self):
        self.ax.scatter(self.wrap_verts[:,0], self.wrap_verts[:,1], color="red")  
        for edge in self.wrap_edges:
            self.ax.plot(self.wrap_verts[edge,0], self.wrap_verts[edge,1], color="orange")

    def draw_probes(self):
        if self.probe_points is not None:
            self.ax.scatter(self.probe_points[:,0], self.probe_points[:,1], color="limegreen")
        if self.probe_segments is not None:
            for pair in self.probe_segments:
                self.ax.plot([pair[0][0], pair[1][0]], [pair[0][1], pair[1][1]], color="darkgreen")
        if self.curr_segments is not None:
            for pair in self.curr_segments:
                self.ax.plot([pair[0][0], pair[1][0]], [pair[0][1], pair[1][1]], color="cyan")        

    def update_wrap(self, new_wrap_verts, new_wrap_edges):
        self.wrap_verts = np.array(new_wrap_verts)
        self.wrap_edges = np.array(new_wrap_edges)

    def update_probe_points(self, new_probe_points):
        self.probe_points = np.array(new_probe_points)

    def update_probe_segments(self, new_probe_segments):
        if self.probe_segments == None:
            self.probe_segments = new_probe_segments
        else:
            self.probe_segments += new_probe_segments

    def update_curr_segments(self, new_segments):
        self.curr_segments = new_segments

    def clear_probes(self):
        self.probe_points = None
        self.probe_segments = None
        
    def retire_segments(self):
        if self.curr_segments is not None:
            if self.probe_segments is not None:
                self.probe_segments += self.curr_segments
            else:
                self.probe_segments = self.curr_segments
            self.curr_segments = None

    def display(self, probes=True, i=None):
        # plotting setup
        # plotting setup
        self.f, self.ax = plt.subplots(1,1)
        self.f.set_size_inches(8, 8)
        self.ax.set_ylim(-1.1,1.1)
        self.ax.set_xlim(-1.1,1.1)
        if i is not None:
            self.ax.set_title(f"Iterations: {i}")
        self.draw_obj()
        self.draw_wrap()
        if probes:
            self.draw_probes()
        plt.show()




def start_points_2d(n_points=100):
    '''
    Returns circle circumference start points, starting directions, and adjacency list
    '''
    start_points = [[math.cos(i/n_points*2.*math.pi), math.sin(i/n_points*2.*math.pi)] for i in range(n_points)]
    start_directions = [[-math.cos(i/n_points*2.*math.pi), -math.sin(i/n_points*2.*math.pi)] for i in range(n_points)]
    adj_list = [[i, i+1] for i in range(n_points-1)] + [[n_points-1, 0]]
    return start_points, start_directions, adj_list

def rotate_verts_2d(shape_verts, point, dir):
    '''
    Calculates a translation + rotation transformation so that point is at the origin and dir is in the +x direction
    Applies this transformation to shape_verts and returns it
    '''
    if dir[0] == 0:
        if dir[1] > 0.:
            theta = math.pi/2.
        else:
            theta = -math.pi/2.
    else:
        theta = math.atan(dir[1] / dir[0])
    # theta = -theta
    if dir[0] < 0.:
        theta = theta + math.pi
    rot_mat = np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]])
    rot_verts = []
    for v in shape_verts:
        rot_verts.append(np.matmul([v - point], rot_mat)[0])
    return np.array(rot_verts)

def intersection_depth_2d(rot_verts, edges):
    depths = []
    for edge in edges:
        seg1 = rot_verts[edge[0]]
        seg2 = rot_verts[edge[1]]
        if int(seg1[1] > 0) + int(seg2[1] > 0) == 1:
            depth = (abs(seg1[1])/(abs(seg1[1]) + abs(seg2[1]))) * seg2[0] + (abs(seg2[1])/(abs(seg1[1]) + abs(seg2[1]))) * seg1[0]
            if depth >= 0.:
                depths.append(depth)
    if len(depths) == 0:
        return False, None
    else:
        return True, min(depths)

def get_depths(shape_verts, shape_edges, points, directions):
    '''
    Gets the surface depth for different rays. 
    TODO: Handle no intersection case
    '''
    all_depths = []
    for i in range(len(points)):
        rot_verts = rotate_verts_2d(shape_verts, points[i], directions[i])
        all_depths.append(intersection_depth_2d(rot_verts, shape_edges)[1])
    return all_depths 

def find_probe(p1, p2, point, dir):
    '''
    Return an x,y coordinate where the line defined by a point and direction intersects the segment between p1 and p2
    Returns None if no such intersection exists
    '''

    # vertical line
    if dir[0] == 0.:
        # see if segment straddles the line
        trans_p1 = p1[0]-point[0]
        trans_p2 = p2[0]-point[0]
        if trans_p1 * trans_p2 <= 0.:
            wgt = abs(trans_p2) / (abs(trans_p1) + abs(trans_p2))
            return [p1[0]*wgt + p2[0]*(1-wgt), p1[1]*wgt + p2[1]*(1-wgt)]
        else:
            return None
    # horizontal line
    if dir[1] == 0.:
        # see if segment straddles the x axis
        trans_p1 = p1[1]-point[1]
        trans_p2 = p2[1]-point[1]
        if trans_p1 * trans_p2 <= 0.:
            wgt = abs(trans_p2) / (abs(trans_p2)+ abs(trans_p1))
            return [p1[0]*wgt + p2[0]*(1-wgt), p1[1]*wgt + p2[1]*(1-wgt)]
        else:
            return None


    # calculate the slope and intercept of the line
    slope = dir[1]/dir[0]
    b = point[1] - point[0]*slope
    
    # check if the segment is horizontal
    if p1[1] == p2[1]:
        y_sol = p1[1]
        x_sol = (y_sol - b) / slope
        if x_sol < max(p1[0], p2[0]) and x_sol > min(p1[0], p2[0]):
            return [x_sol, y_sol]
        else:
            return None
    # check if the segment is vertical
    if p1[0] == p2[0]:
        x_sol = p1[0]
        y_sol = x_sol * slope + b
        if y_sol < max(p1[1], p2[1]) and y_sol > min(p1[1], p2[1]):
            return [x_sol, y_sol]
        else:
            return None

    # check if the segment is parallel
    diff = p2 - p1
    seg_slope = diff[1] / diff[0]
    if seg_slope == slope:
        return None

    # handle general case (neither the segment or line are vertical and they aren't parallel)
    # intercept of the line defined by p1, p2
    c = p1[1] - p1[0] * seg_slope
    # set y equal to y, solve
    x_sol = (b-c)/(seg_slope-slope)
    y_sol = x_sol * seg_slope + c
    #  see if intersection is within segment bounds
    if x_sol < max(p1[0], p2[0]) and x_sol > min(p1[0], p2[0]):
        return [x_sol, y_sol]
    else:
        return None


def initialization(shape_verts, shape_edges, viewer=None):
    start_points, dirs, adj_list = start_points_2d()
    if viewer is not None:
        viewer.update_wrap(start_points, adj_list)
        viewer.display(i=0)
    depths = get_depths(shape_verts, shape_edges, start_points, dirs)
    points = [np.array(start_points[i]) + depths[i]*np.array(dirs[i]) if depths[i] is not None else start_points[i] for i in range(len(start_points))]
    if viewer is not None:
        viewer.update_curr_segments([[start_points[i], points[i]] for i in range(len(start_points))])
        viewer.display(i=0)
        viewer.retire_segments()
        viewer.update_wrap(points, adj_list)
        viewer.display(probes=False,i=1)
        # viewer.display(i)

    return points, dirs, depths, adj_list

def meshing(shape_verts, shape_edges, resolution=0.02):
    viewer = AlgVisualizer2D(shape_verts, shape_edges)
    points, dirs, depths, adj_list = initialization(shape_verts, shape_edges, viewer=viewer)
    dynamic_edges = [edge for edge in adj_list]

    # for i in range(len(points)):
    #     print(f"{i}  -  {points[i]}")
    # print(adj_list)

    iterations = 5
    for i in range(iterations-1):
        new_dynamic_edges = []
        new_dirs = []
        new_probes = []
        for edge in dynamic_edges:
            v1 = edge[0]
            v2 = edge[1]

            # only expand large edges
            v_dist = np.linalg.norm(np.array(points[v1]) - np.array(points[v2]))
            if v_dist <= resolution:
                continue
            else:
                adj_list.remove(edge)


            # find new probe point
            midpoint = (np.array(points[v1]) + np.array(points[v2]))/2.
            diff = points[v1] - points[v2]
            v1_probe = np.array(points[v1]) - depths[v1]*np.array(dirs[v1])
            v2_probe = np.array(points[v2]) - depths[v2]*np.array(dirs[v2])
            dir = np.array([-diff[1], diff[0]])
            dir /= np.linalg.norm(dir)
            new_probe = find_probe(points[v1], v1_probe, midpoint, dir)
            if new_probe is None:
                new_probe = find_probe(points[v2], v2_probe, midpoint, dir)
            if new_probe is None:
                new_probe = v1_probe if abs(np.dot(v1_probe - midpoint, dir)) > abs(np.dot(v2_probe - midpoint, dir)) else v2_probe

            # determine directions from probe
            n_verts_to_add = int(v_dist / resolution)
            for j in range(n_verts_to_add):
                interpolated_surface_point = points[v1] + (j+1.)/(n_verts_to_add+1.)*(points[v2]-points[v1])
                dir = interpolated_surface_point - new_probe
                dir = dir / np.linalg.norm(dir)
                new_dirs.append(dir)
                new_probes.append(new_probe)
                new_point_index = len(points) + len(new_dirs)-1 #new dirs has already had an element added this loop
                if j == 0:
                    new_dynamic_edges.append([v1, new_point_index])
                else:
                    new_dynamic_edges.append([new_point_index-1, new_point_index])
                if j == n_verts_to_add-1:
                    new_dynamic_edges.append([new_point_index, v2])
        
        print(f"Iteration: {i+1}")
        
        # show the baseline at the start of the iteration
        viewer.update_probe_points(new_probes)
        viewer.display(i=i+1)
        
        new_depths = get_depths(shape_verts, shape_edges, new_probes, new_dirs)
        new_points = [new_probes[x] + new_depths[x]*new_dirs[x] for x in range(len(new_probes))]
        points += new_points
        # show the probe segments and new vertices
        viewer.update_curr_segments([[new_probes[i], new_points[i]] for i in range(len(new_probes))])
        viewer.display(i=i+1)
        viewer.retire_segments()

        adj_list += new_dynamic_edges
        dynamic_edges = new_dynamic_edges
        depths += new_depths
        dirs += new_dirs

        # Show the newly updated wrap
        viewer.update_wrap(points, adj_list)
        viewer.display(probes=False, i=i+2)



def show_data(shape_verts, shape_edges, wrap_verts, wrap_edges):
    '''
    Displays the shrinkwrap algorithm components
    '''
    # plotting setup
    f, ax = plt.subplots(1,1)
    f.set_size_inches(8, 8)
    ax.set_ylim(-1.1,1.1)
    ax.set_xlim(-1.1,1.1)


    # input manipulation
    wrap_verts = np.array(wrap_verts)
    

    # plotting
    ax.scatter([0.], [0.], color="green")
    for edge in shape_edges:
        ax.plot(shape_verts[edge,0], shape_verts[edge,1], color="black")
    ax.scatter(wrap_verts[:,0], wrap_verts[:,1], color="red")
    for edge in wrap_edges:
        ax.plot(wrap_verts[edge,0], wrap_verts[edge,1], color="tab:blue")
    plt.show()


def show_rotation(shape_verts, shape_edges, v=40, n=100):
    '''
    Shows how the object is rotated to check the depth of the vth vertex from a circle of n vertices
    '''
    start_points, dirs, adj_list = start_points_2d(n_points=n)
    point = start_points[v]
    dir = dirs[v]
    rot_verts = rotate_verts_2d(shape_verts, point, dir)

    # plotting setup
    f, ax = plt.subplots(1,1)
    f.set_size_inches(8, 8)
    ax.set_ylim(-1.1,1.1)
    ax.set_xlim(-1.1,1.1)


    ax.scatter([point[0]], point[1], color="red")
    ax.plot([point[0], point[0]+0.2*dir[0]], [point[1], point[1]+0.2*dir[1]], color="orange")
    ax.scatter([0.], [0.], color="darkgreen")
    ax.plot([0.0, 0.2], [0.0,0.0], color="limegreen")
    for edge in shape_edges:
        ax.plot(shape_verts[edge][:,0], shape_verts[edge][:,1], color="black")
        ax.plot(rot_verts[edge][:,0], rot_verts[edge][:, 1], color="blue")
    plt.show()

if __name__ == "__main__":
    # start_points, start_directions, adj_list = start_points_2d()
    # show_data(shape_verts, shape_edges, start_points, adj_list)
    meshing(shape_verts, shape_edges)
    n=360
    # for i in range(0, n, 10):
    #     print(i)
    #     show_rotation(shape_verts, shape_edges, v=i, n=n)
