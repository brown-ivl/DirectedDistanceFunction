import numpy as np
import matplotlib.pyplot as plt
import math

shape_verts = [
    [-0.5, 0.1],
    [-0.5, 0.2],
    [-0.2, 0.5],
    [-0.2, 0.7],
    [0.0, 0.8],
    [0.6, 0.8],
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
]

shape_edges = [[i, i+1] for i in range(len(shape_verts)-1)] + [[len(shape_verts)-1, 0]]

def intersection_depth_2d(point, direction, seg1, seg2):
    '''
    Takes in a ray (point, direction), and a line segment (seg1, seg2) defined by two endpoints
    Returns a binary intersection value, which is true iff the ray intersects the segment in the positive direction
    Also returns a depth value at which the intersection occurs
    '''
    seg1 = seg1 - point
    seg2 = seg2 - point
    theta = math.atan(direction[1] / direction[0])
    rot_mat = np.array([[math.cos(-theta), -math.sin(-theta)],[math.sin(-theta), math.cos(-theta)]])
    seg1 = np.matmul([seg1], rot_mat)[0]
    seg2 = np.matmul([seg2], rot_mat)[0]
    # intersected iff one endpoint on each side of x axis
    if int(seg1[1] > 0) + int(seg2[1] > 0) == 1:
        depth = (abs(seg1[1])/(abs(seg1[1]) + abs(seg2[1]))) * seg2[0] + (abs(seg2[1])/(abs(seg1[1]) + abs(seg2[1]))) * seg1[0]
        if depth < 0.:
            return False, None
        else:
            return True, depth
    else:
        return False, None

def start_points_2d(n_points=100):
    '''
    Returns circle circumference start points, starting directions, and adjacency list
    '''
    start_points = [[math.cos(i*2.*math.pi), math.sin(i*2.*math.pi)] for i in range(n_points)]
    start_directions = [[-math.cos(i*2.*math.pi), -math.sin(i*2.*math.pi)] for i in range(n_points)]
    adj_list = [[i, i+1] for i in range(n_points)] + [n_points-1, 0]
    return start_points, start_directions, adj_list

def show_data(shape_verts, shape_edges, wrap_verts, wrap_edges):
    '''
    Displays the shrinkwrap algorithm components
    '''
    f, ax = plt.subplots(1,1)
    shape_verts = np.array(shape_verts + [shape_verts[0]])
    ax.plot(shape_verts[:,0], shape_verts[:,1])
    plt.show()


if __name__ == "__main__":
    show_data(shape_verts, shape_edges, [], [])