'''
A data loader class for rays
'''
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, '../'))
#sys.path.append(os.path.join(FileDirPath, '../v3_utils'))
sys.path.append(os.path.join(FileDirPath, '../../'))
import v3_utils
#import rasterization
#import sampling
#import odf_utils


def DepthData(faces,verts,radius,sampling_methods,sampling_frequency,size=1000000):
        '''
        Faces and verts define a mesh object that is used to generate data
        sampling_methods are methods that are used to choose rays during data generation
        sampling_frequency are weights determining how frequently each sampling method should be used (weights should sum to 1.0)
        size defines the number of datapoints to generate
        '''
        assert(sum(sampling_frequency)==1.0)
        vert_normals = v3_utils.get_vertex_normals(verts, faces)
        near_face_threshold = v3_utils.max_edge(verts, faces)
        sampling_method = sampling_methods[np.random.choice(np.arange(len(sampling_methods)), 
                                           p=sampling_frequency)]
        ray_start,ray_end,v = sampling_method(radius,
                                              verts=verts,
                                              vert_normals=vert_normals)
        direction = ray_end-ray_start
        direction /= np.linalg.norm(direction)
        rot_verts = v3_utils.rotate_mesh(verts, ray_start, ray_end)
        occ, depth = v3_utils.ray_occ_depth(faces, 
                                            rot_verts, 
                                            ray_start_depth=np.linalg.norm(ray_end-ray_start), 
                                            near_face_threshold=near_face_threshold, 
                                            v=v)
        intersect = 1.0 if depth != np.inf else 0.0
        depth = 1.0 if depth == np.inf else depth
        return (torch.tensor([list(ray_start) + list(direction)]).float(),
                torch.tensor(intersect),
                torch.tensor(depth))
