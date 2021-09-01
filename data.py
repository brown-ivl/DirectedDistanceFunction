'''
A data loader class for rays
'''

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterization
import sampling
import utils


class DepthData(Dataset):

    def __init__(self,faces,verts,radius,sampling_methods,sampling_frequency,size=200000):
        '''
        Faces and verts define a mesh object that is used to generate data
        sampling_methods are methods from sampling.py that are used to choose rays during data generation
        sampling_frequency are weights determining how frequently each sampling method should be used (weights should sum to 1.0)
        size defines the number of datapoints to generate
        '''
        assert(sum(sampling_frequency)==1.0)
        self.faces = faces
        self.verts = verts
        self.vert_normals = utils.get_vertex_normals(verts,faces)
        self.radius=radius
        self.near_face_threshold = rasterization.max_edge(verts, faces)
        self.sampling_methods = sampling_methods
        self.sampling_frequency = sampling_frequency
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sampling_method = self.sampling_methods[np.random.choice(np.arange(len(self.sampling_methods)), p=self.sampling_frequency)]
        ray_start,ray_end,v = sampling_method(self.radius,verts=self.verts,vert_normals=self.vert_normals)
        direction = ray_end-ray_start
        direction /= np.linalg.norm(direction)
        rot_verts = rasterization.rotate_mesh(self.verts, ray_start, ray_end)
        occ, depth = rasterization.ray_occ_depth(self.faces, rot_verts, ray_start_depth=np.linalg.norm(ray_end-ray_start), near_face_threshold=self.near_face_threshold, v=v)
        intersect = 1.0 if depth != np.inf else 0.0
        # theta,phi = utils.vector_to_angles(ray_end-ray_start)
        return {
            # 5d coordinates - [x,y,z,theta,phi]
            # "coordinates": torch.tensor([ray_start[0], ray_start[1], ray_start[2], theta, phi], dtype=torch.float32),
            "coordinates": torch.tensor([x for val in list(ray_start)+list(direction) for x in utils.positional_encoding(val)]),
            # is the ray origin inside the mesh?
            "occ": torch.tensor(occ, dtype=torch.float32),
            # does the ray intersect the mesh?
            "intersect": torch.tensor(intersect, dtype=torch.float32),
            # Depth at which the ray intersects the mesh (positive)
            "depth": torch.tensor(depth, dtype=torch.float32),
        }