'''
A data loader class for rays
'''

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterization
import sampling
import odf_utils


class DepthData(Dataset):

    def __init__(self,faces,verts,radius,sampling_methods,sampling_frequency,size=1000000):
        '''
        Faces and verts define a mesh object that is used to generate data
        sampling_methods are methods from sampling.py that are used to choose rays during data generation
        sampling_frequency are weights determining how frequently each sampling method should be used (weights should sum to 1.0)
        size defines the number of datapoints to generate
        '''
        assert(sum(sampling_frequency)>=0.99)
        self.faces = faces
        self.verts = verts
        self.vert_normals = odf_utils.get_vertex_normals(verts, faces)
        self.radius=radius
        self.near_face_threshold = rasterization.max_edge(verts, faces)
        self.sampling_methods = sampling_methods
        self.sampling_frequency = sampling_frequency
        self.size = size

    def __len__(self):
        return self.size

    @staticmethod
    def collate_fn(batch):
        #print(batch)
        data = [torch.vstack([item[0] for item in batch])]
        target = [(torch.vstack([item[1][0] for item in batch]), torch.vstack([item[1][1] for item in batch]))]
        #print(data, target)
        return (data, target) 

    def __getitem__(self, index):
        sampling_method = self.sampling_methods[np.random.choice(np.arange(len(self.sampling_methods)), p=self.sampling_frequency)]
        ray_start,ray_end,v = sampling_method(self.radius,verts=self.verts,vert_normals=self.vert_normals)
        direction = ray_end-ray_start
        direction /= np.linalg.norm(direction)
        rot_verts = rasterization.rotate_mesh(self.verts, ray_start, ray_end)
        occ, depth = rasterization.ray_occ_depth(self.faces, rot_verts, ray_start_depth=np.linalg.norm(ray_end-ray_start), near_face_threshold=self.near_face_threshold, v=v)
        intersect = 1.0 if depth != np.inf else 0.0
        depth = 1.0 if depth==np.inf else depth
        # theta,phi = utils.vector_to_angles(ray_end-ray_start)
        return torch.tensor([list(ray_start)+list(direction)], dtype=torch.float32), (torch.tensor(intersect), torch.tensor(depth))


class MultiDepthDataset(Dataset):

    def __init__(self,faces,verts,radius,sampling_methods,sampling_frequency,size=1000000, intersect_limit=20, pos_enc=True):
        '''
        Faces and verts define a mesh object that is used to generate data
        sampling_methods are methods from sampling.py that are used to choose rays during data generation
        sampling_frequency are weights determining how frequently each sampling method should be used (weights should sum to 1.0)
        size defines the number of datapoints to generate
        '''
        self.faces = faces
        self.verts = verts
        self.vert_normals = odf_utils.get_vertex_normals(verts, faces)
        self.radius=radius
        self.near_face_threshold = rasterization.max_edge(verts, faces)
        self.size = size
        self.intersect_limit = intersect_limit
        self.pos_enc = pos_enc
        self.sampling_methods = sampling_methods
        self.sampling_frequency = sampling_frequency

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sampling_method = self.sampling_methods[np.random.choice(np.arange(len(self.sampling_methods)), p=self.sampling_frequency)]
        ray_start,ray_end,_ = sampling_method(self.radius,verts=self.verts,vert_normals=self.vert_normals)
        direction = ray_end-ray_start
        direction /= np.linalg.norm(direction)
        rot_verts = rasterization.rotate_mesh(self.verts, ray_start, ray_end)
        int_depths = rasterization.ray_all_depths(self.faces, rot_verts,near_face_threshold=self.near_face_threshold, ray_start_depth=np.linalg.norm(ray_end - ray_start))
        int_depths = torch.tensor(int_depths[:self.intersect_limit], dtype=torch.float32)
        intersect = np.zeros((self.intersect_limit,), dtype=float)
        intersect[:int_depths.shape[0]] = 1.
        # changed intersect to be an integer now that we're using CELoss instead of BCE
        n_ints = int_depths.shape[0]
        depths = np.zeros((self.intersect_limit,), dtype=float)
        depths[:int_depths.shape[0]] = int_depths
        if self.pos_enc:
            coordinates_points = torch.tensor([x for val in list(ray_start) + list(ray_end) for x in odf_utils.positional_encoding(val)], dtype=torch.float32)
            coordinates_direction = torch.tensor([x for val in list(ray_start) + list(direction) for x in odf_utils.positional_encoding(val)], dtype=torch.float32)
            coordinates_pluecker = torch.tensor([x for val in list(direction) + list(np.cross(ray_start, direction)) for x in odf_utils.positional_encoding(val)], dtype=torch.float32)
        else:
            coordinates_points = torch.tensor([list(ray_start)+list(ray_end)], dtype=torch.float32)
            coordinates_direction = torch.tensor([list(ray_start)+list(direction)], dtype=torch.float32)
            coordinates_pluecker = torch.tensor([list(direction)+list(np.cross(ray_start, direction))], dtype=torch.float32)
        return {
            # pos encoding
            # "coordinates": torch.tensor([x for val in list(ray_start)+list(direction) for x in utils.positional_encoding(val)]),
            # 6D coords
            "coordinates_points": coordinates_points,
            "coordinates_direction": coordinates_direction,
            "coordinates_pluecker": coordinates_pluecker,
            # Number of intersections the ray has (capped at self.intersect_limit)
            "n_ints": n_ints,
            # does the ray have an nth intersection?
            "intersect": torch.tensor(intersect, dtype=torch.float32),
            # Depth at which the ray intersects the mesh (positive)
            "depths": torch.tensor(depths, dtype=torch.float32),
        }

class DepthConsistencyDataset(Dataset):

    def __init__(self,faces,verts,radius,size=1000000, intersect_limit=20, pos_enc=True):
        pass
        # TODO: Make consistency loss data loader
