import argparse
import random
import beacon.utils as butils
import trimesh
import torch
import math
from tqdm import tqdm
import multiprocessing as mp
from itertools import repeat
from functools import partial

import tk3dv.nocstools.datastructures as ds
from PyQt5.QtWidgets import QApplication
import PyQt5.QtCore as QtCore
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
import numpy as np

from tk3dv.pyEasel import *
from EaselModule import EaselModule
from Easel import Easel
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, '../'))
sys.path.append(os.path.join(FileDirPath, '../../'))
import odf_utils
from odf_dataset import DEFAULT_RADIUS, ODFDatasetVisualizer, ODFDatasetLiveVisualizer

PC_VERT_NOISE = 0.02
PC_TAN_NOISE = 0.02
PC_UNIFORM_RATIO = 100
PC_VERT_RATIO = 0
PC_TAN_RATIO = 0
PC_RADIUS = 1.25
PC_MAX_INTERSECT = 1

class PointCloudSampler():
    def __init__(self, Vertices, VertexNormals):
        self.Vertices = Vertices
        self.VertexNormals = VertexNormals
        assert self.Vertices.shape[0] == self.VertexNormals.shape[0]
        print('[ INFO ]: Found {} vertices with normals.'.format(self.Vertices.shape[0]))

        self.Coordinates = None
        self.Intersects = None
        self.Depths = None

        self.sample()

    @staticmethod
    def sample_directions_numpy(nPoints, device='cpu', ndim=3):
        vec = np.random.randn(nPoints, ndim)
        vec /= np.linalg.norm(vec, axis=0)
        return vec

    @staticmethod
    def sample_directions_torch(nDirs, normal=None, device='cpu', ndim=3, dtype=torch.float32):
        # Sample more than needed then prume
        vec = torch.randn((nDirs, ndim), device=device, dtype=dtype)
        vec /= torch.linalg.norm(vec, axis=0)
        if normal is not None:
            # Select only if direction is in the sae half space as normal
            DotP = torch.sum(torch.mul(vec, normal), dim=1)
            ValidIdx = DotP > 0.0
            InvalidIdx = DotP <= 0.0
            # vec = vec[ValidIdx]
            vec[InvalidIdx] = normal # Set invalid to just be normal
        return vec

    def sample(self, RaysPerVertex=10):
        # For each normal direction, find the point on a sphere of radius DEFAULT_RADIUS
        # print(np.mean(self.Vertices, axis=0))
        # Line-Sphere intersection: https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        Tic = butils.getCurrentEpochTime()

        # Torch version
        nVertices = len(self.Vertices)
        SampledDirections = torch.zeros((RaysPerVertex*nVertices, 3), dtype=self.Vertices.dtype, device=self.Vertices.device)
        for VCtr in range(nVertices):
            SampledDirections[VCtr*RaysPerVertex:(VCtr+1)*RaysPerVertex] = self.sample_directions_torch(RaysPerVertex, normal=self.VertexNormals[VCtr], device=self.Vertices.device, dtype=self.Vertices.dtype)

        Repeats = [RaysPerVertex]*nVertices
        o = torch.repeat_interleave(self.Vertices, torch.tensor(Repeats), dim=0)
        # u = self.VertexNormals
        u = SampledDirections
        c = torch.zeros(1, 3).to(u.device)
        OminusC = o - c
        DotP = torch.sum(torch.mul(u, OminusC), dim=1)
        Delta = DotP**2 - (torch.linalg.norm(OminusC, axis=1) - DEFAULT_RADIUS**2)
        d = - DotP + torch.sqrt(Delta)
        SpherePoints = o + torch.mul(u, d[:, None])

        self.Coordinates = torch.hstack((SpherePoints, - u)).to(torch.float32)
        self.Intersects = torch.ones_like(d).to(torch.float32)
        self.Depths = d.to(torch.float32)


        # # Numpy version
        # o = self.Vertices.numpy()
        # u = self.VertexNormals.numpy()
        # c = np.array([0, 0, 0])
        # OminusC = o - c
        # DotP = np.sum(np.multiply(u, OminusC), axis=1)
        # Delta = DotP**2 - (np.linalg.norm(OminusC, axis=1) - DEFAULT_RADIUS**2)
        # d = - DotP + np.sqrt(Delta)
        # SpherePoints = o + np.multiply(u, d[:, np.newaxis])
        #
        # self.Coordinates = torch.from_numpy(np.asarray(np.hstack((SpherePoints, - u)))).to(torch.float32)
        # self.Intersects = torch.from_numpy(np.asarray(np.ones_like(d))).to(torch.float32)
        # self.Depths = torch.from_numpy(np.asarray(d))

        Toc = butils.getCurrentEpochTime()
        print('[ INFO ]: Processed in {}ms.'.format((Toc-Tic)*1e-3))


    def __getitem__(self, item):
        return self.Coordinate[item], (self.Intersects[item], self.Depths[item])

    def __len__(self):
        return len(self.Depths)

Parser = argparse.ArgumentParser()
Parser.add_argument('-i', '--input', help='Specify the input point cloud and normals in OBJ format.', required=True)
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('-v', '--viz-limit', help='Limit visualizations to these many rays.', required=False, type=int, default=1000)

if __name__ == '__main__':
    Args = Parser.parse_args()
    butils.seedRandom(Args.seed)

    Mesh = trimesh.load(Args.input)
    Verts = Mesh.vertices
    Verts = odf_utils.mesh_normalize(Verts)
    VertNormals = Mesh.vertex_normals.copy()
    Norm = np.linalg.norm(VertNormals, axis=1)
    VertNormals /= Norm[:, None]

    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Device = 'cpu'
    Sampler = PointCloudSampler(torch.from_numpy(Verts).to(Device), torch.from_numpy(VertNormals).to(Device))

    app = QApplication(sys.argv)

    if len(Sampler) < Args.viz_limit:
        Args.viz_limit = len(Sampler)
    mainWindow = Easel([ODFDatasetLiveVisualizer(coord_type='direction', rays=Sampler.Coordinates.cpu(), intersects=Sampler.Intersects.cpu(), depths=Sampler.Depths.cpu())], sys.argv[1:])
    mainWindow.show()
    sys.exit(app.exec_())

