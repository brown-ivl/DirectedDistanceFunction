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

    def sample(self):
        # For each normal direction, find the point on a sphere of radius DEFAULT_RADIUS
        # print(np.mean(self.Vertices, axis=0))
        # Line-Sphere intersection: https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        o = self.Vertices
        u = self.VertexNormals
        c = np.array([0, 0, 0])
        OminusC = o - c
        DotP = np.sum(np.multiply(u, OminusC), axis=1)
        Delta = DotP**2 - (np.linalg.norm(OminusC, axis=1) - DEFAULT_RADIUS**2)
        d = - DotP + np.sqrt(Delta)
        SpherePoints = o + np.multiply(u, d[:, np.newaxis])

        self.Coordinates = torch.from_numpy(np.asarray(np.hstack((SpherePoints, - u)))).to(torch.float32)
        self.Intersects = torch.from_numpy(np.asarray(np.ones_like(d))).to(torch.float32)
        self.Depths = torch.from_numpy(np.asarray(d))

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

    Sampler = PointCloudSampler(Verts, VertNormals)

    app = QApplication(sys.argv)

    if len(Sampler) < Args.viz_limit:
        Args.viz_limit = len(Sampler)
    # mainWindow = Easel([ODFDatasetVisualizer(Data[:int(Args.viz_limit * Args.nsamples)])], sys.argv[1:])
    mainWindow = Easel([ODFDatasetLiveVisualizer(coord_type='direction', rays=Sampler.Coordinates, intersects=Sampler.Intersects, depths=Sampler.Depths)], sys.argv[1:])
    mainWindow.show()
    sys.exit(app.exec_())

