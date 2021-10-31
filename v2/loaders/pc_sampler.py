import argparse
import beacon.utils as butils
import trimesh
import torch
import math
from tqdm import tqdm
import multiprocessing as mp

from PyQt5.QtWidgets import QApplication
import numpy as np

from tk3dv.pyEasel import *
from Easel import Easel

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, '../'))
sys.path.append(os.path.join(FileDirPath, '../../'))
import odf_utils
from odf_dataset import DEFAULT_RADIUS, ODFDatasetVisualizer, ODFDatasetLiveVisualizer
import odf_v2_utils as o2utils

PC_SAMPLER_RADIUS = 1.25
PC_SAMPLER_THRESH = 0.05
PC_NEG_SAMPLER_THRESH = PC_SAMPLER_THRESH
PC_SUBDIVIDE_THRESH = PC_SAMPLER_THRESH * 2
PC_SAMPLER_NEG_MINOFFSET = PC_NEG_SAMPLER_THRESH * 2
PC_SAMPLER_NEG_MAXOFFSET = PC_SAMPLER_RADIUS
PC_SAMPLER_POS_RATIO = 0.5

class PointCloudSampler():
    def __init__(self, Vertices, VertexNormals, TargetRays):
        self.Vertices = Vertices
        self.VertexNormals = VertexNormals
        self.nTargetRays = TargetRays
        assert self.Vertices.shape[0] == self.VertexNormals.shape[0]
        # print('[ INFO ]: Found {} vertices with normals. Will try to sample {} rays in total.'.format(len(self.Vertices), self.nTargetRays))

        self.Coordinates = None
        self.Intersects = None
        self.Depths = None

        self.sample(self.nTargetRays)

    def sample(self, TargetRays, RatioPositive=PC_SAMPLER_POS_RATIO):
        Tic = []
        Toc = []
        Tic.append(butils.getCurrentEpochTime())
        nVertices = len(self.Vertices)
        TargetPositiveRays = math.floor(TargetRays*RatioPositive)
        TargetNegRays = (TargetRays-TargetPositiveRays)
        RaysPerVertex = math.ceil(TargetPositiveRays/nVertices)
        NegRaysPerVertex = math.ceil(TargetNegRays / nVertices)
        # print('[ INFO ]: Aiming for {} positive and {} negative ray samples, {} positive rays per vertex, {} negative rays per vertex.'.format(RaysPerVertex*nVertices, NegRaysPerVertex*nVertices, RaysPerVertex, NegRaysPerVertex))
        Toc.append(butils.getCurrentEpochTime())

        Tic.append(butils.getCurrentEpochTime())
        PCoordinates, PIntersects, PDepths = self.sample_positive(RaysPerVertex=RaysPerVertex, Target=TargetPositiveRays)
        NCoordinates, NIntersects, NDepths = self.sample_negative(RaysPerVertex=NegRaysPerVertex, Target=TargetNegRays)
        Toc.append(butils.getCurrentEpochTime())

        Tic.append(butils.getCurrentEpochTime())
        Coordinates = np.concatenate((PCoordinates, NCoordinates), axis=0)
        Intersects = np.concatenate((PIntersects, NIntersects), axis=0)
        Depths = np.concatenate((PDepths, NDepths), axis=0)
        Toc.append(butils.getCurrentEpochTime())

        Tic.append(butils.getCurrentEpochTime())
        ShuffleIdx = np.random.permutation(len(Coordinates))
        Toc.append(butils.getCurrentEpochTime())
        # ShuffleIdx = np.arange(len(Coordinates))
        # print('[ INFO ]: Sampled {} rays -- {} positive and {} negative.'.format(len(Coordinates), len(PCoordinates), len(NCoordinates)))
        # print('Prep time: {}ms.'.format((Toc[0]-Tic[0])*1e-3))
        # print('Sample time: {}ms.'.format((Toc[1] - Tic[1]) * 1e-3))
        # print('Concat time: {}ms.'.format((Toc[2] - Tic[2]) * 1e-3))
        # print('Permute time: {}ms.'.format((Toc[3] - Tic[3]) * 1e-3))

        self.Coordinates = torch.from_numpy(Coordinates[ShuffleIdx]).to(torch.float32)
        self.Intersects = torch.from_numpy(Intersects[ShuffleIdx]).to(torch.float32).unsqueeze(1)
        self.Depths = torch.from_numpy(Depths[ShuffleIdx]).to(torch.float32).unsqueeze(1)

    def sample_negative(self, RaysPerVertex, Target):
        # Numpy version - seems faster
        Tic = butils.getCurrentEpochTime()
        nVertices = len(self.Vertices)
        # Randomly offset vertices
        RandomDistances = np.random.uniform(PC_SAMPLER_NEG_MINOFFSET, PC_SAMPLER_NEG_MAXOFFSET, len(self.Vertices))
        Offsets = RandomDistances[:, np.newaxis] * self.VertexNormals
        OffsetVertices = self.Vertices + Offsets
        SampledDirections = np.zeros((RaysPerVertex * nVertices, 3))
        VertexRepeats = np.zeros((RaysPerVertex * nVertices, 3))
        ValidDirCtr = 0
        for VCtr in (range(nVertices)):
            ValidDirs = o2utils.sample_directions_prune_numpy(RaysPerVertex, vertex=OffsetVertices[VCtr], points=self.Vertices, thresh=PC_NEG_SAMPLER_THRESH)
            SampledDirections[ValidDirCtr:ValidDirCtr + len(ValidDirs)] = ValidDirs
            VertexRepeats[ValidDirCtr:ValidDirCtr + len(ValidDirs)] = OffsetVertices[np.newaxis, VCtr]
            ValidDirCtr += len(ValidDirs)

        SampledDirections = SampledDirections[:ValidDirCtr]
        VertexRepeats = VertexRepeats[:ValidDirCtr]
        # print('[ INFO ]: Only able to sample {} valid negative rays out of {} requested.'.format(ValidDirCtr, Target))

        # For each normal direction, find the point on a sphere of radius PC_RADIUS
        SpherePoints, Distances = o2utils.find_sphere_points(OriginPoints=VertexRepeats, Directions=SampledDirections,
                                                             SphereCenter=np.zeros(3), Radius=PC_SAMPLER_RADIUS)

        Coordinates = np.asarray(np.hstack((SpherePoints, - SampledDirections)))
        Intersects = np.asarray(np.zeros_like(Distances))
        Depths = np.asarray(np.zeros_like(Distances))

        SpherePointsNorm = np.linalg.norm(SpherePoints, axis=1)
        ValidPointsIdx = np.abs(SpherePointsNorm - PC_SAMPLER_RADIUS) < 0.01 # Epsilon
        # print('Invalid idx:', len(Coordinates) - np.sum(ValidPointsIdx))
        Coordinates = Coordinates[ValidPointsIdx]
        Intersects = Intersects[ValidPointsIdx]
        Depths = Depths[ValidPointsIdx]
        # print('[ INFO ]: Only able to sample {} valid rays out of {} requested.'.format(len(Coordinates), Target))

        Toc = butils.getCurrentEpochTime()
        # print('[ INFO ]: Numpy processed in {}ms.'.format((Toc - Tic) * 1e-3))

        return Coordinates[:Target], Intersects[:Target], Depths[:Target]

    def sample_positive(self, RaysPerVertex, Target):
        # Numpy version - seems faster
        Tic = butils.getCurrentEpochTime()
        nVertices = len(self.Vertices)
        SampledDirections = np.zeros((RaysPerVertex*nVertices, 3))
        VertexRepeats = np.zeros((RaysPerVertex*nVertices, 3))
        ValidDirCtr = 0
        for VCtr in (range(nVertices)):
            # SampledDirections[VCtr*RaysPerVertex:(VCtr+1)*RaysPerVertex] = self.sample_directions_numpy(RaysPerVertex, normal=self.VertexNormals[VCtr])
            ValidDirs = o2utils.sample_directions_prune_normal_numpy(RaysPerVertex, vertex=self.Vertices[VCtr], normal=self.VertexNormals[VCtr], points=self.Vertices, thresh=PC_SAMPLER_THRESH)
            SampledDirections[ValidDirCtr:ValidDirCtr+len(ValidDirs)] = ValidDirs
            VertexRepeats[ValidDirCtr:ValidDirCtr+len(ValidDirs)] = self.Vertices[VCtr]
            ValidDirCtr += len(ValidDirs)

        SampledDirections = SampledDirections[:ValidDirCtr]
        VertexRepeats = VertexRepeats[:ValidDirCtr]
        # print('[ INFO ]: Only able to sample {} valid rays out of {} requested.'.format(ValidDirCtr, Target))

        # For each normal direction, find the point on a sphere of radius PC_RADIUS
        SpherePoints, Distances = o2utils.find_sphere_points(OriginPoints=VertexRepeats, Directions=SampledDirections,
                                                             SphereCenter=np.zeros(3), Radius=PC_SAMPLER_RADIUS)
        Coordinates = np.asarray(np.hstack((SpherePoints, - SampledDirections)))
        Intersects = np.asarray(np.ones_like(Distances))
        Depths = np.asarray(Distances)

        SpherePointsNorm = np.linalg.norm(SpherePoints, axis=1)
        ValidPointsIdx = np.abs(SpherePointsNorm - PC_SAMPLER_RADIUS) < 0.1 # Epsilon
        Coordinates = Coordinates[ValidPointsIdx]
        Intersects = Intersects[ValidPointsIdx]
        Depths = Depths[ValidPointsIdx]
        # print('[ INFO ]: Only able to sample {} valid rays out of {} requested.'.format(len(Coordinates), Target))

        Toc = butils.getCurrentEpochTime()
        # print('[ INFO ]: Numpy processed in {}ms.'.format((Toc-Tic)*1e-3))

        return Coordinates[:Target], Intersects[:Target], Depths[:Target]

    def __getitem__(self, item):
        return self.Coordinate[item], (self.Intersects[item], self.Depths[item])

    def __len__(self):
        return len(self.Depths)

Parser = argparse.ArgumentParser()
Parser.add_argument('-i', '--input', help='Specify the input point cloud and normals in OBJ format.', required=True)
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('-v', '--viz-limit', help='Limit visualizations to these many rays.', required=False, type=int, default=1000)
Parser.add_argument('-n', '--target-rays', help='Attempt to sample n rays per vertex.', required=False, type=int, default=100)

if __name__ == '__main__':
    Args = Parser.parse_args()
    butils.seedRandom(Args.seed)

    Mesh = trimesh.load(Args.input)
    Verts = odf_utils.mesh_normalize(Mesh.vertices)
    # Verts, Faces = trimesh.remesh.subdivide_to_size(Verts, Mesh.faces, max_edge=PC_SUBDIVIDE_THRESH)
    # Mesh = trimesh.Trimesh(vertices=Verts, faces=Faces, process=False)
    VertNormals = Mesh.vertex_normals.copy()
    Norm = np.linalg.norm(VertNormals, axis=1)
    VertNormals /= Norm[:, None]

    Sampler = PointCloudSampler(Verts, VertNormals, TargetRays=Args.target_rays)

    app = QApplication(sys.argv)

    if len(Sampler) < Args.viz_limit:
        Args.viz_limit = len(Sampler)
    mainWindow = Easel([ODFDatasetLiveVisualizer(coord_type='direction', rays=Sampler.Coordinates.cpu(), intersects=Sampler.Intersects.cpu(), depths=Sampler.Depths.cpu(), DataLimit=Args.viz_limit)], sys.argv[1:])
    mainWindow.show()
    sys.exit(app.exec_())

