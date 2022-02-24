import argparse
import beacon.utils as butils
import trimesh
import torch
import math
from tqdm import tqdm
import multiprocessing as mp
from threading import Thread

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

DEPTH_SAMPLER_RADIUS = 1.25
DEPTH_SAMPLER_POS_RATIO = 0.5

# Assume input in Shivam's file format
# For each object, I have dumped rendered data from 50 randomly sampled viewpoints. Each viewpoint has following data:
# unprojected_normalized_pts : ray endpoints, lying on the object mesh. some points will have depth -1 (which can be filtered out as: unprojected_normalized_pts[unprojected_normalized_pts[:,2]!=-1])
# viewpoint : ray start points lying on sphere on radius 1.25
# depth_map : rendered depth map
# rest elements are camera intrinsics and extrinsics.
class DepthMapSampler():
    def __init__(self, NPData, TargetRays, UsePosEnc=False):
        self.NPData = NPData
        self.nTargetRays = TargetRays
        self.UsePosEnc = UsePosEnc
        # print('[ INFO ]: Found {} vertices with normals. Will try to sample {} rays in total.'.format(len(self.Vertices), self.nTargetRays))

        self.Coordinates = None
        self.Intersects = None
        self.Depths = None

        self.Interior = np.min(NPData["depth_map"]) < 0.

        self.sample(self.nTargetRays)

    def sample(self, TargetRays, RatioPositive=DEPTH_SAMPLER_POS_RATIO):
        AllEndPoints = self.NPData['unprojected_normalized_pts']
        StartPoint = self.NPData['viewpoint'] # There is only 1 start point, the camera center
        AllStartPoints = np.tile(StartPoint, (AllEndPoints.shape[0], 1))
        AllIntersects = self.NPData['invalid_depth_mask']

        nPosTargetRays = math.floor(TargetRays*RatioPositive)
        nNegTargetRays = (TargetRays-nPosTargetRays)

        AllPosIdx = np.where(AllIntersects == False)[0]
        AllNegIdx = np.where(AllIntersects == True)[0]
        PosShuffleIdx = np.random.permutation(len(AllPosIdx))
        NegShuffleIdx = np.random.permutation(len(AllNegIdx))


        # TODO: make the depths negative if they should be negative
        SampledPosEndPts = AllEndPoints[AllPosIdx[PosShuffleIdx[:nPosTargetRays]]]
        SampledPosStartPts = AllStartPoints[AllPosIdx[PosShuffleIdx[:nPosTargetRays]]]
        SampledPosDir = SampledPosEndPts - SampledPosStartPts
        SampledPosDirNorm = np.linalg.norm(SampledPosDir, axis=1)
        SampledPosDir /= SampledPosDirNorm[:, np.newaxis]
        SampledPosDepths = np.linalg.norm(SampledPosEndPts - SampledPosStartPts, axis=1)

        SampledPosIntersects = np.ones((len(SampledPosEndPts), 1))
        if self.Interior:
            SampledPosDir = -1. * SampledPosDir
            SampledPosDepths =-1. * SampledPosDepths

        SampledNegEndPts = AllEndPoints[AllNegIdx[NegShuffleIdx[:nNegTargetRays]]]
        SampledNegStartPts = AllStartPoints[AllNegIdx[NegShuffleIdx[:nNegTargetRays]]]
        SampledNegDir = SampledNegEndPts - SampledNegStartPts
        SampledNegDirNorm = np.linalg.norm(SampledNegDir, axis=1)
        SampledNegDir /= SampledNegDirNorm[:, np.newaxis]
        SampledNegDepths = np.zeros(len(SampledNegEndPts))
        SampledNegIntersects = np.zeros((len(SampledNegEndPts), 1))
        if self.Interior:
            # All interior rays should intersect
            assert(AllNegIdx.shape[0] == 0)
            # SampledNegDir = -1. * SampledNegDir
            # SampledNegDepths =-1. * SampledNegDepths

        Coordinates = np.vstack((np.hstack((SampledPosStartPts, SampledPosDir)), np.hstack((SampledNegStartPts, SampledNegDir))))
        Intersects = np.vstack((SampledPosIntersects, SampledNegIntersects))
        Depths = np.expand_dims(np.concatenate((SampledPosDepths, SampledNegDepths)), axis=1)

        ShuffleIdx = np.random.permutation(len(Coordinates))

        self.Coordinates = torch.from_numpy(Coordinates[ShuffleIdx]).to(torch.float32)
        self.Intersects = torch.from_numpy(Intersects[ShuffleIdx]).to(torch.float32)
        self.Depths = torch.from_numpy(Depths[ShuffleIdx]).to(torch.float32)

    def __getitem__(self, item):
        return self.Coordinates[item], (self.Intersects[item], self.Depths[item])

    def __len__(self):
        return len(self.Depths)
