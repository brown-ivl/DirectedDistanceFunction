import argparse
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

import v3_utils

DEPTH_SAMPLER_RADIUS = 1.25
DEPTH_SAMPLER_POS_RATIO = 0.5

# Assume input in Shivam's file format
# For each object, I have dumped rendered data from 50 randomly sampled viewpoints. Each viewpoint has following data:
# unprojected_normalized_pts : ray endpoints, lying on the object mesh. some points will have depth -1 (which can be filtered out as: unprojected_normalized_pts[unprojected_normalized_pts[:,2]!=-1])
# viewpoint : ray start points lying on sphere on radius 1.25
# depth_map : rendered depth map
# rest elements are camera intrinsics and extrinsics.
class DepthMapSampler():
    def __init__(self, NPData, TargetRays, UsePosEnc=False, Aug=True):
        self.NPData = NPData
        self.nTargetRays = TargetRays
        self.UsePosEnc = UsePosEnc
        # print('[ INFO ]: Found {} vertices with normals. Will try to sample {} rays in total.'.format(len(self.Vertices), self.nTargetRays))

        self.Coordinates = None
        self.Intersects = None
        self.Depths = None
        self.Aug = Aug
        self.Interior = np.min(NPData["depth_map"]) < 0.

        self.sample(self.nTargetRays)

    def sample(self, TargetRays, RatioPositive=DEPTH_SAMPLER_POS_RATIO):
        AllEndPoints = self.NPData['unprojected_normalized_pts']
        StartPoint = self.NPData['viewpoint'] # There is only 1 start point, the camera center
        if len(StartPoint.shape)==1:
            AllStartPoints = np.tile(StartPoint, (AllEndPoints.shape[0], 1))
        else:
            AllStartPoints = StartPoints
        AllIntersects = self.NPData['invalid_depth_mask']

        nPosTargetRays = math.floor(TargetRays*RatioPositive) # 65536
        nNegTargetRays = (TargetRays-nPosTargetRays) # 65536

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
        #if self.Interior:
        #    SampledPosDir = -1. * SampledPosDir
        #    SampledPosDepths =-1. * SampledPosDepths

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

        if self.Aug:
            mask1 = self.Intersects.to(bool).view(-1)
            mask2 = torch.logical_not(mask1)
            # augment non-intersecting points
            nonIntCoor = self.Coordinates[mask2]
            self.Coordinates[:,:3][mask2] = nonIntCoor[:,:3]+nonIntCoor[:,3:]*torch.rand(len(nonIntCoor), 1)*2.6
            # augment intersecting points from start point to surface
            intCoor = self.Coordinates[mask1]
            #shift = self.Depths[mask1]*torch.rand(len(intCoor), 1)
            shift = self.Depths[mask1]*(torch.rand(len(intCoor), 1))#*0.5+0.5)
            self.Coordinates[:,:3][mask1] = intCoor[:,:3]+intCoor[:,3:]*shift
            self.Depths[mask1] -= shift

            mask3 = torch.logical_and((self.Depths>0).view(-1), mask1)
            #mask4 = torch.logical_and((self.Depths<0).view(-1), mask1)
            # augment inside points from outside points
            outToInCoor = self.Coordinates[mask3]
            outToInInt = torch.ones((len(outToInCoor), 1))
            outToInDepths = (0.001+torch.rand(len(outToInCoor), 1)*0.1)
            #outToInDepths = (0.001+torch.rand(len(outToInCoor), 1)*0.05)
            outToInCoor[:,:3] = outToInCoor[:,:3]+outToInCoor[:,3:]*(self.Depths[mask3]+outToInDepths)
            outToInCoor[:,3:] *= -1
            # augment outside points from inside points
            #inToOutCoor = self.Coordinates[mask4]
            #inToOutInt = torch.ones((len(inToOutCoor), 1))
            #inToOutDepths = 0.001+torch.rand(len(inToOutCoor), 1)*0.1
            #inToOutCoor[:,:3] = inToOutCoor[:,:3]+inToOutCoor[:,3:]*(self.Depths[mask4]-inToOutDepths)   

            # augment intersecting points from the surface points
            # this data will not have an influence on depth branch
            
            mask = self.Intersects.to(bool).view(-1)         
            shift = self.Depths[mask]
            intSur = self.Coordinates[mask]
            surf = intSur[:,:3]+intSur[:,3:]*shift
            look = torch.tensor(v3_utils.sphere_surface_sampler(len(surf)))
            shift = torch.rand(len(surf), 1)*0.1
            intSur[:,:3] = surf+look*shift
            intSur[:,3:] = -look
            intSurInt = torch.ones(len(surf), 1)
            intSurDepth = shift #torch.ones(len(surf), 1)*100
            #intSurDepth[shift<0.1] = shift[shift<0.1]
            #print(surf.shape, look.shape, intSur.shape)
            
            
            # augment non-intersecting points from a sphere
            #nonIntSphere = torch.ones(self.Coordinates.shape)
            #nonIntSphere[:,:3] = torch.tensor(v3_utils.sphere_surface_sampler(len(nonIntSphere), 1.1+np.random.rand()*0.25))
            #nonIntSphere[:,3:] = torch.tensor(v3_utils.sample_hemisphere(nonIntSphere[:,:3]))
            #nonIntSphereInt = torch.zeros(len(nonIntSphere), 1)
            #nonIntSphereDepth = torch.ones(len(nonIntSphere), 1)
            

            #print(self.Coordinates.shape, self.Intersects.shape, self.Depths.shape)
            self.Coordinates = torch.vstack([self.Coordinates, outToInCoor, intSur])#, nonIntSphere])
            self.Intersects = torch.vstack([self.Intersects, outToInInt, intSurInt])#, nonIntSphereInt])
            self.Depths = torch.vstack([self.Depths, outToInDepths, intSurDepth])#, nonIntSphereDepth])
            #self.Coordinates = torch.vstack([outToInCoor, intSur])#, nonIntSphere])
            #self.Intersects = torch.vstack([outToInInt, intSurInt])#, nonIntSphereInt])
            #self.Depths = torch.vstack([outToInDepths, intSurDepth])#, nonIntSphereDepth])
            #print(self.Coordinates.shape, self.Intersects.shape, self.Depths.shape)

        #self.Coordinates = torch.vstack([intSur, nonIntSphere])
        #self.Intersects = torch.vstack([intSurInt, nonIntSphereInt])
        #self.Depths = torch.vstack([intSurDepth, nonIntSphereDepth])          
        

    def __getitem__(self, item):
        return self.Coordinates[item], (self.Intersects[item], self.Depths[item])

    def __len__(self):
        return len(self.Depths)
