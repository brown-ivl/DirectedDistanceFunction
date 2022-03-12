import argparse
import trimesh
import torch
import math
from tqdm import tqdm
import multiprocessing as mp
from threading import Thread

from PyQt5.QtWidgets import QApplication
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

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
    def __init__(self, NPData, TargetRays, UsePosEnc=False, NearSurfacePointsOffset=-1., NearSurfaceFraction=0.1, TangentRaysRatio=0.0, AdditionalIntersections=0):
        self.NPData = NPData
        self.nTargetRays = TargetRays
        self.UsePosEnc = UsePosEnc
        # print('[ INFO ]: Found {} vertices with normals. Will try to sample {} rays in total.'.format(len(self.Vertices), self.nTargetRays))
        self.NearSurfacePointsOffset = NearSurfacePointsOffset
        self.TangentRaysRatio=TangentRaysRatio
        self.AdditionalIntersections=AdditionalIntersections
        self.NearSurfaceFraction = NearSurfaceFraction

        self.Coordinates = None
        self.Intersects = None
        self.Depths = None
        self.ValidDepthMask = None
        

        self.Interior = np.min(NPData["depth_map"]) < 0.

        self.sample(self.nTargetRays)

    def additional_positive_intersections(self, unprojected_points, multiplier=3, max_perturbation=0.4):
        base_points = np.concatenate([unprojected_points,]*multiplier, axis=0)
        perturbations = v3_utils.sphere_interior_sampler(base_points.shape[0], radius=max_perturbation)
        view_directions = perturbations * -1 / np.linalg.norm(perturbations, axis=-1, keepdims=True)
        perturbed_points = base_points + perturbations
        valid_points = np.linalg.norm(perturbed_points, axis=-1) < DEPTH_SAMPLER_RADIUS
        perturbed_points = perturbed_points[valid_points]
        view_directions = view_directions[valid_points]
        coordinates = np.concatenate([perturbed_points, view_directions], axis=-1)
        intersections = np.ones((perturbed_points.shape[0],))
        placeholder_depths = np.ones((perturbed_points.shape[0],))*0.1
        valid_depths_mask = np.zeros((perturbed_points.shape[0],))
        return coordinates, intersections, placeholder_depths, valid_depths_mask

    def additional_near_surface_points(self, end_points, start_points, max_offset=0.01):
        selection_fraction = self.NearSurfaceFraction
        n_samples = int(end_points.shape[0]*selection_fraction)
        selection_indices = np.zeros((end_points.shape[0],), dtype=bool)
        selection_indices[:n_samples] = 1
        np.random.shuffle(selection_indices)
        end_points = end_points[selection_indices]
        start_points = start_points[selection_indices]

        view_dirs = end_points - start_points
        depths = np.linalg.norm(view_dirs, axis=-1)
        view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=-1, keepdims=True)
        offsets = np.random.uniform(low=-max_offset, high=max_offset, size=depths.shape)
        new_positions = start_points + np.hstack([(depths+offsets)[:, None],]*3) * view_dirs
        new_depths = -1. * offsets
        new_coordinates = np.concatenate([new_positions, view_dirs], axis=-1)
        new_intersections = np.ones(new_depths.shape)
        new_valid_depths_mask = np.ones(new_depths.shape)
        return new_coordinates, new_intersections, new_depths, new_valid_depths_mask


    def sample(self, TargetRays, RatioPositive=DEPTH_SAMPLER_POS_RATIO):
        AllEndPoints = self.NPData['unprojected_normalized_pts']        
        StartPoint = self.NPData['viewpoint'] # There is only 1 start point, the camera center
        AllStartPoints = np.tile(StartPoint, (AllEndPoints.shape[0], 1))
        AllIntersects = self.NPData['invalid_depth_mask']

        mask_dim = int(math.sqrt(self.NPData['invalid_depth_mask'].shape[0]))
        depth_mask = np.logical_not(np.reshape(self.NPData['invalid_depth_mask'], (mask_dim, mask_dim)))
        tangent_mask = np.logical_xor(binary_dilation(depth_mask, iterations=3, border_value=0),binary_erosion(depth_mask, iterations=3, border_value=1))
        tangent_mask = tangent_mask.flatten().astype(bool)

        nTanTargetRays = math.floor(TargetRays*self.TangentRaysRatio)
        nPosTargetRays = math.floor((TargetRays-nTanTargetRays)*RatioPositive)
        nNegTargetRays = (TargetRays-nPosTargetRays-nTanTargetRays)

        AllTanIdx = np.where(tangent_mask == True)[0]
        AllPosIdx = np.where(AllIntersects == False)[0]
        AllNegIdx = np.where(AllIntersects == True)[0]
        TanShuffleIdx = np.random.permutation(len(AllTanIdx))
        PosShuffleIdx = np.random.permutation(len(AllPosIdx))
        NegShuffleIdx = np.random.permutation(len(AllNegIdx))


        # Sample Tangent Rays
        SampledTanEndPts = AllEndPoints[AllTanIdx[TanShuffleIdx[:nTanTargetRays]]]
        SampledTanStartPts = AllStartPoints[AllTanIdx[TanShuffleIdx[:nTanTargetRays]]]
        SampledTanDir = SampledTanEndPts - SampledTanStartPts
        SampledTanDirNorm = np.linalg.norm(SampledTanDir, axis=1)
        SampledTanDir /= SampledTanDirNorm[:, np.newaxis]
        SampledTanDepths = np.linalg.norm(SampledTanEndPts - SampledTanStartPts, axis=1)
        SampledTanIntersects = np.logical_not(AllIntersects[AllTanIdx[TanShuffleIdx[:nTanTargetRays]]])

        # Sample Intersecting Rays
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


        # Sample Non-Intersecting Rays
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

        Coordinates = np.vstack((np.hstack((SampledTanStartPts, SampledTanDir)), np.hstack((SampledPosStartPts, SampledPosDir)), np.hstack((SampledNegStartPts, SampledNegDir))))
        Intersects = np.vstack((SampledTanIntersects, SampledPosIntersects, SampledNegIntersects))
        Depths = np.expand_dims(np.concatenate((SampledTanDepths, SampledPosDepths, SampledNegDepths)), axis=1)

        ShuffleIdx = np.random.permutation(len(Coordinates))

        self.Coordinates = torch.from_numpy(Coordinates[ShuffleIdx]).to(torch.float32)
        self.Intersects = torch.from_numpy(Intersects[ShuffleIdx]).to(torch.float32)
        self.Depths = torch.from_numpy(Depths[ShuffleIdx]).to(torch.float32)
        self.ValidDepthMask = torch.from_numpy(Intersects[ShuffleIdx]).to(torch.float32)

        if self.AdditionalIntersections > 0:
            coordinates, intersections, placeholder_depths, valid_depths = self.additional_positive_intersections(SampledPosEndPts, multiplier=self.AdditionalIntersections)
            self.Coordinates = torch.cat([self.Coordinates, torch.from_numpy(coordinates).to(torch.float32)], dim=0)
            self.Intersects = torch.cat([self.Intersects, torch.from_numpy(intersections[:,None]).to(torch.float32)], dim=0)
            self.Depths = torch.cat([self.Depths, torch.from_numpy(placeholder_depths[:,None]).to(torch.float32)], dim=0)
            self.ValidDepthMask = torch.cat([self.ValidDepthMask, torch.from_numpy(valid_depths[:,None]).to(torch.float32)], dim=0)

        if self.NearSurfacePointsOffset > 0.:
            coordinates, intersections, depths, valid_depths = self.additional_near_surface_points(SampledPosEndPts, SampledPosStartPts, max_offset=self.NearSurfacePointsOffset)
            self.Coordinates = torch.cat([self.Coordinates, torch.from_numpy(coordinates).to(torch.float32)], dim=0)
            self.Intersects = torch.cat([self.Intersects, torch.from_numpy(intersections[:,None]).to(torch.float32)], dim=0)
            self.Depths = torch.cat([self.Depths, torch.from_numpy(depths[:,None]).to(torch.float32)], dim=0)
            self.ValidDepthMask = torch.cat([self.ValidDepthMask, torch.from_numpy(valid_depths[:,None]).to(torch.float32)], dim=0)

    def __getitem__(self, item):
        return self.Coordinates[item], (self.Intersects[item], self.Depths[item], self.ValidDepthMask[item])

    def __len__(self):
        return len(self.Depths)
