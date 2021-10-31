import numpy as np
import math

def sample_directions_numpy(nDirs, normal=None, ndim=3):
    vec = np.random.randn(nDirs, ndim)
    vec /= np.linalg.norm(vec, axis=0)
    if normal is not None:
        # Select only if direction is in the sae half space as normal
        DotP = np.sum(np.multiply(vec, normal), axis=1)
        ValidIdx = DotP > 0.0
        InvalidIdx = DotP <= 0.0
        # vec = vec[ValidIdx]
        vec[InvalidIdx] = normal  # Set invalid to just be normal

    return vec

def sample_directions_prune_normal_numpy(nDirs, vertex, normal, points, thresh):
    Dirs = np.random.randn(nDirs, 3)
    Norm = np.linalg.norm(Dirs, axis=1)
    Dirs = np.divide(Dirs, Norm[:, np.newaxis])

    # Select only if direction is in the same half space as normal
    DotP = np.sum(np.multiply(Dirs, normal), axis=1)
    ValidIdx = DotP > 0.0
    Dirs = Dirs[ValidIdx]

    # Next check if chosen directions are within a threshold of vertices
    d = np.linalg.norm(vertex)
    PlaneEq = np.array([[normal[0], normal[1], normal[2], d]])
    HSVal = np.dot(PlaneEq, np.hstack((points, np.ones((len(points), 1)))).T)
    # print(HSVal.shape)
    # print(np.min(HSVal), np.min(HSVal))
    HSIdx = np.squeeze(HSVal) < 0
    # print(np.sum(HSIdx))
    HalfSpacePoints = points[HSIdx]
    # Point-line distance, ray form: https://www.math.kit.edu/ianm2/lehre/am22016s/media/distance-harvard.pdf
    PQ = vertex - HalfSpacePoints
    P2LDistances = np.linalg.norm(np.abs(np.cross(PQ[:, None, :], Dirs[None, :, :])), axis=2)
    FailedIdx = P2LDistances < thresh  # Boolean array of all possible vertices and ray intersections that failed the threshold test
    # Find directions that failed the test
    FailedDirIdxSum = np.sum(FailedIdx, axis=0)
    SuccessDirIdx = FailedDirIdxSum == 0  # The vertex will be at a distance of 0 so, we look for anything more than 1

    Dirs = Dirs[SuccessDirIdx]

    return Dirs


def sample_directions_prune_numpy(nDirs, vertex, points, thresh):
    Dirs = np.random.randn(nDirs, 3)
    Norm = np.linalg.norm(Dirs, axis=1)
    Dirs = np.divide(Dirs, Norm[:, np.newaxis])

    # Point-line distance, ray form: https://www.math.kit.edu/ianm2/lehre/am22016s/media/distance-harvard.pdf
    PQ = vertex - points
    P2LDistances = np.linalg.norm(np.abs(np.cross(PQ[:, None, :], Dirs[None, :, :])), axis=2)
    FailedIdx = P2LDistances < thresh  # Boolean array of all possible vertices and ray intersections that failed the threshold test
    # Find directions that failed the test
    FailedDirIdxSum = np.sum(FailedIdx, axis=0)
    SuccessDirIdx = FailedDirIdxSum == 0

    Dirs = Dirs[SuccessDirIdx]

    return Dirs

def prune_rays(Start, End, Vertices, thresh):
        ValidIdx = np.ones(len(Start), dtype=bool) * True
        RaysPerVertex = int(len(Start) / len(Vertices))

        for VIdx, p in enumerate(Vertices):
            Mask = np.ones(len(Start), dtype=bool)
            Mask[VIdx * RaysPerVertex:(VIdx+1) * RaysPerVertex] = False
            # Exclude the point itself
            a = Start[Mask]
            b = End[Mask]

            # https://stackoverflow.com/questions/54442057/calculate-the-euclidian-distance-between-an-array-of-points-to-a-line-segment-in/54442561#54442561
            # normalized tangent vector
            d = np.divide(b - a, np.linalg.norm(b - a, axis=0))

            # signed parallel distance components
            # s = np.dot(a - p, d)
            s = np.sum(np.multiply(a - p, d), axis=1)
            # t = np.dot(p - b, d)
            t = np.sum(np.multiply(p - b, d), axis=1)

            # clamped parallel distance
            h = np.maximum.reduce([s, t, np.zeros(len(s))])
            # perpendicular distance component
            c = np.cross(p - a, d)
            Distances = np.hypot(h, np.linalg.norm(c, axis=1))
            # print(np.min(Distances), np.max(Distances))
            ValidIdx[Mask] &= (Distances > thresh)

        return ValidIdx


def find_sphere_points(OriginPoints, SphereCenter, Directions, Radius):
    # Line-Sphere intersection: https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    o = OriginPoints
    u = Directions
    if len(OriginPoints.shape) == 1:
        o = OriginPoints[np.newaxis, :]
        u = Directions[np.newaxis, :]
    c = SphereCenter
    OminusC = o - c
    DotP = np.sum(np.multiply(u, OminusC), axis=1)
    Delta = np.square(DotP) - ( (np.linalg.norm(OminusC, axis=1) ** 2) - (Radius ** 2) )
    d = - DotP + np.sqrt(Delta)
    SpherePoints = o + np.multiply(u, d[:, np.newaxis])

    if len(OriginPoints.shape) == 1:
        SpherePoints = np.squeeze(SpherePoints)
        d = np.squeeze(d)

    return SpherePoints, d
