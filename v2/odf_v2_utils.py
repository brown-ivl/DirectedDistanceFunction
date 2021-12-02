import numpy as np
import math
import os
import torch

def save_latent_vectors(save_directory, experiment_name, latent_vec, epoch):
    latent_codes_dir = os.path.join(save_directory, f"{experiment_name}___latent_vecs")
    filename = f"{experiment_name}_{epoch}"
    if not os.path.exists(latent_codes_dir):
        os.mkdir(latent_codes_dir)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )

def load_latent_vectors(save_directory, experiment_name, lat_vecs):
    latent_codes_dir = os.path.join(save_directory, f"{experiment_name}___latent_vecs")
    filename = os.listdir(latent_codes_dir)[0]
    full_filename = os.path.join(latent_codes_dir, filename)

    data = torch.load(full_filename)

    for i, lat_vec in enumerate(data["latent_codes"]["weight"]):
        print(f"VEC: {i+1}")
        print(lat_vec)
        lat_vecs.weight.data[i, :] = lat_vec

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

def sample_directions_batch_prune(nDirs, vertices, points, thresh):
    nVertices = len(vertices)
    Dirs = np.random.randn(nDirs*nVertices, 3)
    Norm = np.linalg.norm(Dirs, axis=1)
    Dirs = np.divide(Dirs, Norm[:, np.newaxis])
    Dirs = np.reshape(Dirs, (nVertices, nDirs, -1))

    # Point-line distance, ray form: https://www.math.kit.edu/ianm2/lehre/am22016s/media/distance-harvard.pdf
    # PQ = np.reshape(points[:, None, :] - vertices[None, :, :], (-1, 3))
    PQ = vertices[:, None, :] - points[None, :, :]

    SampledDirections = np.zeros((nDirs * nVertices, 3))
    VertexRepeats = np.zeros_like(SampledDirections)
    ValidDirCtr = 0
    for VCtr in range(nVertices):
        P2LDistances = np.linalg.norm(np.abs(np.cross(PQ[VCtr, :, None, :], Dirs[VCtr, None, :, :])), axis=2)
        FailedIdx = P2LDistances < thresh  # Boolean array of all possible vertices and ray intersections that failed the threshold test
        # Find directions that failed the test
        FailedDirIdxSum = np.sum(FailedIdx, axis=0)
        SuccessDirIdx = FailedDirIdxSum == 0
        nSuccessIdx = np.sum(SuccessDirIdx)
        SampledDirections[ValidDirCtr:ValidDirCtr + nSuccessIdx] = Dirs[VCtr, SuccessDirIdx]
        VertexRepeats[ValidDirCtr:ValidDirCtr + nSuccessIdx] = vertices[np.newaxis, VCtr]
        ValidDirCtr += nSuccessIdx

    SampledDirections = SampledDirections[:ValidDirCtr]
    VertexRepeats = VertexRepeats[:ValidDirCtr]

    return SampledDirections, VertexRepeats

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
    Delta[Delta<=0] = 0
    d = - DotP + np.sqrt(Delta)
    SpherePoints = o + np.multiply(u, d[:, np.newaxis])
    SpherePoints[Delta<=0] = np.zeros(3)

    if len(OriginPoints.shape) == 1:
        SpherePoints = np.squeeze(SpherePoints)
        d = np.squeeze(d)

    return SpherePoints, d

def get_positional_enc(in_array, L = 10):
    '''
    val - Array of values (usually Nx6)
    L   - controls the size of the encoding (size = 2*L  - see paper for details)
    Implements the positional encoding described in section 5.1 of NeRF
    https://arxiv.org/pdf/2003.08934.pdf
    '''
    N = in_array.shape[0]
    nCoords = in_array.shape[1]
    out_array = np.zeros((N, nCoords*2*L))
    for Idx in range(nCoords):
        Val = in_array[:, Idx]
        PosEnc = [x for i in range(L) for x in [np.sin(2 ** (i) * math.pi * Val), np.cos(2 ** (i) * math.pi * Val)]]
        out_array[:, Idx*2*L:(Idx+1)*2*L] = np.array(PosEnc).T

    # # Test
    # Row = 73
    # Col = 3
    # Val = in_array[Row, Col]
    # TestOrig = [x for i in range(L) for x in [math.sin(2 ** (i) * math.pi * Val), math.cos(2 ** (i) * math.pi * Val)]]
    # print(TestOrig)
    # print(out_array[Row, Col*2*L:(Col+1)*2*L])

    return out_array

