import torch.utils.data
import os
import argparse
import zipfile
import glob
import random
import beacon.utils as butils
import sys
import trimesh
import pickle
import math
from tqdm import tqdm

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, '../'))
sys.path.append(os.path.join(FileDirPath, '../../'))

from data import MultiDepthDataset
from sampling import sample_uniform_4D, sampling_preset_noise, sample_vertex_4D, sample_tangential_4D
import odf_utils

MESH_DATASET_NAME = 'bunny_dataset'
MESH_DATASET_URL = 'TBD' # todo
DEFAULT_VERT_NOISE = 0.02
DEFAULT_TAN_NOISE = 0.02
DEFAULT_UNIFORM_RATIO = 100
DEFAULT_VERT_RATIO = 0
DEFAULT_TAN_RATIO = 0
DEFAULT_RADIUS = 1.25
DEFAULT_MAX_INTERSECT = 1

class ODFDatasetLoader(torch.utils.data.Dataset):
    def __init__(self, root, train=True, download=True, limit=None, mode='mesh', n_samples=1e5, sampling_methods=None, sampling_frequency=None, usePositionalEncoding=True):
        self.FileName = MESH_DATASET_NAME + '.zip'
        self.DataURL = MESH_DATASET_URL
        self.Mode = mode
        self.nSamples = n_samples
        self.PositionalEnc = usePositionalEncoding
        print('[ INFO ]: Loading ODFDatasetLoader dataset in mode:', self.Mode)

        self.init(root, train, download, limit, sampling_methods, sampling_frequency)
        self.loadData()

    def init(self, root, train=True, download=True, limit=None, sampling_methods=None, sampling_frequency=None):
        self.DataDir = root
        self.isTrainData = train
        self.isDownload = download
        self.DataLimit = limit

        self.Images = None
        self.Predictions = None

        self.SamplingMethods = sampling_methods
        self.SamplingFrequency = sampling_frequency
        if sampling_methods is None:
            self.SamplingMethods = [sample_uniform_4D,
                                    sampling_preset_noise(sample_vertex_4D, DEFAULT_VERT_NOISE),
                                    sampling_preset_noise(sample_tangential_4D, DEFAULT_TAN_NOISE)]
        if sampling_frequency is None:
            self.SamplingFrequency = [0.01 * DEFAULT_UNIFORM_RATIO, 0.01 * DEFAULT_VERT_RATIO, 0.01 * DEFAULT_TAN_RATIO]

        self.CurrentODFCacheFile = None
        self.CurrentODFCacheSamples = None

    def loadData(self):
        # First check if unzipped directory exists
        DatasetDir = os.path.join(butils.expandTilde(self.DataDir), os.path.splitext(self.FileName)[0])
        if os.path.exists(DatasetDir) is False:
            DataPath = os.path.join(butils.expandTilde(self.DataDir), self.FileName)
            if os.path.exists(DataPath) is False:
                if self.isDownload:
                    print('[ INFO ]: Downloading', DataPath)
                    butils.downloadFile(self.DataURL, DataPath)

                if os.path.exists(DataPath) is False:  # Not downloaded
                    raise RuntimeError('Specified data path does not exist: ' + DataPath)
            # Unzip
            with zipfile.ZipFile(DataPath, 'r') as File2Unzip:
                print('[ INFO ]: Unzipping.')
                File2Unzip.extractall(butils.expandTilde(self.DataDir))

        FilesPath = os.path.join(DatasetDir, 'val/mesh/')
        if self.isTrainData:
            FilesPath = os.path.join(DatasetDir, 'train/mesh/')

        self.OBJList = (glob.glob(FilesPath + '*.obj')) # Only OBJ supported
        self.OBJList.sort()

        if len(self.OBJList) == 0 or self.OBJList is None:
            raise RuntimeError('[ ERR ]: No files found during data loading.')

        self.ODFCacheList = glob.glob(FilesPath +  '/*' + self.getCachePostFixes() + '*.odf')
        self.ODFCacheList.sort()
        if len(self.ODFCacheList) == 0 or self.ODFCacheList is None:
            print('[ INFO ]: No ODF cache found. Will compute and write out cache.')
            self.createODFCache(FilesPath)

        if self.DataLimit is None:
            self.DataLimit = len(self.OBJList)
        DatasetLength = self.DataLimit if self.DataLimit < len(self.OBJList) else len(self.OBJList)
        self.OBJList = self.OBJList[:DatasetLength]

    def getCachePostFixes(self):
        PostFixes = '_' + 'samples-' + str(self.nSamples).zfill(2) + '_'
        PostFixes += 'posenc-' + str(self.PositionalEnc)

        return PostFixes

    def createODFCache(self, BaseDirPath):
        # Create ODF samples and write to cache.
        # Also write parameters info into each cache to enable loading
        for OBJFileName in self.OBJList:
            Mesh = trimesh.load(OBJFileName)
            Faces = Mesh.faces
            Verts = Mesh.vertices
            Verts = odf_utils.mesh_normalize(Verts)

            MeshODF = MultiDepthDataset(Faces, Verts, DEFAULT_RADIUS, self.SamplingMethods, self.SamplingFrequency, size=self.nSamples, intersect_limit=DEFAULT_MAX_INTERSECT, pos_enc=self.PositionalEnc)
            ODFSamples = []
            assert len(MeshODF) == self.nSamples
            for Idx in tqdm(range(len(MeshODF))):
                ODFSamples.append(MeshODF[Idx])
            SerializeFName = os.path.splitext(os.path.basename(OBJFileName))[0] + self.getCachePostFixes() + '.odf'
            SerializeFPath = os.path.join(BaseDirPath, SerializeFName)
            self.ODFCacheList.append(SerializeFPath)
            with open(SerializeFPath, 'wb') as SerializeFile:
                pickle.dump(ODFSamples, SerializeFile)
            print('[ INFO ]: Dumped {} ODF samples to {}'.format(len(MeshODF), SerializeFName))

    def loadODFCache(self, ODFCacheFilePath):
        # Load ODF cache
        with open(ODFCacheFilePath, 'rb') as DeserializeFile:
            ODFSamples = pickle.load(DeserializeFile)

        return ODFSamples

    def __len__(self):
        return (len(self.OBJList) * self.nSamples) # Number of objects * number of samples. todo: Will this overflow?

    def __getitem__(self, idx):
        OBJFileIdx = idx % len(self.OBJList)
        RayIdx = math.floor(idx / len(self.OBJList))
        # print(OBJFileIdx, RayIdx)

        if self.CurrentODFCacheFile != self.ODFCacheList[OBJFileIdx]:
            print('[ INFO ]: Swapping ODF cache to {}'.format(os.path.basename(self.ODFCacheList[OBJFileIdx])))
            self.CurrentODFCacheFile = self.ODFCacheList[OBJFileIdx]
            self.CurrentODFCacheSamples = self.loadODFCache(self.ODFCacheList[OBJFileIdx])
        assert len(self.CurrentODFCacheSamples) == self.nSamples
        # print(ODFSamples[RayIdx])

        Coordinates = self.CurrentODFCacheSamples[RayIdx]['coordinates_direction']
        Intersects = self.CurrentODFCacheSamples[RayIdx]['intersect']
        Depths = self.CurrentODFCacheSamples[RayIdx]['depths']

        return Coordinates, (Intersects, Depths)

    def visualizeRandom(self, nSamples=100):
        if nSamples >= len(self):
            RandIdx = list(range(0, len(self))) * int(nSamples/len(self))
        else:
            RandIdx = random.sample(range(0, len(self)), nSamples)

        for Idx in range(0, nSamples):
            Data = self[RandIdx[Idx]]


Parser = argparse.ArgumentParser()
Parser.add_argument('-d', '--data-dir', help='Specify the location of the directory to download and store HerthaSim', required=True)
Parser.add_argument('-m', '--mode', help='Specify the dataset mode.', required=False, choices=['mesh'], default='mesh')
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('-n', '--nsamples', help='How many rays of ODF to sample.', required=False, type=int, default=100000)

if __name__ == '__main__':
    Args = Parser.parse_args()
    butils.seedRandom(Args.seed)

    Data = ODFDatasetLoader(root=Args.data_dir, train=True, download=True, mode=Args.mode, n_samples=Args.nsamples)
    Data[650]
    Data[651]
    # Data[65038]
    # Data.visualizeRandom()
