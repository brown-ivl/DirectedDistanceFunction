import torch.utils.data
import argparse
import zipfile
import glob
import random
import beacon.utils as butils
import trimesh
from tqdm import tqdm

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
sys.path.append(os.path.join(FileDirPath, '../losses'))
sys.path.append(os.path.join(FileDirPath, '../../'))

import odf_utils
import odf_v2_utils as o2utils
from odf_dataset import ODFDatasetLiveVisualizer
from pc_sampler import PointCloudSampler

# PC_DATASET_NAME = 'bunny_dataset'
PC_DATASET_NAME = 'bunny_100_dataset'
PC_DATASET_URL = 'https://neuralodf.s3.us-east-2.amazonaws.com/' + PC_DATASET_NAME + '.zip'

class PCODFDatasetLoader(torch.utils.data.Dataset):
    def __init__(self, root, train=True, download=True, limit=None, target_samples=1e3, usePositionalEncoding=True, coord_type='direction'):
        self.FileName = PC_DATASET_NAME + '.zip'
        self.DataURL = PC_DATASET_URL
        self.nTargetSamples = target_samples # Per shape
        self.PositionalEnc = usePositionalEncoding
        assert self.PositionalEnc == False
        self.Sampler = None
        self.CoordType = coord_type # Options: 'points', 'direction', 'pluecker'
        print('[ INFO ]: Loading {} dataset. Positional Encoding: {}, Coordinate Type: {}'.format(self.__class__.__name__, self.PositionalEnc, self.CoordType))

        self.init(root, train, download, limit)
        self.loadData()

    def init(self, root, train=True, download=True, limit=None):
        self.DataDir = root
        self.isTrainData = train
        self.isDownload = download
        self.DataLimit = limit

    @staticmethod
    def collate_fn(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        return (data, target)

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

        self.BaseDirPath = FilesPath

        self.OBJList = (glob.glob(FilesPath + '*.obj')) # Only OBJ supported
        self.OBJList.sort()

        if len(self.OBJList) == 0 or self.OBJList is None:
            raise RuntimeError('[ ERR ]: No files found during data loading.')

        if self.DataLimit is None:
            self.DataLimit = len(self.OBJList)
        DatasetLength = self.DataLimit if self.DataLimit < len(self.OBJList) else len(self.OBJList)
        self.OBJList = self.OBJList[:DatasetLength]

    def __len__(self):
        return (len(self.OBJList))

    def __getitem__(self, idx, PosEnc=None):
        Mesh = trimesh.load(self.OBJList[idx])
        Verts = Mesh.vertices
        Verts = odf_utils.mesh_normalize(Verts)
        VertNormals = Mesh.vertex_normals.copy()
        Norm = np.linalg.norm(VertNormals, axis=1)
        VertNormals /= Norm[:, None]

        # if self.Sampler is None: # todo: TEMP for testing with same samples
        self.Sampler = PointCloudSampler(Verts, VertNormals, TargetRays=self.nTargetSamples)

        return self.Sampler.Coordinates, (self.Sampler.Intersects, self.Sampler.Depths)


Parser = argparse.ArgumentParser()
Parser.add_argument('-d', '--data-dir', help='Specify the location of the directory to download and store dataset.', required=True)
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('-n', '--nsamples', help='How many rays of ODF to aim to sample per shape.', required=False, type=int, default=1000)
Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
Parser.add_argument('--no-posenc', help='Choose not to use positional encoding.', action='store_true', required=False)
Parser.set_defaults(no_posenc=False)
Parser.add_argument('-v', '--viz-limit', help='Limit visualizations to these many rays.', required=False, type=int, default=1000)


if __name__ == '__main__':
    Args = Parser.parse_args()
    butils.seedRandom(Args.seed)
    usePoseEnc = not Args.no_posenc

    Data = PCODFDatasetLoader(root=Args.data_dir, train=True, download=True, target_samples=Args.nsamples, usePositionalEncoding=usePoseEnc, coord_type=Args.coord_type)
    LoadedData = Data[0]

    app = QApplication(sys.argv)

    mainWindow = Easel([ODFDatasetLiveVisualizer(coord_type='direction', rays=LoadedData[0].cpu(),
                                                 intersects=LoadedData[1][0].cpu(), depths=LoadedData[1][1].cpu(),
                                                 DataLimit=Args.viz_limit)], sys.argv[1:])
    mainWindow.show()
    sys.exit(app.exec_())
