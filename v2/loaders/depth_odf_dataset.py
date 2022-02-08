import torch.utils.data
import argparse
import zipfile
import glob
import random
import sys
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
from depth_sampler import DepthMapSampler

DEPTH_DATASET_NAME = 'torus'
DEPTH_DATASET_URL = 'TDB'# 'https://neuralodf.s3.us-east-2.amazonaws.com/' + DEPTH_DATASET_NAME + '.zip'

class DepthODFDatasetLoader(torch.utils.data.Dataset):
    def __init__(self, root, train=True, download=True, limit=None, target_samples=1e3, usePositionalEncoding=True, coord_type='direction', ad=False):
        self.FileName = DEPTH_DATASET_NAME + '.zip'
        self.DataURL = DEPTH_DATASET_URL
        self.nTargetSamples = target_samples # Per image
        self.PositionalEnc = usePositionalEncoding
        self.Sampler = None
        self.CoordType = coord_type # Options: 'points', 'direction', 'pluecker'
        self.ad = ad #autodecoder
        print('[ INFO ]: Loading {} dataset. Positional Encoding: {}, Coordinate Type: {}, Autodecoder: {}'.format(self.__class__.__name__, self.PositionalEnc, self.CoordType, self.ad))

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

        FilesPath = os.path.join(DatasetDir, 'depth', 'val')
        if self.isTrainData:
            FilesPath = os.path.join(DatasetDir, 'depth', 'train')

        self.BaseDirPath = FilesPath

        self.DepthList = (glob.glob(os.path.join(FilesPath, '*.npy'))) # Only npy supported
        self.DepthList.sort()

        if len(self.DepthList) == 0 or self.DepthList is None:
            raise RuntimeError('[ ERR ]: No depth image files found during data loading.')

        if self.DataLimit is None:
            self.DataLimit = len(self.DepthList)
        DatasetLength = self.DataLimit if self.DataLimit < len(self.DepthList) else len(self.DepthList)
        self.DepthList = self.DepthList[:DatasetLength]

        self.LoadedDepths = []
        for FileName in self.DepthList:
            DepthData = np.load(FileName, allow_pickle=True).item()
            self.LoadedDepths.append(DepthData)

    def __len__(self):
        return (len(self.DepthList))

    def __getitem__(self, idx, PosEnc=None):
        DepthData = self.LoadedDepths[idx]

        self.Sampler = DepthMapSampler(DepthData, TargetRays=self.nTargetSamples, UsePosEnc=self.PositionalEnc)

        #Include latent vector if we are using an AutoDecoder
        if not self.ad:
            return self.Sampler.Coordinates, (self.Sampler.Intersects, self.Sampler.Depths)
        else:
            return (self.Sampler.Coordinates, torch.tensor([idx]*self.Sampler.Coordinates.size()[0])), (self.Sampler.Intersects, self.Sampler.Depths)

Parser = argparse.ArgumentParser()
Parser.add_argument('-d', '--data-dir', help='Specify the location of the directory to download and store dataset.', required=True)
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('-n', '--nsamples', help='How many rays of ODF to aim to sample per image.', required=False, type=int, default=1000)
Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
Parser.add_argument('--no-posenc', help='Choose not to use positional encoding.', action='store_true', required=False)
Parser.set_defaults(no_posenc=False)
Parser.add_argument('-v', '--viz-limit', help='Limit visualizations to these many rays.', required=False, type=int, default=1000)


if __name__ == '__main__':
    Args = Parser.parse_args()
    butils.seedRandom(Args.seed)
    usePoseEnc = not Args.no_posenc

    Data = DepthODFDatasetLoader(root=Args.data_dir, train=True, download=True, target_samples=Args.nsamples, usePositionalEncoding=usePoseEnc, coord_type=Args.coord_type)

    ODFVizList = []
    for i in range(10):
        LoadedData = Data[i]
        ODFVizList.append(ODFDatasetLiveVisualizer(coord_type='direction', rays=LoadedData[0].cpu(),
                                  intersects=LoadedData[1][0].cpu(), depths=LoadedData[1][1].cpu(),
                                  DataLimit=Args.viz_limit))

    app = QApplication(sys.argv)

    mainWindow = Easel(ODFVizList, sys.argv[1:])
    mainWindow.show()
    sys.exit(app.exec_())
