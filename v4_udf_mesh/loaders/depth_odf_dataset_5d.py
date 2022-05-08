import torch.utils.data
import argparse
import zipfile
import glob
import random
import sys
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

# from odf_dataset import ODFDatasetLiveVisualizer
#from depth_sampler_5d import DepthMapSampler
from DepthData import DepthData
import v3_utils

# DEPTH_DATASET_NAME = 'torus'
#DEPTH_DATASET_URL = 'TDB'# 'https://neuralodf.s3.us-east-2.amazonaws.com/' + DEPTH_DATASET_NAME + '.zip'

class DepthODFDatasetLoader(torch.utils.data.Dataset):
    def __init__(self, root, name, train=True, target_samples=1e3, usePositionalEncoding=True, coord_type='direction', ad=False, sampling_frequency=[1.0, 0.0, 0.0, 0.0], vert_noise=0.00001, tan_noise=0.00001, radius=1.25):
        self.FileName = name
        self.nTargetSamples = target_samples
        self.PositionalEnc = usePositionalEncoding
        self.CoordType = coord_type # Options: 'points', 'direction', 'pluecker'
        self.ad = ad #autodecoder
        print('[ INFO ]: Loading {} dataset. Positional Encoding: {}, Coordinate Type: {}, Autodecoder: {}'.format(self.__class__.__name__, self.PositionalEnc, self.CoordType, self.ad))
        self.radius = radius
        self.samplingFreq = sampling_frequency
        self.sampling = [v3_utils.sample_uniform_ray_space, 
                         v3_utils.sampling_preset_noise(v3_utils.sample_vertex_noise, 
                                                        vert_noise),
                         v3_utils.sampling_preset_noise(v3_utils.sample_vertex_all_directions, 
                                                        vert_noise),
                         v3_utils.sampling_preset_noise(v3_utils.sample_vertex_tangential, 
                                                        tan_noise)]
        self.init(root, train)
        self.loadData()

    def init(self, root, train=True):
        self.DataDir = root
        self.isTrainData = train

    #@staticmethod
    #def collate_fn(batch):
    #    data = [item[0] for item in batch]
    #    target = [item[1] for item in batch]
    #    return (data, target)

    @staticmethod
    def collate_fn(batch):
        #print(batch)
        data = [torch.vstack([item[0] for item in batch])]
        target = [(torch.vstack([item[1][0] for item in batch]), torch.vstack([item[1][1] for item in batch]))]
        print(data, target)
        return (data, target)


    def loadData(self):
        DatasetDir = os.path.join(v3_utils.expandTilde(self.DataDir), self.FileName)
        print(DatasetDir)
        #self.mesh_vertices, self.mesh_faces, _ = v3_utils.load_object_shapenet(os.path.join(v3_utils.expandTilde(self.DataDir), self.FileName))
        self.mesh_vertices, self.mesh_faces, _ = v3_utils.load_object(self.FileName, v3_utils.expandTilde(self.DataDir))

        mesh = trimesh.load(os.path.join(v3_utils.expandTilde(self.DataDir), self.FileName+'.obj'))
        self.mesh_faces = mesh.faces
        self.mesh_vertices = mesh.vertices
        self.mesh_vertices = v3_utils.mesh_normalize(self.mesh_vertices)


    def __len__(self):
        return 2 #self.nTargetSamples

    def __getitem__(self, idx, PosEnc=None):
        data = DepthData(self.mesh_faces, 
                         self.mesh_vertices,
                         self.radius,
                         self.sampling,
                         self.samplingFreq,
                         size=self.nTargetSamples)
        return data[0], (data[1], data[2])


Parser = argparse.ArgumentParser()
Parser.add_argument('-d', '--data-dir', help='Specify the location of the directory to download and store dataset.', required=True)
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('-n', '--nsamples', help='How many rays of ODF to aim to sample per image.', required=False, type=int, default=1000)
Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
Parser.add_argument('--no-posenc', help='Choose not to use positional encoding.', action='store_true', required=False)
Parser.set_defaults(no_posenc=False)
Parser.add_argument('-v', '--viz-limit', help='Limit visualizations to these many rays.', required=False, type=int, default=1000)


# if __name__ == '__main__':
#     Args = Parser.parse_args()
#     butils.seedRandom(Args.seed)
#     usePoseEnc = not Args.no_posenc

#     Data = DepthODFDatasetLoader(root=Args.data_dir, train=True, download=True, target_samples=Args.nsamples, usePositionalEncoding=usePoseEnc, coord_type=Args.coord_type)

#     ODFVizList = []
#     for i in range(10):
#         LoadedData = Data[i]
#         ODFVizList.append(ODFDatasetLiveVisualizer(coord_type='direction', rays=LoadedData[0].cpu(),
#                                   intersects=LoadedData[1][0].cpu(), depths=LoadedData[1][1].cpu(),
#                                   DataLimit=Args.viz_limit))

#     app = QApplication(sys.argv)

#     mainWindow = Easel(ODFVizList, sys.argv[1:])
#     mainWindow.show()
#     sys.exit(app.exec_())
