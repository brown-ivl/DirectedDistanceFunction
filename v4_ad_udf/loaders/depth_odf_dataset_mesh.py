from lib2to3.pytree import Base
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
from depth_sampler_5d import DepthMapSampler
import v3_utils

# DEPTH_DATASET_NAME4 'torus'
DEPTH_DATASET_URL = 'TDB'# 'https://neuralodf.s3.us-east-2.amazonaws.com/' + DEPTH_DATASET_NAME + '.zip'
# AD_TRAIN_CUTOFF = 30 #number of instances to use for training the autodecoder. further instances are used as validation
# AD_VAL_CUTOFF = 36
N_INSTANCES_AD = 20
TRAIN_PER_INSTANCE_AD = 32 #20
VAL_PER_INSTANCE_AD = 20


def load_object(object_path):
    obj_file = os.path.join(object_path, "models", "model_normalized.obj")

    obj_mesh = trimesh.load_mesh(obj_file)
    obj_mesh = as_mesh(obj_mesh)
    # obj_mesh.show()

    ## deepsdf normalization
    mesh_vertices = obj_mesh.vertices
    mesh_faces = obj_mesh.faces
    center = (mesh_vertices.max(axis=0) + mesh_vertices.min(axis=0))/2.0
    max_dist = np.linalg.norm(mesh_vertices - center, axis=1).max()
    max_dist = max_dist * 1.03
    mesh_vertices = (mesh_vertices - center) / max_dist
    #obj_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
    
    return mesh_vertices, mesh_faces#, obj_mesh


class DepthODFDatasetLoader(torch.utils.data.Dataset):
    def __init__(self, root, name, train=True, download=True, limit=None, target_samples=1e3, usePositionalEncoding=True, coord_type='direction', ad=False, aug=True, instance_index_map=None):
        self.FileName = name + '.zip'
        self.DataURL = DEPTH_DATASET_URL
        self.nTargetSamples = target_samples # Per image
        self.PositionalEnc = usePositionalEncoding
        self.Sampler = None
        self.CoordType = coord_type # Options: 'points', 'direction', 'pluecker'
        self.ad = ad #autodecoder
        self.aug = aug
        print('[ INFO ]: Loading {} dataset. Positional Encoding: {}, Coordinate Type: {}, Autodecoder: {}'.format(self.__class__.__name__, self.PositionalEnc, self.CoordType, self.ad))

        self.DepthList = None
        self.IndicesList = None
        self.instance_index_map = instance_index_map

        self.init(root, train, download, limit)
        self.loadData()

    def init(self, root, train=True, download=True, limit=None):
        self.DataDir = root
        self.isTrainData = train
        self.isDownload = download
        self.DataLimit = limit

    #@staticmethod
    #def collate_fn(batch):
        #data = [item[0] for item in batch]
        #target = [item[1] for item in batch]
        #return (data, target)

    @staticmethod
    def collate_fn(batch):
        data = [(torch.vstack([item[0][0] for item in batch]), torch.hstack([item[0][1] for item in batch]))]
        target = [(torch.vstack([item[1][0] for item in batch]), torch.vstack([item[1][1] for item in batch]))]
        return (data, target)


    def loadData(self):
        if self.ad:
            if self.instance_index_map is None:
                object_dirs = os.listdir(class_dir)
                random.shuffle(object_dirs)
                instance_list = object_dirs[:N_INSTANCES_AD]
                    
                instance_list = object_dirs
                self.n_instances = len(instance_list)
                instance_numbers = range(self.n_instances)
                self.instance_index_map = {instance_list[i] : instance_numbers[i] for i in range(self.n_instances)}
            else:
                self.n_instances = len(self.instance_index_map)
                #print(self.instance_index_map)
            self.MeshList = []
            self.IndicesList = []
            
            for object_name in self.instance_index_map.keys():
                full_object_path = os.path.join(class_dir, object_name)
                mesh_vertices, mesh_faces = load_object(full_object_path)
                index = self.instance_index_map[instance]
                new_indices = [index,]*target_samples
                self.MeshList.append((mesh_vertices, mesh_faces))
                self.IndicesList.append(new_indices)
        else:
            pass
            #FilesPath = os.path.join(DatasetDir, 'depth', 'val')
            #if self.isTrainData:
            #    FilesPath = os.path.join(DatasetDir, 'depth', 'train')

            #self.BaseDirPath = FilesPath
            #self.DepthList = (glob.glob(os.path.join(FilesPath, '*.npy'))) # Only npy supported
            #self.DepthList.sort()

        if len(self.MeshList) == 0 or self.MeshList is None:
            raise RuntimeError('[ ERR ]: No mesh files found during data loading.')


    def __len__(self):
        return (len(self.MeshList))

    def __getitem__(self, idx, PosEnc=None):
        DepthData = self.LoadedDepths[idx]
        self.Sampler = DepthMapSampler(DepthData, TargetRays=self.nTargetSamples, UsePosEnc=self.PositionalEnc, Aug=self.aug)

        #Include latent vector if we are using an AutoDecoder
        if not self.ad:
            return self.Sampler.Coordinates, (self.Sampler.Intersects, self.Sampler.Depths)
        else:
            instance_idx = self.IndicesList[idx] 
            return (self.Sampler.Coordinates, torch.tensor([instance_idx]*self.Sampler.Coordinates.size()[0])), (self.Sampler.Intersects, self.Sampler.Depths)

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
