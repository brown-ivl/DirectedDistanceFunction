import torch.utils.data
import argparse
import zipfile
import glob
import random
import sys
import trimesh
import os
from tqdm import tqdm


import numpy as np


FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, '../'))
sys.path.append(os.path.join(FileDirPath, '../losses'))
sys.path.append(os.path.join(FileDirPath, '../../'))

# from odf_dataset import ODFDatasetLiveVisualizer
from depth_sampler_5d import PositiveSampler
import v5_utils

# DEPTH_DATASET_NAME = 'torus'
DEPTH_DATASET_URL = 'TDB'# 'https://neuralodf.s3.us-east-2.amazonaws.com/' + DEPTH_DATASET_NAME + '.zip'

class OccNetLoader(torch.utils.data.Dataset):
    def __init__(self, root, name, train=True, download=True, target_samples=1e3, length=1000):
        self.FileName = name + '.obj'
        self.nTargetSamples = target_samples # Per image
        self.length= length if train else int(length*0.1)
        print('[ INFO ]: Loading {} dataset.'.format(self.__class__.__name__))

        self.init(root, train, download)
        self.loadData()

    def init(self, root, train=True, download=True):
        self.DataDir = root
        self.isTrainData = train
        self.isDownload = download


    @staticmethod
    def collate_fn(batch):
        coords = [item[0] for item in batch]
        depths = [item[1] for item in batch]
        mask_points = [item[2] for item in batch]
        mask_labels = [item[3] for item in batch]
        return (coords, depths, mask_points, mask_labels)

    def genData(self):
        points = v5_utils.sphere_interior_sampler(self.nTargetSamples, radius=1.25)
        occupancies = self.mesh.contains(points)
        points = points.reshape((points.shape[0], points.shape[1], 1))
        occupancies = occupancies.reshape((occupancies.shape[0], 1))
        return points, occupancies
        

    def loadData(self):
        suffix = "_train" if self.isTrainData else "_val"
        DataDumpDir = os.path.join(v5_utils.expandTilde(self.DataDir), os.path.splitext(self.FileName)[0]+suffix)
        if not os.path.exists(DataDumpDir):
            print("Dumped data not found, generating data now.")
            
            # Load Mesh
            OBJFile = os.path.join(v5_utils.expandTilde(self.DataDir), self.FileName)
            if not os.path.exists(OBJFile):
                print("Could not find the requested .obj file")

            self.mesh = trimesh.load(OBJFile)

            points = []
            occupancies = []
            for i in tqdm(range(self.length)):
                p, o = self.genData()
                points.append(p[None,...])
                occupancies.append(o[None,...])
            points = np.concatenate(points)
            occupancies = np.concatenate(occupancies)
            occupancy_data = {
                "points": points,
                "occupancies": occupancies,
            }
            self.occupancy_data = occupancy_data
            os.mkdir(DataDumpDir)
            np.save(os.path.join(DataDumpDir, "occupancy_data.npy"), occupancy_data)
        else:
            print("Loading...")
            self.occupancy_data = np.load(os.path.join(DataDumpDir, "occupancy_data.npy"), allow_pickle=True).item()

    def __len__(self):
        return self.length

    def __getitem__(self, idx, PosEnc=None):
        # points = v5_utils.sphere_interior_sampler(self.nTargetSamples, radius=1.25)
        # occupancies = self.mesh.contains(points)
        # points = torch.tensor(points, dtype=torch.float32)
        # occupancies = torch.tensor(occupancies, dtype=torch.float32)
        # points = points.reshape((points.shape[0], points.shape[1], 1))
        # occupancies = occupancies.reshape((occupancies.shape[0], 1))
        points = self.occupancy_data["points"][idx]
        occupancies = self.occupancy_data["occupancies"][idx]
        points = torch.tensor(points, dtype=torch.float32)
        occupancies = torch.tensor(occupancies, dtype=torch.float32)
        return (None, None, points, occupancies)

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
