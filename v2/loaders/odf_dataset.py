import torch.utils.data
import argparse
import zipfile
import glob
import random
import beacon.utils as butils
import trimesh
import pickle
import math
from tqdm import tqdm
import numpy as np

from tk3dv.common import drawing
import tk3dv.nocstools.datastructures as ds
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

from data import MultiDepthDataset
from sampling import sample_uniform_4D, sampling_preset_noise, sample_vertex_4D, sample_tangential_4D
import odf_utils
from single_losses import SingleDepthBCELoss, SINGLE_MASK_THRESH

MESH_DATASET_NAME = 'bunny_dataset'
MESH_DATASET_URL = 'https://neuralodf.s3.us-east-2.amazonaws.com/' + MESH_DATASET_NAME + '.zip'
DEFAULT_VERT_NOISE = 0.02
DEFAULT_TAN_NOISE = 0.02
DEFAULT_UNIFORM_RATIO = 100
DEFAULT_VERT_RATIO = 0
DEFAULT_TAN_RATIO = 0
DEFAULT_RADIUS = 1.25
DEFAULT_MAX_INTERSECT = 1


class ODFDatasetLoader(torch.utils.data.Dataset):
    def __init__(self, root, train=True, download=True, limit=None, mode='mesh', n_samples=1e5, sampling_methods=None, sampling_frequency=None, usePositionalEncoding=True, coord_type='direction'):
        self.FileName = MESH_DATASET_NAME + '.zip'
        self.DataURL = MESH_DATASET_URL
        self.Mode = mode
        self.nSamples = n_samples
        self.PositionalEnc = usePositionalEncoding
        self.CoordType = coord_type # Options: 'points', 'direction', 'pluecker'
        print('[ INFO ]: Loading ODFDatasetLoader dataset. Mode: {}, Positional Encoding: {}, Coordinate Type: {}'.format(self.Mode, self.PositionalEnc, self.CoordType))

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
        PostFixes = '_' + 'samples-' + str(self.nSamples).zfill(2)
        # PostFixes += '_' + 'posenc-' + str(self.PositionalEnc)

        return PostFixes

    def createODFCache(self, BaseDirPath):
        # Create ODF samples and write to cache.
        # Also write parameters info into each cache to enable loading
        for OBJFileName in self.OBJList:
            Mesh = trimesh.load(OBJFileName)
            Faces = Mesh.faces
            Verts = Mesh.vertices
            Verts = odf_utils.mesh_normalize(Verts)

            # Get without positional encoding and do that later
            MeshODF = MultiDepthDataset(Faces, Verts, DEFAULT_RADIUS, self.SamplingMethods, self.SamplingFrequency, size=self.nSamples, intersect_limit=DEFAULT_MAX_INTERSECT, pos_enc=False)
            ODFSamples = []
            assert len(MeshODF) == self.nSamples
            for Idx in tqdm(range(len(MeshODF))):
                DefaultDict = MeshODF[Idx]
                for Rep in ['coordinates_points', 'coordinates_direction', 'coordinates_pluecker']:
                    DefaultDict[Rep+'_posenc'] = torch.tensor([x for val in list(torch.squeeze(DefaultDict[Rep])) for x in odf_utils.positional_encoding(val.item())], dtype=torch.float32)
                ODFSamples.append(DefaultDict)
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

    def __getitem__(self, idx, PosEnc=None):
        OBJFileIdx = math.floor(idx / self.nSamples)
        RayIdx = idx % self.nSamples
        # print('\nOBJFileIdx, RayIdx:', OBJFileIdx, RayIdx)

        if self.CurrentODFCacheFile != self.ODFCacheList[OBJFileIdx]:
            print('[ INFO ]: Swapping ODF cache to {}. Please wait.'.format(os.path.basename(self.ODFCacheList[OBJFileIdx])))
            self.CurrentODFCacheFile = self.ODFCacheList[OBJFileIdx]
            self.CurrentODFCacheSamples = self.loadODFCache(self.ODFCacheList[OBJFileIdx])
        assert len(self.CurrentODFCacheSamples) == self.nSamples
        # print(ODFSamples[RayIdx])

        if PosEnc is None:
            if self.PositionalEnc:
                Coordinates = self.CurrentODFCacheSamples[RayIdx]['coordinates_' + self.CoordType + '_posenc']
            else:
                Coordinates = self.CurrentODFCacheSamples[RayIdx]['coordinates_' + self.CoordType]
        else:
            if PosEnc:
                Coordinates = self.CurrentODFCacheSamples[RayIdx]['coordinates_' + self.CoordType + '_posenc']
            else:
                Coordinates = self.CurrentODFCacheSamples[RayIdx]['coordinates_' + self.CoordType]
        Intersects = self.CurrentODFCacheSamples[RayIdx]['intersect']
        Depths = self.CurrentODFCacheSamples[RayIdx]['depths']

        return Coordinates, (Intersects, Depths)

class ODFDatasetVisualizer(EaselModule):
    def __init__(self, Data=None, Offset=[0, 0, 0], DataLimit=10000):
        super().__init__()
        self.isVBOBound = False
        self.showSphere = False
        self.RayLength = 0.1
        self.Offset = Offset
        self.DataLimit = DataLimit # This is number of rays

        self.ODFData = Data
        self.CoordType = self.ODFData.CoordType
        if self.CoordType == 'pluecker':
            self.nCoords = 120
        else:
            self.nCoords = 6


    def init(self, argv=None):
        self.update()
        self.updateVBOs()

    def update(self):
        self.Rays = np.empty((0, self.nCoords), np.float64)
        self.Depths = np.empty((0, 1), np.float64)
        self.ShapePoints = np.empty((0, 3), np.float64)
        self.RayPoints = np.empty((0, 3), np.float64)
        print('[ INFO ]: Loading ODF data for visualization.')
        Limit = self.DataLimit if self.DataLimit < len(self.ODFData) else len(self.ODFData)
        print('[ INFO ]: Limiting visualization to first {} rays.'.format(Limit))
        for Idx in tqdm(range(Limit)):
            ODFRay = self.ODFData.__getitem__(Idx, PosEnc=False) # Pose encoding must always be false for visualization
            if ODFRay[1][0] > 0:
                Ray = np.squeeze(ODFRay[0].numpy())
                Depth = np.squeeze(ODFRay[1][1].numpy())
                self.Rays = np.vstack((self.Rays, Ray))
                self.Depths = np.vstack((self.Depths, Depth))
                if self.CoordType == 'points':
                    Direction = (Ray[3:] - Ray[:3])
                    Norm = np.linalg.norm(Direction)
                    if Norm == 0.0:
                        continue
                    Direction /= Norm
                elif self.CoordType == 'direction':
                    Direction = Ray[3:]
                ShapePoint = np.array(Ray[:3] + (Direction * Depth))
                self.ShapePoints = np.vstack((self.ShapePoints, ShapePoint))
                self.RayPoints = np.vstack((self.RayPoints, ShapePoint))
                self.RayPoints = np.vstack((self.RayPoints, ShapePoint - self.RayLength * Direction)) # Unit direction point, updated in VBO update

        print('[ INFO ]: Found {} intersecting rays.'.format(len(self.Rays)))

    def updateVBOs(self):
        # VBOs
        self.nPoints = self.ShapePoints.shape[0]
        self.nRayPoints = self.RayPoints.shape[0]
        if self.nPoints == 0:
            return

        Direction = self.RayPoints[0::2, :] - self.RayPoints[1::2, :]
        Norm = np.expand_dims(np.linalg.norm(Direction, axis=1), axis=1)
        # print(Norm.shape)
        Direction /= np.repeat(Norm, 3, axis=1)
        # print(Direction.shape)
        self.RayPoints[1::2, :] = (self.RayPoints[0::2, :] - self.RayLength * Direction)
        self.VBOPoints = glvbo.VBO(self.ShapePoints)
        self.VBORayPoints = glvbo.VBO(self.RayPoints)
        self.isVBOBound = True

    def step(self):
        pass

    def draw(self):
        if self.CoordType == 'pluecker':
            print('[ WARN ]: Visualization of pluecker coordinates and positional encodings not yet implemented.')
            return
        if self.isVBOBound == False:
            print('[ WARN ]: VBOs not bound. Call update().')
            return

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()

        ScaleFact = 200
        gl.glScale(ScaleFact, ScaleFact, ScaleFact)
        gl.glTranslate(self.Offset[0], self.Offset[1], self.Offset[2])

        if self.showSphere:
            gl.glPushMatrix()
            gl.glRotatef(90, 1, 0, 0)
            drawing.drawWireSphere(DEFAULT_RADIUS, 32, 32)
            gl.glPopMatrix()

        gl.glPushAttrib(gl.GL_POINT_BIT)

        gl.glPointSize(10)
        gl.glColor3f(0, 0, 1)
        if self.VBOPoints is not None:
            self.VBOPoints.bind()
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3, gl.GL_DOUBLE, 0, self.VBOPoints)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.nPoints)

        gl.glColor3f(1, 0, 0)
        if self.VBORayPoints is not None:
            self.VBORayPoints.bind()
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3, gl.GL_DOUBLE, 0, self.VBORayPoints)
        gl.glDrawArrays(gl.GL_LINES, 0, self.nRayPoints)

        gl.glPopAttrib()

        gl.glPopMatrix()

    def keyPressEvent(self, a0: QKeyEvent):
        if a0.key() == QtCore.Qt.Key_Plus:
            if self.RayLength < 0.9:
                self.RayLength += 0.05
                self.updateVBOs()
            print('[ INFO ]: Updated ray length: ', self.RayLength, flush=True)

        if a0.key() == QtCore.Qt.Key_Minus:
            if self.RayLength > 0.1:
                self.RayLength -= 0.05
                self.updateVBOs()
            print('[ INFO ]: Updated ray length: ', self.RayLength, flush=True)

        if a0.key() == QtCore.Qt.Key_S:
            self.showSphere = not self.showSphere

class ODFDatasetLiveVisualizer(ODFDatasetVisualizer):
    def __init__(self, coord_type, rays, intersects, depths, Offset=[0, 0, 0], DataLimit=10000):
        self.CoordType = coord_type
        self.Offset = Offset
        self.DataLimit = DataLimit  # This is number of rays
        if self.CoordType == 'pluecker':
            self.nCoords = 120
        else:
            self.nCoords = 6

        self.Rays = rays
        self.Intersects = intersects
        self.Depths = depths

        self.isVBOBound = False
        self.showSphere = False
        self.RayLength = 0.1

    def init(self, argv=None):
        self.update()
        self.updateVBOs()

    def update(self):
        self.ShapePoints = np.empty((0, 3), np.float64)
        self.RayPoints = np.empty((0, 3), np.float64)
        nValidRays = 0
        print('[ INFO ]: Updating live data for visualization.')
        Limit = self.DataLimit if self.DataLimit < len(self.Rays) else len(self.Rays)
        print('[ INFO ]: Limiting visualization to first {} rays.'.format(Limit))
        for Idx in tqdm(range(Limit)):
            R = self.Rays[Idx]
            I = self.Intersects[Idx]
            D = self.Depths[Idx]
            isIntersect = torch.sigmoid(I) > SINGLE_MASK_THRESH
            if isIntersect:
                nValidRays += 1
                Ray = np.squeeze(R.numpy())
                Depth = np.squeeze(D.numpy())
                if self.CoordType == 'points':
                    Direction = (Ray[3:] - Ray[:3])
                    Norm = np.linalg.norm(Direction)
                    if Norm == 0.0:
                        continue
                    Direction /= Norm
                elif self.CoordType == 'direction':
                    Direction = Ray[3:]
                ShapePoint = np.array(Ray[:3] + (Direction * Depth))
                self.ShapePoints = np.vstack((self.ShapePoints, ShapePoint))
                self.RayPoints = np.vstack((self.RayPoints, ShapePoint))
                self.RayPoints = np.vstack((self.RayPoints, ShapePoint - self.RayLength * Direction))

        print('[ INFO ]: Found {} intersecting rays.'.format(nValidRays))

    def updateVBOs(self):
        # VBOs
        self.nPoints = self.ShapePoints.shape[0]
        self.nRayPoints = self.RayPoints.shape[0]
        if self.nPoints == 0:
            return

        Direction = self.RayPoints[0::2, :] - self.RayPoints[1::2, :]
        Norm = np.expand_dims(np.linalg.norm(Direction, axis=1), axis=1)
        # print(Norm.shape)
        Direction /= np.repeat(Norm, 3, axis=1)
        # print(Direction.shape)
        self.RayPoints[1::2, :] = (self.RayPoints[0::2, :] - self.RayLength * Direction)
        self.VBOPoints = glvbo.VBO(self.ShapePoints)
        self.VBORayPoints = glvbo.VBO(self.RayPoints)
        self.isVBOBound = True


Parser = argparse.ArgumentParser()
Parser.add_argument('-d', '--data-dir', help='Specify the location of the directory to download and store HerthaSim', required=True)
Parser.add_argument('-m', '--mode', help='Specify the dataset mode.', required=False, choices=['mesh'], default='mesh')
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('-n', '--nsamples', help='How many rays of ODF to sample per shape.', required=False, type=int, default=100000)
Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
Parser.add_argument('--no-posenc', help='Choose not to use positional encoding.', action='store_true', required=False)
Parser.set_defaults(no_posenc=False)
Parser.add_argument('-v', '--viz-limit', help='Limit visualizations to these many rays.', required=False, type=int, default=1000)


if __name__ == '__main__':
    Args = Parser.parse_args()
    butils.seedRandom(Args.seed)
    usePoseEnc = not Args.no_posenc

    Data = ODFDatasetLoader(root=Args.data_dir, train=True, download=True, mode=Args.mode, n_samples=Args.nsamples, usePositionalEncoding=usePoseEnc, coord_type=Args.coord_type)
    # print(Data[650])
    # Data[65038]
    # Data.visualizeRandom()

    # Loss = SingleDepthBCELoss()
    # N = 0
    # for i in range(len(Data)):
    #     if Data[i][1][0][0] > 0:
    #         N = i
    #         break
    # target = (Data[N][1][0].unsqueeze(1), Data[N][1][1].unsqueeze(1))
    # output = (torch.tensor(0.9), Data[N][1][1].unsqueeze(1))
    # print(target)
    # print(output)
    # print(Loss(output, target))

    app = QApplication(sys.argv)

    if len(Data) < Args.viz_limit:
        Args.viz_limit = len(Data)
    # mainWindow = Easel([ODFDatasetVisualizer(Data[:int(Args.viz_limit * Args.nsamples)])], sys.argv[1:])
    mainWindow = Easel([ODFDatasetVisualizer(Data, DataLimit=Args.viz_limit)], sys.argv[1:])
    mainWindow.show()
    sys.exit(app.exec_())
