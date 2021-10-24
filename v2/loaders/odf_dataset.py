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
from single_losses import SingleDepthBCELoss

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

        if self.CoordType == 'direction':
            Coordinates = self.CurrentODFCacheSamples[RayIdx]['coordinates_direction']
        elif self.CoordType == 'points':
            Coordinates = self.CurrentODFCacheSamples[RayIdx]['coordinates_points']
        elif self.CoordType == 'pluecker':
            Coordinates = self.CurrentODFCacheSamples[RayIdx]['coordinates_pluecker']

        Intersects = self.CurrentODFCacheSamples[RayIdx]['intersect']
        Depths = self.CurrentODFCacheSamples[RayIdx]['depths']

        return Coordinates, (Intersects, Depths)


class VizModule(EaselModule):
    def __init__(self, Data):
        super().__init__()
        self.ODFData = Data
        self.CoordType = Data.CoordType
        if self.CoordType == 'pluecker':
            self.nCoords = 120
        else:
            self.nCoords = 6

        self.isVBOBound = False
        self.showSphere = False
        self.RayLength = 0.2

    def init(self, argv=None):
        self.update()

    def update(self):
        self.Rays = np.empty((0, self.nCoords), np.float64)
        self.Depths = np.empty((0, 1), np.float64)
        self.ShapePoints = np.empty((0, 3), np.float64)
        self.RayPoints = np.empty((0, 3), np.float64)
        print('[ INFO ]: Loading ODF data for visualization.')
        for ODFRay in tqdm(self.ODFData):
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
                self.RayPoints = np.vstack((self.RayPoints, ShapePoint - self.RayLength * Direction))

        print('[ INFO ]: Found {} intersecting rays.'.format(len(self.Rays)))

        # VBOs
        self.nPoints = self.ShapePoints.shape[0]
        self.nRayPoints = self.RayPoints.shape[0]
        if self.nPoints == 0:
            return
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
                self.RayLength += 0.1
                self.update()

        if a0.key() == QtCore.Qt.Key_Minus:
            if self.RayLength > 0.1:
                self.RayLength -= 0.1
                self.update()
        # print(self.RayLength, flush=True)

        if a0.key() == QtCore.Qt.Key_S:
            self.showSphere = not self.showSphere


Parser = argparse.ArgumentParser()
Parser.add_argument('-d', '--data-dir', help='Specify the location of the directory to download and store HerthaSim', required=True)
Parser.add_argument('-m', '--mode', help='Specify the dataset mode.', required=False, choices=['mesh'], default='mesh')
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('-n', '--nsamples', help='How many rays of ODF to sample.', required=False, type=int, default=100000)
Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
Parser.add_argument('--no-posenc', help='Choose not to use positional encoding.', action='store_true', required=False)
Parser.set_defaults(no_posenc=False)


if __name__ == '__main__':
    Args = Parser.parse_args()
    butils.seedRandom(Args.seed)
    usePoseEnc = not Args.no_posenc

    Data = ODFDatasetLoader(root=Args.data_dir, train=True, download=True, mode=Args.mode, n_samples=Args.nsamples, usePositionalEncoding=usePoseEnc, coord_type=Args.coord_type)
    # print(Data[650])
    # Data[65038]
    # Data.visualizeRandom()

    Loss = SingleDepthBCELoss()

    target = Data[650][1]
    output = (torch.from_numpy(np.array([[0.7, 0.0]])), Data[650][1][1])

    print(Loss(output, target))

    app = QApplication(sys.argv)

    mainWindow = Easel([VizModule(Data)], sys.argv[1:])
    mainWindow.show()
    sys.exit(app.exec_())
