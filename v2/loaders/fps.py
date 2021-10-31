import argparse
import beacon.utils as butils
import trimesh
import numpy as np
from scipy.spatial import cKDTree, distance_matrix

from PyQt5.QtWidgets import QApplication
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
sys.path.append(os.path.join(FileDirPath, '../../'))
import odf_utils
from odf_dataset import DEFAULT_RADIUS, ODFDatasetVisualizer, ODFDatasetLiveVisualizer
import odf_v2_utils as o2utils

def fps(x, num_points, idx=None):
    nv = x.shape[0]
    # d = distance_matrix(x, x)
    if idx is None:
        idx = np.random.randint(low=0, high=nv - 1)
    elif idx == 'center':
        c = np.mean(x, axis=0, keepdims=True)
        d = distance_matrix(c, x)
        idx = np.argmax(d)

    y = np.zeros(shape=(num_points, 3))
    indices = np.zeros(shape=(num_points,), dtype=np.int32)
    p = x[np.newaxis, idx, ...]
    dist = distance_matrix(p, x)
    for i in range(num_points):
        y[i, ...] = p
        indices[i] = idx
        d = distance_matrix(p, x)
        dist = np.minimum(d, dist)
        idx = np.argmax(dist)
        p = x[np.newaxis, idx, ...]
    return y, indices


class FPSVisualizer(EaselModule):
    def __init__(self, Vertices, VertexNormals, TargetPoints=2048, DataLimit=10000):
        super().__init__()
        self.setup()
        self.Vertices = Vertices
        self.VertexNormals = VertexNormals
        self.nTargetPoints = TargetPoints

    def setup(self):
        self.isVBOBound = False
        self.showSphere = False
        self.PointSize = 5.0
        self.showFPSVertices = True
        self.showVertices = True

    def init(self, argv=None):
        self.update()
        self.updateVBOs()

    def update(self):
        self.FPSSampledVertices, _ = fps(self.Vertices, self.nTargetPoints)
        print(self.FPSSampledVertices.shape)

    def updateVBOs(self):
        # VBOs
        if len(self.Vertices) != 0:
            self.VBOVertices = glvbo.VBO(self.Vertices)
        else:
            self.VBOVertices = None
        if len(self.FPSSampledVertices) != 0:
            self.VBOFPSVertices = glvbo.VBO(self.FPSSampledVertices)
        else:
            self.VBOFPSVertices = None

        self.isVBOBound = True

    def step(self):
        pass

    def draw(self):
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

        if self.showVertices:
            gl.glPushMatrix()
            gl.glTranslate(-1, 0, 0)
            gl.glPointSize(self.PointSize)
            gl.glColor4f(0, 1, 0, 0.6)
            if self.VBOVertices is not None:
                self.VBOVertices.bind()
                gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                gl.glVertexPointer(3, gl.GL_DOUBLE, 0, self.VBOVertices)
            gl.glDrawArrays(gl.GL_POINTS, 0, len(self.Vertices))
            gl.glPopMatrix()
        if self.showFPSVertices:
            gl.glPushMatrix()
            gl.glTranslate(1, 0, 0)
            gl.glPointSize(self.PointSize)
            gl.glColor4f(1, 0, 0, 0.6)
            if self.VBOFPSVertices is not None:
                self.VBOFPSVertices.bind()
                gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                gl.glVertexPointer(3, gl.GL_DOUBLE, 0, self.VBOFPSVertices)
            gl.glDrawArrays(gl.GL_POINTS, 0, len(self.FPSSampledVertices))
            gl.glPopMatrix()

        gl.glPopAttrib()

        gl.glPopMatrix()

    def keyPressEvent(self, a0: QKeyEvent):
        if a0.key() == QtCore.Qt.Key_A:
            if self.PointSize < 50.0:
                self.PointSize += 1.0
            print('[ INFO ]: Updated point size: ', self.PointSize, flush=True)

        if a0.key() == QtCore.Qt.Key_Z:
            if self.PointSize > 1.0:
                self.PointSize -= 1.0
            print('[ INFO ]: Updated point size: ', self.PointSize, flush=True)

        if a0.key() == QtCore.Qt.Key_S:
            self.showSphere = not self.showSphere

        if a0.key() == QtCore.Qt.Key_O:
            self.showVertices = not self.showVertices

        if a0.key() == QtCore.Qt.Key_F:
            self.showFPSVertices = not self.showFPSVertices


Parser = argparse.ArgumentParser()
Parser.add_argument('-i', '--input', help='Specify the input point cloud and normals in OBJ format.', required=True)
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('-v', '--viz-limit', help='Limit visualizations to these many rays.', required=False, type=int, default=1000)
Parser.add_argument('-t', '--target-samples', help='How many points to sample with FPS.', required=False, type=int, default=1024)

if __name__ == '__main__':
    Args = Parser.parse_args()
    butils.seedRandom(Args.seed)

    Mesh = trimesh.load(Args.input)
    Verts = Mesh.vertices
    Verts = odf_utils.mesh_normalize(Verts)

    Verts, Faces = trimesh.remesh.subdivide_to_size(Verts, Mesh.faces, max_edge=0.05)
    Mesh = trimesh.Trimesh(vertices=Verts, faces=Faces, process=False)
    VertNormals = Mesh.vertex_normals.copy()
    Norm = np.linalg.norm(VertNormals, axis=1)
    VertNormals /= Norm[:, None]

    print(len(Verts))
    print(len(VertNormals))
    FPSViz = FPSVisualizer(Verts, VertNormals, TargetPoints=Args.target_samples, DataLimit=Args.viz_limit)

    app = QApplication(sys.argv)

    mainWindow = Easel([FPSViz], sys.argv[1:])
    mainWindow.show()
    sys.exit(app.exec_())

