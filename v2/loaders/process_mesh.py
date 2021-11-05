import trimesh
import argparse
import beacon.utils as butils
import numpy as np
import sys
import os

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
from pc_sampler import PC_SAMPLER_THRESH
from palettable.colorbrewer.qualitative import Dark2_7
from palettable.cmocean.sequential import Thermal_18, Haline_4
from palettable.matplotlib import Inferno_20, Plasma_20
from palettable.mycarta import Cube1_20
# from tk3dv.common.trimesh_visualizer import TrimeshVisualizer

def lightsOn():
    gl.glShadeModel(gl.GL_SMOOTH)
    mat_specular = np.array([1.0, 1.0, 1.0, 1.0])
    mat_shininess = np.array([128])
    light_position = np.array([0.0, 3.0, 0.0, 0.0])
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_position)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, mat_specular)

    gl.glEnable(gl.GL_LIGHTING)
    gl.glEnable(gl.GL_LIGHT0)

def lightsOff():
    gl.glDisable(gl.GL_LIGHT0)
    gl.glDisable(gl.GL_LIGHTING)


class TrimeshVisualizer(object):
    def __init__(self, TrimeshObject, RenderVertexColors=False):
        self.isVBOBound = False
        self.VBOVertices = None
        self.VBOColors = None
        self.VBONormals = None

        self.Trimesh = TrimeshObject
        self.RenderVertexColors = RenderVertexColors
        self.Vertices = self.Trimesh.vertices
        self.Normals = self.Trimesh.face_normals
        self.VertNormals = self.Trimesh.vertex_normals
        self.Faces = self.Trimesh.faces
        self.VertColors = self.Trimesh.visual.vertex_colors
        self.ReplVertices = np.empty((0, 3)) # These are repeated vertices for drawing
        self.ReplVertColors = np.empty((0, 4))  # These are repeated vertex colors for drawing
        self.ReplVertNormals = np.empty((0, 3))  # These are repeated vertex normals for drawing

        if len(self.Faces) > 0:
            # todo: Make this faster using numpy
            for Face in self.Faces:
                self.ReplVertices = np.vstack((self.ReplVertices, self.Vertices[Face[0]]))
                self.ReplVertices = np.vstack((self.ReplVertices, self.Vertices[Face[1]]))
                self.ReplVertices = np.vstack((self.ReplVertices, self.Vertices[Face[2]]))
                self.ReplVertNormals = np.vstack((self.ReplVertNormals, self.VertNormals[Face[0]]))
                self.ReplVertNormals = np.vstack((self.ReplVertNormals, self.VertNormals[Face[1]]))
                self.ReplVertNormals = np.vstack((self.ReplVertNormals, self.VertNormals[Face[2]]))
                if self.RenderVertexColors:
                    self.ReplVertColors = np.vstack((self.ReplVertColors, self.VertColors[Face[0]]))
                    self.ReplVertColors = np.vstack((self.ReplVertColors, self.VertColors[Face[1]]))
                    self.ReplVertColors = np.vstack((self.ReplVertColors, self.VertColors[Face[2]]))

        self.update()

    def __del__(self):
        if self.isVBOBound:
            if self.VBOVertices is not None:
                self.VBOVertices.delete()
            if self.VBONormals is not None:
                self.VBONormals.delete()
            if self.VBOColors is not None:
                self.VBOColors.delete()


    def update(self):
        self.nPoints = len(self.ReplVertices)
        if self.nPoints == 0:
            return

        self.nPoints = len(self.ReplVertices)
        self.VBOVertices = glvbo.VBO(self.ReplVertices)
        self.VBONormals = glvbo.VBO(self.ReplVertNormals)
        if self.RenderVertexColors:
            self.VBOColors = glvbo.VBO(self.ReplVertColors)
        self.isVBOBound = True

    def draw(self, PointSize=10.0, isWireFrame=False):
        if self.isVBOBound == False:
            print('[ WARN ]: VBOs not bound. Call update().')
            return

        lightsOn()
        mat_shininess = np.array([128])
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, mat_shininess)
        gl.glPushAttrib(gl.GL_POINT_BIT)
        gl.glPointSize(PointSize)

        if self.VBOVertices is not None:
            self.VBOVertices.bind()
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3, gl.GL_DOUBLE, 0, self.VBOVertices)

        if self.VBONormals is not None:
            self.VBONormals.bind()
            gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
            gl.glNormalPointer(gl.GL_DOUBLE, 0, self.VBONormals)

        if self.VBOColors is not None and self.RenderVertexColors:
            self.VBOColors.bind()
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glColorPointer(3, gl.GL_DOUBLE, 0, self.VBOColors)
        else:
            gl.glColor4f(0.5, 0.5, 0.5, 0.8)

        if len(self.Faces) > 0:
            if isWireFrame:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.nPoints)
        else:
            gl.glDrawArrays(gl.GL_POINTS, 0, self.nPoints)

        if self.VBOColors is not None:
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        if self.VBOVertices is not None:
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        if self.VBONormals is not None:
            gl.glDisableClientState(gl.GL_NORMAL_ARRAY)

        gl.glPopAttrib()
        lightsOff()


class MeshProcessVisualize(EaselModule):
    def __init__(self, TrimeshMeshObject):
        super().__init__()
        self.Mesh = TrimeshMeshObject
        self.MeshVisualizer = TrimeshVisualizer(TrimeshMeshObject)
        self.setup()

    def setup(self):
        self.isVBOBound = False
        self.PointSize = 10.0
        self.NormalLength = 0.06
        self.showNormals = False
        self.showMesh = False
        self.VBONormalPoints = None
        self.VBOPoints = None
        self.VBOColors = None

    def init(self, argv=None):
        self.update()
        self.updateVBOs()

    def update(self):
        pass

    def updateVBOs(self):
        # VBOs
        self.nPoints = len(self.Mesh.vertices)
        if self.nPoints != 0:
            self.VBOPoints = glvbo.VBO(self.Mesh.vertices)
            Colors = Inferno_20.mpl_colormap(self.Mesh.visual.vertex_colors[:, 0])
            self.VBOColors = glvbo.VBO(Colors)
        else:
            self.VBOPoints = None


        self.nNormalPoints = len(self.Mesh.vertex_normals)*2
        self.NormalPoints = np.zeros((self.nNormalPoints, 3))
        if self.nNormalPoints != 0:
            self.NormalPoints[0::2, :] = self.Mesh.vertices
            self.NormalPoints[1::2, :] = self.Mesh.vertices + self.NormalLength * self.Mesh.vertex_normals
            self.VBONormalPoints = glvbo.VBO(self.NormalPoints)
        else:
            self.VBONormalPoints = None

        self.isVBOBound = True

    # def __del__(self):
    #     if self.isVBOBound:
    #         if self.VBOPoints is not None:
    #             self.VBOPoints.delete()
    #         if self.VBOColors is not None:
    #             self.VBOColors.delete()
    #         if self.VBONormalPoints is not None:
    #             self.VBONormalPoints.delete()

    def step(self):
        pass

    def draw(self):
        if self.isVBOBound == False:
            print('[ WARN ]: VBOs not bound. Call update().')
            return

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()

        ScaleFact = 200
        gl.glScale(ScaleFact, ScaleFact, ScaleFact)

        if self.showMesh:
            self.MeshVisualizer.draw()

        gl.glPushAttrib(gl.GL_POINT_BIT)

        gl.glPointSize(self.PointSize)
        if self.VBOPoints is not None:
            self.VBOPoints.bind()
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3, gl.GL_DOUBLE, 0, self.VBOPoints)
            if self.VBOColors is not None:
                self.VBOColors.bind()
                gl.glEnableClientState(gl.GL_COLOR_ARRAY)
                gl.glColorPointer(4, gl.GL_DOUBLE, 0, self.VBOColors)

            gl.glDrawArrays(gl.GL_POINTS, 0, self.nPoints)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            if self.VBOColors is not None:
                gl.glDisableClientState(gl.GL_COLOR_ARRAY)

        gl.glPopAttrib()

        if self.showNormals:
            gl.glPushAttrib(gl.GL_LINE_BIT)

            gl.glColor3f(1, 0, 0)
            if self.VBONormalPoints is not None:
                self.VBONormalPoints.bind()
                gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                gl.glVertexPointer(3, gl.GL_DOUBLE, 0, self.VBONormalPoints)
                gl.glDrawArrays(gl.GL_LINES, 0, self.nNormalPoints)
                gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

            gl.glPopAttrib()

        gl.glPopMatrix()

    def keyPressEvent(self, a0: QKeyEvent):
        if a0.key() == QtCore.Qt.Key_Plus:
            if self.NormalLength < 0.95:
                self.NormalLength += 0.05
                self.updateVBOs()
            # print('[ INFO ]: Updated normal length: ', self.NormalLength, flush=True)

        if a0.key() == QtCore.Qt.Key_Minus:
            if self.NormalLength > 0.05:
                self.NormalLength -= 0.05
                self.updateVBOs()
            # print('[ INFO ]: Updated normal length: ', self.NormalLength, flush=True)

        if a0.key() == QtCore.Qt.Key_A:
            if self.PointSize < 50.0:
                self.PointSize += 1.0
            # print('[ INFO ]: Updated point size: ', self.PointSize, flush=True)

        if a0.key() == QtCore.Qt.Key_Z:
            if self.PointSize > 1.0:
                self.PointSize -= 1.0
            # print('[ INFO ]: Updated point size: ', self.PointSize, flush=True)

        if a0.key() == QtCore.Qt.Key_N:
            self.showNormals = not self.showNormals

        if a0.key() == QtCore.Qt.Key_M:
            self.showMesh = not self.showMesh


Parser = argparse.ArgumentParser()
Parser.add_argument('-i', '--input', help='Specify the input mesh file name. Any format supported by trimesh.', required=True)
Parser.add_argument('-o', '--output', help='Specify the output mesh file name. Will write in OBJ format.', required=False)
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('-v', '--visualize', help='Choose to visualize sampling using tk3dv.', action='store_true', required=False)
Parser.set_defaults(visualze=False)

if __name__ == '__main__':
    Args = Parser.parse_args()
    butils.seedRandom(Args.seed)

    # Load the mesh
    Mesh = trimesh.load(Args.input)
    Mesh.vertices = odf_utils.mesh_normalize(Mesh.vertices)

    # # Subdivide mesh
    # Revertices, Refaces = trimesh.remesh.subdivide_to_size(Mesh.vertices, Mesh.faces, max_edge=PC_SAMPLER_THRESH, max_iter=10)
    # Remesh = trimesh.Trimesh(Revertices, Refaces)
    # # Remesh = Mesh
    # print('[ INFO ]: Remeshing done.', flush=True)
    # Curvature = trimesh.curvature.discrete_mean_curvature_measure(Remesh, Remesh.vertices, radius=PC_SAMPLER_THRESH)
    # # Curvature = trimesh.curvature.discrete_gaussian_curvature_measure(Remesh, Remesh.vertices, radius=PC_SAMPLER_THRESH*10)
    # Curvature = np.abs(Curvature)
    # print(np.min(Curvature), np.max(Curvature), flush=True)
    # # Curvature = 1 / (1 + np.exp(-Curvature)) # Sigmoid normalization
    # Max = np.max(Curvature)
    # Min = np.min(Curvature)
    # Curvature = (Curvature - Min) / (Max - Min) # Linear normalization
    # print(np.min(Curvature), np.max(Curvature))
    # Remesh.visual.vertex_colors = np.tile(Curvature, (3, 1)).T
    Remesh = Mesh

    if Args.output is not None:
        Remesh.export(Args.output, include_normals=True, include_color=True, include_texture=True)

    if Args.visualize:
        app = QApplication(sys.argv)
        mainWindow = Easel([MeshProcessVisualize(Remesh)], sys.argv[1:])
        mainWindow.show()
        sys.exit(app.exec_())
