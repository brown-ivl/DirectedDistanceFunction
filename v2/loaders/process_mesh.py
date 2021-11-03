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

Parser = argparse.ArgumentParser()
Parser.add_argument('-i', '--input', help='Specify the input mesh file name. Any format supported by trimesh.', required=True)
Parser.add_argument('-s', '--seed', help='Seed for any random samples.', required=True)
Parser.add_argument('-v', '--visualize', help='Choose to visualize sampling using tk3dv.', action='store_true', required=False)
Parser.set_defaults(visualze=False)



if __name__ == '__main__':
    Args = Parser.parse_args()
    butils.seedRandom(Args.seed)

    # Load the mesh
    Mesh = trimesh.load(Args.input)
    Mesh.vertices = odf_utils.mesh_normalize(Mesh.vertices)
    Norm = np.linalg.norm(Mesh.vertex_normals, axis=1)
    Mesh.vertex_normals /= Norm[:, None]

    app = QApplication(sys.argv)

    mainWindow = Easel([ODFDatasetLiveVisualizer(coord_type='direction', rays=LoadedData[0].cpu(),
                                                 intersects=LoadedData[1][0].cpu(), depths=LoadedData[1][1].cpu(),
                                                 DataLimit=Args.viz_limit)], sys.argv[1:])
    mainWindow.show()
    sys.exit(app.exec_())
