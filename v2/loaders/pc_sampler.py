import argparse
import random
import beacon.utils as butils
import trimesh
import math
from tqdm import tqdm
import multiprocessing as mp
from itertools import repeat
from functools import partial

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

PC_VERT_NOISE = 0.02
PC_TAN_NOISE = 0.02
PC_UNIFORM_RATIO = 100
PC_VERT_RATIO = 0
PC_TAN_RATIO = 0
PC_RADIUS = 1.25
PC_MAX_INTERSECT = 1

class PointCloudSampler():
    def __init__(self):
        pass

Parser = argparse.ArgumentParser()
Parser.add_argument('-i', '--input', help='Specify the input point cloud and normals in OBJ format.', required=True)
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)

if __name__ == '__main__':
    Args = Parser.parse_args()
    butils.seedRandom(Args.seed)


