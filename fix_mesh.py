import bpy 
import numpy as np
import math
from functools import partial 
import bmesh
import mathutils
from importlib import reload
import sys
import os
print(os.getcwd())
sys.path.append(os.getcwd())
import utils, drawTools, node, scope, production, graphTools, bbox, roofTools
reload(utils)
reload(drawTools)
reload(node)
reload(scope)
reload(production)
reload(graphTools)
reload(bbox)
reload(roofTools)
from utils import *
from drawTools import *
from node import *
from scope import *
from production import * 
from graphTools import *
from bbox import *
from roofTools import *
import numpy as np 
from PIL import Image
from skimage.measure import label
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import dilation
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from perlin_numpy import (
    generate_fractal_noise_2d, generate_fractal_noise_3d,
    generate_perlin_noise_2d, generate_perlin_noise_3d
)

rot_matrix = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
])

hom_rot_mat = homogenize_rotation_matrix(rot_matrix)
for obj in bpy.data.objects : 
    if obj.type == 'MESH': 
        print(obj.name)
        apply_matrix_to_mesh_obj(obj, hom_rot_mat)