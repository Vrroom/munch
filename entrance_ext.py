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
import utils, drawTools, node, scope, production, graphTools, bbox, roofTools, assetDict
reload(utils)
reload(drawTools)
reload(node)
reload(scope)
reload(production)
reload(graphTools)
reload(bbox)
reload(roofTools)
reload(assetDict)
from utils import *
from drawTools import *
from node import *
from scope import *
from production import * 
from graphTools import *
from bbox import *
from roofTools import *
from assetDict import *
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

#set_visible(load_blend_hierarchy('./assets/pillars.blend', 'pillar_1'))
#print(list(bpy.data.objects))

#seed_everything(42) 

ASSET_DICT = eval(ASSET_DICT)
[set_invisible(_) for _ in filter(is_object, list_dict_flatten(ASSET_DICT))]

FOOTPRINT = Scope(3, np.array([0, 0, 0]), np.eye(3), [6, 6, 6]) 
def FACING_POS_X (node, *args, **kwargs) : 
    x = np.array([1.0, 0.0, 0.0])
    outward_vec = np.cross(node.scope.global_c[:, 0], node.scope.global_c[:, 1])
    return np.isclose(np.dot(x, outward_vec), 1.0)

# FLOOR PARAMETERS
NUM_FLOORS = 2
FLOOR_HEIGHT = 3
FLOOR_WIDTH = 4.5
FLOOR_LENGTH = 5
GROUND_BASE = random.uniform(0.5, 1)
FOOTPRINT = Scope(3, np.array([0, 0, 0]), np.eye(3), [FLOOR_WIDTH, FLOOR_LENGTH, GROUND_BASE + (NUM_FLOORS + 1) * FLOOR_HEIGHT])

#####################################################################
## Entrance Exterior
#####################################################################
DOOR_WIDTH = 1
DOOR_HEIGHT = 1.5
DOOR_DEPTH = .1
PILLAR_THICKNESS = random.uniform(0.15, 0.3)

entrance_ext = Production(
    priority=2,
    pred='entrance_ext', 
    cond=ALWAYS,
    scope_modifiers=[
        partial(subdiv, axis=2, args=[GROUND_BASE, rfloat(1)]),
        partial(subscope, sz=GROUND_BASE), 
        partial(subscope, sz=GROUND_BASE), 
    ],
    succ=[
        ['steps_and_slab', 'overhang'], 
        ['steps_and_slab'],
        ['steps_railing_literal']
    ],
    prob=[
        0.4,
        0.4,
        0.2
    ]
)

steps_and_slab = Production(
    priority=2,
    pred='steps_and_slab',
    cond=ALWAYS,
    scope_modifiers=[
        identity,
        partial(subdiv, axis=1, args=[rfloat(0.5), rfloat(0.5)])
    ],
    succ=[
        ['steps_and_slab_literal'],
        ['steps', 'slab_literal'],
    ],
    prob=[
        0.5,
        0.5
    ]
)

overhang = Production(
    priority=2,
    pred='overhang',
    cond=ALWAYS,
    scope_modifiers=[
        partial(subdiv, axis=2, args=[rfloat(0.7), rfloat(0.3)]),
        partial(subdiv, axis=2, args=[rfloat(0.7), rfloat(0.3)])
    ],
    succ=[
        ['epsilon', 'covering'],
        ['pillars', 'covering']
    ],
    prob=[0.5, 0.5]
)

steps_and_slab_literal = Production(
    priority=2,
    pred='steps_and_slab_literal',
    cond=ALWAYS,
)

slab_literal = Production(
    priority=2,
    pred='slab_literal',
)

steps = Production(
    priority=2,
    pred='steps',
    cond=ALWAYS,
    scope_modifiers=[identity],
    succ=[['steps_literal']],
    prob=[1.0]
)

steps_literal = Production(priority=2, pred='steps_literal')

steps_railing_literal = Production(
    priority=2,
    pred='steps_railing_literal',
)

epsilon = Production(priority=2, pred='epsilon')

covering = Production(priority=2, pred='covering')

pillars = Production(
    priority=2,
    pred='pillars',
    cond=ALWAYS,
    scope_modifiers=[
        partial(four_corner, sx=PILLAR_THICKNESS, sy=PILLAR_THICKNESS)
    ],
    succ=[['pillar_literal']],
    prob=[1.0]
)

pillar_literal = Production(priority=2, pred='piller_literal')

geometry_registry = dict(
    steps_and_slab_literal=dict(object=ASSET_DICT['steps_and_slab'][0]['obj']),
    steps_literal=dict(object=ASSET_DICT['steps'][1]['obj']),
    steps_railing_literal=dict(object=ASSET_DICT['steps'][0]['obj']),
    slab_literal=dict(object=ASSET_DICT['slab'][0]['obj']),
    covering=dict(objects=[_['obj'] for _ in ASSET_DICT['covering']], any=True),
    pillar_literal=dict(objects=[_['obj'] for _ in ASSET_DICT['pillar']], fix=True)
)

prods = [entrance_ext, steps_and_slab, overhang, 
    steps_and_slab_literal, steps_literal, slab_literal,
    steps, steps_literal, steps_railing_literal, epsilon,
    covering, pillars, pillar_literal]

TEST = Scope(3, np.array([0, 0, 0]), np.eye(3), [0.6 + DOOR_WIDTH, 1, GROUND_BASE + DOOR_HEIGHT + 0.5])
node = run_derivation(prods, 'entrance_ext', TEST)
print(node)

[_.draw(geometry_registry) for _ in leaves(node)]


