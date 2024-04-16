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
## BALCONY 
#####################################################################
DOOR_WIDTH = 1
DOOR_HEIGHT = 1.5
DOOR_DEPTH = .1
BALCONY_HEIGHT = random.uniform(0.3, 0.5) * DOOR_HEIGHT
BALUSTRADE_THICKNESS = random.uniform(0.15, 0.3)
PILLAR_THICKNESS = random.uniform(0.15, 0.3)
N_TAPER = random.randint(1, 5)
MAX_TAPER = 5
TAPER_MULTIPLIER = 0.75

balcony_all = Production(
    priority=2,
    pred='balcony_all', 
    cond=ALWAYS,
    scope_modifiers=[
        partial(subdiv, axis=2, args=[GROUND_BASE, BALCONY_HEIGHT, rfloat(1)]),
        partial(subdiv, axis=2, args=[GROUND_BASE, BALCONY_HEIGHT, rfloat(1)]),
        partial(subdiv, axis=2, args=[GROUND_BASE, BALCONY_HEIGHT, rfloat(1)]),
    ],
    succ=[
        ['taper_all', 'balcony', 'overhang'], 
        ['epsilon', 'balcony_literal', 'epsilon'],
        ['taper_all', 'balcony', 'epsilon'],
    ],
)

balcony = Production(
    priority=2,
    pred='balcony', 
    cond=ALWAYS,
    scope_modifiers=[
        partial(comp, split_type=ComponentSplitType.SIDE_FACES),
    ],
    succ=[
        ['balustrade', 'epsilon', 'balustrade', 'balustrade']
    ],
    prob=[1.0]
)

balustrade = Production(
    priority=2, 
    pred='balustrade',
    cond=ALWAYS,
    scope_modifiers=[
        compose_scope_modifiers(
            partial(extrude, sz=BALUSTRADE_THICKNESS),
            partial(translate, tz=-BALUSTRADE_THICKNESS),
        )
    ],
    succ=[['balustrade_literal']],
    prob=[1.0]
)   

balustrade_literal = Production(priority=2, pred='balustrade_literal')

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

slab_literal = Production(
    priority=2,
    pred='slab_literal',
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

taper_all = Production(
    priority=2,
    pred='taper_all',
    scope_modifiers=[flip_z],
    succ=[['taper']],
    prob=[1.0]
)

taper = Production(
    priority=2, 
    pred='taper',
    cond=lambda node: node.scope.size[2] > 0.15, 
    scope_modifiers=[
        two_level_compose(
            partial(subdiv, axis=2, args=[0.15, rfloat(1)]),
            [identity, partial(scale_center, sx=rfloat(TAPER_MULTIPLIER), sy=rfloat(TAPER_MULTIPLIER))]
        ),
        partial(subdiv, axis=2, args=[0.15, rfloat(1)])
    ],
    succ=[
        ['slab_literal', 'taper'],
        ['slab_literal', 'epsilon']
    ],
    prob=[0.8, 0.2]
) 

taper2 = Production(
    priority=2, 
    pred='taper',
    cond=lambda node: node.scope.size[2] <= 0.15, 
    scope_modifiers=[identity],
    succ=[['epsilon']],
    prob=[1.]
) 

pillar_literal = Production(priority=2, pred='piller_literal')

balcony_literal = Production(priority=2, pred='balcony_literal')

geometry_registry = dict(
    slab_literal=dict(object=ASSET_DICT['slab'][0]['obj']),
    covering=dict(objects=[_['obj'] for _ in ASSET_DICT['covering']], any=True),
    pillar_literal=dict(objects=[_['obj'] for _ in ASSET_DICT['pillar']], fix=True),
    balcony_literal=dict(object=ASSET_DICT['balcony'][0]['obj']),
    balustrade_literal=dict(objects=[_['obj'] for _ in ASSET_DICT['balustrade']], fix=True),
)

prods = [balcony_all, overhang, slab_literal, balcony_literal,
    epsilon, covering, pillars, pillar_literal,
    balcony, balustrade, balustrade_literal, taper_all, taper, taper2]

TEST = Scope(3, np.array([0, 0, 0]), np.eye(3), [0.6 + DOOR_WIDTH, 1, GROUND_BASE + DOOR_HEIGHT + 0.5])
node = run_derivation(prods, 'balcony_all', TEST)
print(node)

[_.draw(geometry_registry) for _ in leaves(node)]

