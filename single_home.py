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


# WINDOW_SPACING = 1.7
# WINDOW_WIDTH = .4
# WINDOW_HEIGHT = .8
# WINDOW_DEPTH = .1
# WALL_DEPTH = .1
# WINDOW_OBJ = load_blend_model('assets/window.blend', 'wind_low')
# DOOR_OBJ = load_blend_model('assets/door.blend', 'Mansion Door')
# BRICK_OBJ = load_blend_model('assets/single_brick.blend', 'single_brick')
# ROOF_TILE_OBJ = load_blend_model('assets/roof.blend', 'roof_tile') 
# #BRICK_MAT = import_material_from_file('assets/brick.blend', 'brick_material')
# BRICK_WIDTH = 0.2
# BRICK_HEIGHT = 0.07
# BRICK_COLOR = [1.0, 0.69, 0.38, 1.0]
# ROOF_TILE_SZ = 1.0

# multi_storey = Production(
#     priority=0, 
#     pred='multi_storey',
#     cond=ALWAYS,
#     scope_modifiers=[
#         partial(repeat, axis=2, val=2)
#     ],
#     succ=[['ground_floor', 'floor', 'top_floor']],
#     prob=[1.0]
# )
# 
# ground_floor = Production(
#     priority=1,
#     pred='ground_floor', 
#     cond=ALWAYS, 
#     scope_modifiers=[identity], 
#     succ=[['ground_floor_facades']], 
#     prob=[1.0]
# )
# 
# def custom_scope_modifier (scope) : 
#     return scope, scale_scope(scale_scope(scope, 0.5, 0), 1.5, 1)
# 
# floor = Production(
#     priority=1,
#     pred='floor', 
#     cond=ALWAYS, 
# #    scope_modifiers=[identity],
#     scope_modifiers=[custom_scope_modifier], 
# #    succ=[['facades'],
#     succ=[['facades', 'facades']], 
#     prob=[1.0]
# )
# 
# top_floor = Production(
#     priority=1,
#     pred='top_floor', 
#     cond=ALWAYS, 
#     scope_modifiers=[
#         partial(subdiv, axis=2, args=[rfloat(0.8), rfloat(0.2)])
#     ],
#     succ=[['facades', 'roof']], 
#     prob=[1.0]
# )
# 
# #house = Production(
# #    id='1', 
# #    priority=1,
# #    pred='house',
# #    cond=ALWAYS,
# #    scope_modifiers=[
# #        partial(subdiv, axis=2, args=[rfloat(0.95), rfloat(0.05)])
# #    ],
# #    succ=[['facades', 'roof']],
# #    prob=[1.0]
# #)
# 
# roof = Production(
#     priority=3, 
#     pred='roof', 
#     cond=ALWAYS, 
#     scope_modifiers=[
# #        partial(subscope, sx=rfloat(1.02), sy=rfloat(1.02), sz=rfloat(1.2), tx=-0.06, ty=-0.06, tz=0.0)
#     ],
# #    succ=[['expanded_roof']],
#     succ=[],
#     prob=[],#1.0]
# )
# 
# expanded_roof = Production(
#     priority=3,
#     pred='expanded_roof', 
#     cond=ALWAYS,
#     scope_modifiers=[
#         compose(
#             partial(repeat, axis=0, val=ROOF_TILE_SZ),
#             partial(map, partial(repeat, axis=1, val=ROOF_TILE_SZ)),
#             flatten,
#             list
#         )
#     ],
#     succ=[['roof_tile']],
#     prob=[1.0]
# )
# 
# roof_tile = Production(
#     priority=3,
#     pred='roof_tile', 
#     cond=ALWAYS,
#     scope_modifiers=[],
#     succ=[],
#     prob=[]
# )             
# 
# ground_floor_facades = Production(
#     priority=2, 
#     pred='ground_floor_facades', 
#     cond=ALWAYS, 
#     scope_modifiers=[partial(comp, split_type=ComponentSplitType.SIDE_FACES)],
#     succ=[['ground_floor_facade']],
#     prob=[1.0]
# )
# 
# facades = Production(
#     priority=2, 
#     pred='facades', 
#     cond=ALWAYS, 
#     scope_modifiers=[partial(comp, split_type=ComponentSplitType.SIDE_FACES)],
#     succ=[['facade']],
#     prob=[1.0]
# )
# 
# ground_floor_facade = Production(
#     priority=2, 
#     pred='ground_floor_facade', 
#     cond=FACING_POS_X,
#     scope_modifiers=[
#         partial(subdiv, axis=0, args=[rfloat(1), 1.5 * DOOR_WIDTH]), 
#         partial(subdiv, axis=0, args=[1.5 * DOOR_WIDTH, rfloat(1)])
#     ],
#     succ=[
#         ['tiles', 'entrance'], 
#         ['entrance', 'tiles']
#     ],
#     prob=[0.5, 0.5]
# )
# 
# ground_floor_facade_2 = Production(
#     priority=2, 
#     pred='ground_floor_facade', 
#     cond=complement(FACING_POS_X),
#     scope_modifiers=[identity],
#     succ=[['tiles']],
#     prob=[1.0]
# )
# 
# facade = Production(
#     priority=2, 
#     pred='facade', 
#     cond=ALWAYS,
#     scope_modifiers=[identity],
#     succ=[['tiles']],
#     prob=[1.0]
# )
# 
# tiles = Production(
#     priority=2, 
#     pred='tiles',
#     cond=ALWAYS,
#     scope_modifiers=[partial(repeat, axis=0, val=WINDOW_SPACING)],
#     succ=[['tile']],
#     prob=[1.0],
# )
# 
# tile = Production(
#     priority=2,
#     pred='tile',
#     cond=ALWAYS,
#     scope_modifiers=[partial(subdiv, axis=0, args=[rfloat(1), WINDOW_WIDTH, rfloat(1)])],
#     succ=[['wall', 'wall_window', 'wall']],
#     prob=[1.0]
# )
# 
# def OCCLUDED (node, *args, **kwargs) : 
#     return check_intersect(node, no_parent(node))
# 
# window = Production(
#     priority=4,
#     pred='window',
#     cond=complement(OCCLUDED),
#     scope_modifiers=[partial(extrude, sz=DOOR_DEPTH)],
#     succ=[['window_literal']],
#     prob=[1.0],
# )
# 
# window2 = Production(
#     priority=4,
#     pred='window',
#     cond=OCCLUDED,
#     scope_modifiers=[identity],
#     succ=[['wall']],
#     prob=[1.0],
# )
# 
# wall = Production(
#     priority=2,
#     pred='wall',
#     cond=ALWAYS,
# #    scope_modifiers=[],
# #    succ=[],
# #    prob=[]
#     scope_modifiers=[partial(extrude, sz=0.01)],
#     succ=[['wall_literal']],
#     prob=[1.0]
# )
# 
# entrance = Production(
#     priority=2,
#     pred='entrance',
#     cond=ALWAYS,
#     scope_modifiers=[partial(subdiv, axis=0, args=[rfloat(1), DOOR_WIDTH, rfloat(1)])],
#     succ=[['wall', 'wall_door', 'wall']],
#     prob=[1.0]
# )
# 
# wall_window = Production(
#     priority=2,
#     pred='wall_window',
#     cond=ALWAYS,
#     scope_modifiers=[partial(subdiv, axis=1, args=[rfloat(2), WINDOW_HEIGHT, rfloat(1)])],
#     succ=[['wall', 'window', 'wall']],
#     prob=[1.0]
# )
# 
# wall_door = Production(
#     priority=2,
#     pred='wall_door',
#     cond=ALWAYS,
#     scope_modifiers=[partial(subdiv, axis=1, args=[DOOR_HEIGHT, rfloat(1)])],
#     succ=[['door', 'wall']],
#     prob=[1.0]
# )
# 
# door = Production(
#     priority=2,
#     pred='door',
#     cond=ALWAYS, 
#     scope_modifiers=[partial(extrude, sz=DOOR_DEPTH)],
#     succ=[['door_literal']],
#     prob=[1.0],
# )
# 
# window_literal = Production(
#     priority=2,
#     pred='window_literal',
#     cond=ALWAYS,
#     scope_modifiers=[],
#     succ=[],
#     prob=[],
# )
# 
# door_literal = Production(
#     priority=2,
#     pred='door_literal',
#     cond=ALWAYS,
#     scope_modifiers=[],
#     succ=[],
#     prob=[],
# )
# 
# wall_literal = Production(
#     priority=2,
#     pred='wall_literal',
#     cond=ALWAYS,
#     scope_modifiers=[],
#     succ=[],
#     prob=[],
# )
# 
# bricks = Production(
#     priority=2,
#     pred='bricks',
#     cond=ALWAYS,
#     scope_modifiers=[
#         compose(
#             partial(repeat, axis=0, val=BRICK_WIDTH),
#             partial(map, partial(repeat, axis=1, val=BRICK_HEIGHT)),
#             flatten,
#             list,
#             partial(map, partial(subscope, sx=rfloat(0.95), sy=rfloat(0.95), sz=rfloat(0.95), tx=0, ty=0, tz=0)), 
#             flatten,
#             list
#         )
#     ],
#     succ=[['brick']],
#     prob=[1.0],
# )
# 
# ASSET_DICT = eval(ASSET_DICT)
# [set_invisible(_) for _ in filter(is_object, list_dict_flatten(ASSET_DICT))]
# 
# geometry_registry = dict(
#     window_literal=dict(object=WINDOW_OBJ),
#     door_literal=dict(object=DOOR_OBJ),
#     brick=dict(object=BRICK_OBJ),
#     roof_tile=dict(object=ROOF_TILE_OBJ),
#     roof=dict(object=mansard_roof (10, 10, 2, 0.2, name='mansard')),
#     wall_literal=dict(objects=[_['obj'] for _ in ASSET_DICT['wall'] if 'light' in _['props']])
# #    wall_literal=dict(material=BRICK_MAT)
# )
# 
# for k in geometry_registry.keys() : 
#     if 'object' in geometry_registry[k] : 
#         set_invisible(geometry_registry[k]['object'])
#  
# all_prods = [
#     multi_storey,
#     ground_floor, floor, top_floor,
#     ground_floor_facades, facades,
#     ground_floor_facade, ground_floor_facade_2,
#     facade,
#     tiles, tile,
#     window, window2, wall, entrance,
#     wall_door, door, wall_window,
#     window_literal, door_literal,
# #    expanded_roof, roof_tile, roof
#     roof, wall_literal
# ]
# 
# set_material_color('Procedural Curved Pottery Clay', BRICK_COLOR)
# 
# node = run_derivation(all_prods, 'multi_storey', FOOTPRINT)
# #clear_all()
# 
# #print(len(no_parent(node.children[0])))
# #FOOTPRINT.draw()
# #scale_scope(FOOTPRINT, 0.5, 0).draw()
# #print(node)
# [_.draw(geometry_registry) for _ in leaves(node)]
# 
# #print([_['obj'] for _ in ASSET_DICT['wall'] if 'light' in _['props']])
# 
# #gabled_roof (5, 10, 5)
# #gabled_roof (5, 5, 5)



