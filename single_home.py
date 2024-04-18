import bpy 
import numpy as np
import math
from functools import partial 
from itertools import product, combinations
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

#####################################################################
## FLOOR
#####################################################################
NUM_FLOORS = 2
FLOOR_HEIGHT = 3
FLOOR_WIDTH = 4.5
FLOOR_LENGTH = 5
GROUND_BASE = random.uniform(0.5, 1)
FOOTPRINT = Scope(3, np.array([0, 0, 0]), np.eye(3), [FLOOR_WIDTH, FLOOR_LENGTH, GROUND_BASE + (NUM_FLOORS + 1) * FLOOR_HEIGHT])

#####################################################################
## DOOR
#####################################################################
DOOR_WIDTH = 1
DOOR_HEIGHT = 1.5
DOOR_DEPTH = .1
 
#####################################################################
## WINDOW
#####################################################################
WINDOW_SPACING = 1.5
WINDOW_WIDTH = 0.8
WINDOW_HEIGHT = 1.6

#####################################################################
## ABUTMENT
#####################################################################
ABUTMENT_SIDES = random.choice([6, 8])
ABUTMENT_WIDTH = random.uniform(1.5, 3)
ABUTMENT_LENGTH = ABUTMENT_WIDTH
N_ABUTMENT = random.randint(1, 4)

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

# WALL COLORS
WALL_COLORS = [
    np.array((14, 70, 163, 255)) / 255.,
    np.array((118, 171, 174, 255)) / 255.,
    np.array((164, 206, 149, 255)) / 255.,
    np.array((139, 50, 44, 255)) / 255. ,
    np.array((255, 235, 178, 255)) / 255. 
]

GROUND_COLORS = [
    np.array((255, 255, 255, 255)) / 255. ,
    np.array((12, 12, 12, 255)) / 255. ,
    np.array((34, 40, 49, 255)) / 255.,
]

WALL_COLOR = random.choice(WALL_COLORS)
GROUND_COLOR = random.choice(GROUND_COLORS)
    
floor = Production(
    priority=1,
    pred='floor', 
    cond=ALWAYS, 
    scope_modifiers=[
        partial(copy_n, n=2),
        partial(copy_n, n=2)
    ],
    succ=[
        ['facades', 'epsilon'],
        ['facades', 'abutments'],
    ],
    prob=[0.0, 1.0]
)

epsilon = Production(priority=2, pred='epsilon')

def MAKE_ABUTMENT (scope, n_abutment) : 
    
    nbrs = [(0, 1), (0, 3), (1, 2), (2, 4), (3, 5), (5, 6), (4, 7), (6, 7)]
    
    while True :
        abuts = random.sample([0, 1, 2, 3, 4, 5, 6, 7], k=n_abutment)
        found_mismatch = False
        for (i, j) in combinations(abuts, 2) :
            if (i, j) in nbrs or (j, i) in nbrs :
                found_mismatch = True
        if not found_mismatch :
            break
    
    factor = random.uniform(0.05, 0.1)
    fracs = [factor, 0.5, 1. - factor]
    center_rel_coords = list(product(fracs, fracs))
    center_rel_coords.remove(center_rel_coords[4])
    new_scopes = []
    
    for abut in abuts :
        rx, ry = center_rel_coords[abut]
        new_scope = deepcopy(scope)
        new_scope.size[0] = ABUTMENT_WIDTH
        new_scope.size[1] = ABUTMENT_LENGTH
        
        centre = scope.global_x + \
            ((rx * scope.size[0]) * scope.global_c[:, 0]) + \
            ((ry * scope.size[1]) * scope.global_c[:, 1])
        
        new_scope.global_x = centre - \
            ((new_scope.size[0] / 2.)* scope.global_c[:, 0]) - \
            ((new_scope.size[1] / 2.)* scope.global_c[:, 1])
        
        new_scopes.append(new_scope)
        
    return new_scopes 

abutments = Production(
    priority=1,
    pred='abutments',
    cond=ALWAYS,
    scope_modifiers=[
        partial(MAKE_ABUTMENT, n_abutment=N_ABUTMENT),
    ],
    succ=[['abutment']],
    prob=[1.0]
)

abutment = Production(
    priority=1,
    pred='abutment',
    cond=ALWAYS,
    scope_modifiers=[
        identity,
        partial(
            split_into_faces_using_shape, 
            shape=generate_ngon(ABUTMENT_SIDES, ABUTMENT_WIDTH / 2),
            top_and_bottom=True,
        )
    ],
    succ=[
        ['facades'],
        ['facade'] * ABUTMENT_SIDES + ['ground'] * 2
    ],
)

facades = Production(
    priority=2, 
    pred='facades', 
    cond=ALWAYS, 
    scope_modifiers=[partial(comp, split_type=ComponentSplitType.FACES)],
    succ=[['facade'] * 4 + ['ground'] * 2],
    prob=[1.0]
)

ground = Production(
    priority=2,
    pred='ground',
    cond=ALWAYS,
    scope_modifiers=[
        compose_scope_modifiers(
            partial(extrude, sz=0.15),
            partial(scale_center, sx=rfloat(1.1), sy=rfloat(1.1))
        )
    ],
    succ=[['ground_literal']],
#    succ=[['epsilon']],
    prob=[1.0]
)

def FACING_POS_X (node, *args, **kwargs) :
    x = np.array([1.0, 0.0, 0.0])
    outward_vec = np.cross(node.scope.global_c[:, 0], node.scope.global_c[:, 1])
    return np.isclose(np.dot(x, outward_vec), 1.0)

facade = Production(
    priority=2,
    pred='facade',
    cond=FACING_POS_X,
    scope_modifiers=[
        partial(subdiv, axis=0, args=[rfloat(1), 1.5 * DOOR_WIDTH]),
        partial(subdiv, axis=0, args=[1.5 * DOOR_WIDTH, rfloat(1)])
    ],
    succ=[
        ['tiles', 'entrance'],
        ['entrance', 'tiles']
    ],
    prob=[0.5, 0.5]
)

facade2 = Production(
    priority=2,
    pred='facade',
    cond=complement(FACING_POS_X),
    scope_modifiers=[identity],
    succ=[['tiles']],
    prob=[1.0]
)

def ENTRANCE_COND(node) :
    allnodes = all_nodes(find_root(node))
    axioms = [_.id for _ in allnodes]
    ancestors = get_ancs(node)
    anc_axioms = [_.id for _ in ancestors]
    cond1 = 'wall_door' not in axioms
    cond2 = 'abutments' not in anc_axioms
    cond3 = node.scope.size[1] >= DOOR_WIDTH
    abutment_nodes = [_ for _ in allnodes if _.id == 'abutment']
    cond4 = not check_contained(node, abutment_nodes)
    return cond1 and cond2 and cond3 and cond4
 
entrance = Production(
    priority=4,
    pred='entrance', 
    cond=ENTRANCE_COND, # make sure that none of the children of floor have expanded door yet,
    scope_modifiers=[
        partial(subdiv, axis=0, args=[rfloat(1), DOOR_WIDTH, rfloat(1)])
    ],
    succ=[['wall', 'wall_door', 'wall']],
    prob=[1.0]
)

entrance2 = Production(
    priority=4,
    pred='entrance',
    cond=complement(ENTRANCE_COND), 
    scope_modifiers=[
        identity,
    ],
    succ=[['tiles']],
    prob=[1.0]
)

wall_door = Production(
    priority=4,
    pred='wall_door',
    cond=ALWAYS,
    scope_modifiers=[partial(subdiv, axis=1, args=[DOOR_HEIGHT, rfloat(1)])],
    succ=[['door_balcony', 'wall']],
    prob=[1.0]
)

@validate_scopes
def door_balcony_scope_mod (scope) : 
    scope1 = extrude(scope, sz=DOOR_DEPTH)[0]
    scope2 = scale_center(
        subscope(
            extrude(scope, sz=1)[0], 
            tz=DOOR_DEPTH
        )[0],
        sx=0.6 + DOOR_WIDTH,
    )[0]
    x = np.copy(scope2.global_c[:, 0])
    y = np.copy(scope2.global_c[:, 1])
    z = np.copy(scope2.global_c[:, 2])
    scope2.global_c[:, 0] = x
    scope2.global_c[:, 1] = -z
    scope2.global_c[:, 2] = y
    scope2.global_x += z * scope2.size[2]
    scope2.size[1], scope2.size[2] = scope2.size[2], scope2.size[1]
    scope2.global_x -= GROUND_BASE * scope2.global_c[:, 2]
    scope2.size[2] = GROUND_BASE + DOOR_HEIGHT + 0.5
    return [scope1, scope2]

door_balcony = Production(
    priority=4, 
    pred='door_balcony', 
    cond=ALWAYS, 
    scope_modifiers=[door_balcony_scope_mod], 
    succ=[['door_literal', 'balcony_all']],
    prob=[1.0]
)       

balcony_all = Production(
    priority=4,
    pred='balcony_all', 
    cond=ALWAYS,
    scope_modifiers=[
        partial(subdiv, axis=2, args=[GROUND_BASE, BALCONY_HEIGHT, rfloat(1)]),
#        partial(subdiv, axis=2, args=[GROUND_BASE, BALCONY_HEIGHT, rfloat(1)]),
#        partial(subdiv, axis=2, args=[GROUND_BASE, BALCONY_HEIGHT, rfloat(1)]),
    ],
    succ=[
        ['taper_all', 'balcony', 'overhang'], 
#        ['epsilon', 'balcony_literal', 'epsilon'],
#        ['taper_all', 'balcony', 'epsilon'],
    ],
)

balcony = Production(
    priority=4,
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
    priority=4, 
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
    priority=4,
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

door_literal = Production(priority=4, pred='door_literal')

#facade = Production(
#    priority=2, 
#    pred='facade', 
#    cond=ALWAYS,
#    scope_modifiers=[identity],
#    succ=[['tiles']],
#    prob=[1.0]
#)

tiles = Production(
    priority=2, 
    pred='tiles',
    cond=ALWAYS,
    scope_modifiers=[partial(repeat, axis=0, val=WINDOW_SPACING)],
    succ=[['tile']],
    prob=[1.0],
)

tile = Production(
    priority=2,
    pred='tile',
    cond=lambda node : node.scope.size[0] > WINDOW_WIDTH,
    scope_modifiers=[partial(subdiv, axis=0, args=[rfloat(1), WINDOW_WIDTH, rfloat(1)])],
    succ=[['wall', 'wall_window', 'wall']],
    prob=[1.0]
)

tile2 = Production(
    priority=2,
    pred='tile',
    cond=lambda node : node.scope.size[0] <= WINDOW_WIDTH,
    scope_modifiers=[identity],
    succ=[['wall']],
    prob=[1.0]
)

wall = Production(
    priority=3,
    pred='wall',
    cond=lambda node: not check_contained(node, list(filter(lambda x : x.id != 'abutments', no_parent(node)))),
    scope_modifiers=[partial(extrude, sz=0.01)],
    succ=[['wall_literal']],
    prob=[1.0]
)

wall_literal = Production(priority=3, pred='wall_literal')
ground_literal = Production(priority=3, pred='ground_literal')

wall2 = Production(
    priority=3,
    pred='wall',
    cond=lambda node: check_contained(node, list(filter(lambda x : x.id != 'abutments', no_parent(node)))),
    scope_modifiers=[partial(extrude, sz=0.01)],
    succ=[['epsilon']],
    prob=[1.0]
)

wall_window = Production(
    priority=2,
    pred='wall_window',
    cond=ALWAYS,
    scope_modifiers=[partial(subdiv, axis=1, args=[rfloat(2), WINDOW_HEIGHT, rfloat(1)])],
    succ=[['wall', 'window', 'wall']],
    prob=[1.0]
)

window = Production(
    priority=4,
    pred='window',
    cond=lambda node: not check_intersect(node, list(filter(lambda x : x.id != 'abutments', no_parent(node)))),
    scope_modifiers=[
        compose_scope_modifiers(
            partial(extrude, sz=DOOR_DEPTH),
            partial(scale_center, sx=rfloat(1.1), sy=rfloat(1.1))
        )
    ],
    succ=[['window_literal']],
    prob=[1.0],
)

window2 = Production(
    priority=4,
    pred='window',
    cond=lambda node: check_intersect(node, list(filter(lambda x : x.id != 'abutments', no_parent(node)))),
    scope_modifiers=[identity],
    succ=[['wall']],
    prob=[1.0],
)

window_literal = Production(
    priority=4,
    pred='window_literal',
    cond=ALWAYS,
    scope_modifiers=[],
    succ=[],
    prob=[],
)
    
FLOOR_FOOTPRINT = Scope(3, np.array([0, 0, 0]), np.eye(3), [FLOOR_WIDTH, FLOOR_LENGTH, FLOOR_HEIGHT])

geometry_registry = dict(
    slab_literal=dict(object=ASSET_DICT['slab'][0]['obj']),
    covering=dict(objects=[_['obj'] for _ in ASSET_DICT['covering']], any=True),
    pillar_literal=dict(objects=[_['obj'] for _ in ASSET_DICT['pillar']], fix=True),
    balcony_literal=dict(object=ASSET_DICT['balcony'][0]['obj']),
    balustrade_literal=dict(objects=[_['obj'] for _ in ASSET_DICT['balustrade']], fix=True),
    window_literal=dict(object=ASSET_DICT['window'][0]['obj']),
    door_literal=dict(object=ASSET_DICT['door'][0]['obj']),
)

#balcony_prods = [balcony_all, overhang, slab_literal, balcony_literal,
#    epsilon, covering, pillars, pillar_literal,
#    balcony, balustrade, balustrade_literal, taper_all, taper, taper2]

prods = [floor, epsilon, abutments, 
        abutment, facades, facade, facade2, 
        tiles, tile, tile2, wall, 
        wall2, wall_literal, wall_window, wall_door, 
        window, window2, window_literal,
        ground, entrance, entrance2,
        door_balcony, door_literal, 
        balcony_all, overhang, slab_literal, balcony_literal,
        covering, pillars, pillar_literal, balcony, balustrade, 
        balustrade_literal, taper_all, taper, taper2,]

#TEST = Scope(3, np.array([0, 0, 0]), np.eye(3), [3, 3, 10])
#R = euler_xyz_rotation_matrix(0.2, 0.2, 0.2)
#TEST = rotate_scope(TEST, R)
#shape = generate_ngon(6, 1.5)
#ShapeScope.create_from_scope(shape, TEST).draw()
#scopes = split_into_faces_using_shape(TEST, shape)
#[_.draw() for _ in scopes]
#TEST.draw()
node = run_derivation(prods, 'floor', FLOOR_FOOTPRINT)
#print(node)
leaf_nodes = leaves(node)
clear_all()
all_objs = [_.draw(geometry_registry) for _ in leaf_nodes]

wall_objs = [_ for n, _ in zip(leaf_nodes, all_objs) if n.id == 'wall_literal']
ground_objs = [_ for n, _ in zip(leaf_nodes, all_objs) if n.id == 'ground_literal']
#solid_texture_objects(wall_objs)

for obj in wall_objs : 
    subdivide_n(obj, 3)
    
for obj in ground_objs : 
    subdivide_n(obj, 3)

noise = generate_perlin_noise_3d(
    (128, 128, 128), (4, 4, 4), #1,# tileable=(True, True, True)
)

#print(noise.min(), noise.max())
##noise = (noise - noise.min()) / (noise.max() - noise.min())

def apply_wall_color(pt) :
    i, j, k = [int(127 * _) for _ in pt]
    return clamp_elem_wise_np(noise[i, j, k] / 2.0 + WALL_COLOR, 0, 1)

def apply_ground_color (pt) :
    i, j, k = [int(127 * _) for _ in pt]
    return clamp_elem_wise_np(noise[i, j, k] / 2.0 + GROUND_COLOR, 0, 1)

solid_texture_objects(wall_objs, apply_wall_color)
solid_texture_objects(ground_objs, apply_ground_color)
