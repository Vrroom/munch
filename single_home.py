from infinigen.assets.lighting import sky_lighting
from infinigen.assets.scatters import grass, ivy
from infinigen.assets.materials import ice, bark_birch, dirt, lava, mud, sand, sandstone, cobble_stone
from infinigen.assets.fluid import liquid_particle_material, fluid
from infinigen.assets.weather import kole_clouds
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
import utils, drawTools, node, scope, production, graphTools, bbox, roofTools, assetDict, cameraTools
reload(utils)
reload(drawTools)
reload(node)
reload(scope)
reload(production)
reload(graphTools)
reload(bbox)
reload(roofTools)
reload(assetDict)
reload(cameraTools)
from utils import *
from drawTools import *
from node import *
from scope import *
from production import * 
from graphTools import *
from bbox import *
from roofTools import *
from assetDict import *
from cameraTools import *
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
import argparse

#seed_everything(42) 

bpy.ops.preferences.addon_enable(module='flip_fluids_addon')
bpy.ops.flip_fluid_operators.complete_installation()

#####################################################################
## PERLIN NOISE
#####################################################################

def generate_house_on_footprint (footprint) : 
    pass

NOISE = generate_perlin_noise_3d(
    (128, 128, 128), (4, 4, 4),
)

FRACTAL_NOISE = generate_fractal_noise_3d(
    (128, 128, 128), (4, 4, 4), 1, tileable=(True, True, True)
)

#####################################################################
## ASSET DATABASE
#####################################################################
ASSET_DICT = eval(ASSET_DICT)
[set_invisible(_) for _ in filter(is_object, list_dict_flatten(ASSET_DICT))]

#####################################################################
## FLOOR
#####################################################################
NUM_FLOORS = random.randint(2, 4)
FLOOR_HEIGHT = 3
FLOOR_WIDTH = random.uniform(4., 6.)
FLOOR_LENGTH = random.uniform(4., 7.)
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
WINDOW_SPACING = random.uniform(1.2, 1.8)
WINDOW_WIDTH = 0.8
WINDOW_HEIGHT = 1.6

#####################################################################
## ABUTMENT
#####################################################################
ABUTMENT_SIDES = random.choice([6, 8])
ABUTMENT_WIDTH = random.uniform(1.5, 3)
ABUTMENT_LENGTH = ABUTMENT_WIDTH
N_ABUTMENT = random.randint(1, 4) 
NBRS = [(0, 1), (0, 3), (1, 2), (2, 4), (3, 5), (5, 6), (4, 7), (6, 7)]

#####################################################################
## DECORATIONS
#####################################################################
APPLY_IVY = random.choice([True, False])
GROUND_MAT = random.choice([mud, sand, sandstone, cobble_stone])
    
while True :
    ABUTS = random.sample([0, 1, 2, 3, 4, 5, 6, 7], k=N_ABUTMENT)
    found_mismatch = False
    for (i, j) in combinations(ABUTS, 2) :
        if (i, j) in NBRS or (j, i) in NBRS :
            found_mismatch = True
    if not found_mismatch :
        break

ABUTS_FACTOR = random.uniform(0.05, 0.1)
ABUTS_FRAC = [ABUTS_FACTOR, 0.5, 1. - ABUTS_FACTOR]

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

#####################################################################
## CHIMNEY 
#####################################################################
CHIMNEY_X = random.random()
CHIMNEY_Y = random.random()
CHIMNEY_WIDTH = 0.5
CHIMNEY_HEIGHT = random.uniform(2., 4.)

#####################################################################
## COLORS
#####################################################################
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

ROOF_COLORS = [
    np.array((165, 47, 31, 255)) / 255.,
    np.array((195, 102, 89, 255)) / 255.,
    np.array((184, 66, 33, 255)) / 255.,
    np.array((6, 50, 48, 255)) / 255.
]

WALL_COLOR = random.choice(WALL_COLORS)
GROUND_COLOR = random.choice(GROUND_COLORS)
ROOF_COLOR = random.choice(ROOF_COLORS)

single_home = Production(
    priority=0, 
    pred='single_home', 
    cond=ALWAYS, 
    scope_modifiers=[
        partial(subdiv, axis=2, args=[GROUND_BASE + FLOOR_HEIGHT, *([FLOOR_HEIGHT] * NUM_FLOORS)]),
    ],
    succ=[['ground_floor', *(['floor'] * (NUM_FLOORS - 1)), 'roof']],
)

def ROOF_SCOPE_MOD (scope) : 
    chimney_scope = Scope(
        3,
        scope.global_x + \
            CHIMNEY_X * scope.size[0] * scope.global_c[:, 0] + \
            CHIMNEY_Y * scope.size[1] * scope.global_c[:, 1],
        scope.global_c, 
        [CHIMNEY_WIDTH, CHIMNEY_WIDTH, CHIMNEY_HEIGHT]
    )
    smoke_scope = Scope(
        3,
        scope.global_x + \
            CHIMNEY_X * scope.size[0] * scope.global_c[:, 0] + \
            CHIMNEY_Y * scope.size[1] * scope.global_c[:, 1] + \
            (CHIMNEY_HEIGHT - CHIMNEY_WIDTH) * scope.global_c[:, 2],
        scope.global_c,
        [CHIMNEY_WIDTH, CHIMNEY_WIDTH, CHIMNEY_WIDTH]
    )

    roof_scope = deepcopy(scope)
    roof_scope.size[2] *= random.uniform(0.5, 1.0)
    return [scope, chimney_scope, smoke_scope] 

roof = Production(
    priority=1, 
    pred='roof',
    cond=ALWAYS, 
    scope_modifiers=[
        ROOF_SCOPE_MOD
    ],
    succ=[['roof_literal', 'chimney_literal', 'epsilon']],
    prob=[1.0]
)

chimney_literal = Production(priority=1, pred='chimney_literal')

smoke_literal = Production(priority=1, pred='smoke_literal')

roof_literal = Production(priority=1, pred='roof_literal')

ground_floor = Production(
    priority=1, 
    pred='ground_floor', 
    cond = ALWAYS,
    scope_modifiers=[
        partial(subdiv, axis=2, args=[GROUND_BASE, FLOOR_HEIGHT]),
    ],
    succ=[['slab_literal', 'floor']],
    prob=[1.0]
)
    
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
    prob=[0.1, 9.0]
)

epsilon = Production(priority=2, pred='epsilon')

def MAKE_ABUTMENT (scope, n_abutment) : 
    center_rel_coords = list(product(ABUTS_FRAC, ABUTS_FRAC))
    center_rel_coords.remove(center_rel_coords[4])
    new_scopes = []
    
    for abut in ABUTS :
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

def DOES_NOT_INTERSECT_DOOR_LITERAL (node) :
    floor_node = find_anc_with_axiom(node, 'floor') 
    door_literals = [_ for _ in leaves(floor_node) if _.id == 'door_literal']
    return not check_intersect(node, door_literals)

abutment = Production(
    priority=5,
    pred='abutment',
    cond=DOES_NOT_INTERSECT_DOOR_LITERAL,
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

abutment2 = Production(
    priority=5,
    pred='abutment',
    cond=complement(DOES_NOT_INTERSECT_DOOR_LITERAL),
    scope_modifiers=[identity],
    succ=[['epsilon']],
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
    prob=[1.0]
)

def FACADE_COND (node, *args, **kwargs) :
    ancestors = get_ancs(node)
    anc_axioms = [_.id for _ in ancestors]
    big_enough = node.scope.size[0] >= DOOR_WIDTH
    if not big_enough or 'abutments' in anc_axioms: 
        return False
    if 'ground_floor' in anc_axioms :
        # for ground floor, entrance should face a direction
        x = np.array([1.0, 0.0, 0.0])
        outward_vec = np.cross(node.scope.global_c[:, 0], node.scope.global_c[:, 1])
        return np.isclose(np.dot(x, outward_vec), 1.0)
    else :
        # on a floor, do whatever you want. 
        return True 
   
facade = Production(
    priority=2,
    pred='facade',
    cond=FACADE_COND,
    scope_modifiers=[
        partial(subdiv, axis=0, args=[rfloat(1), 1.5 * DOOR_WIDTH]),
        partial(subdiv, axis=0, args=[1.5 * DOOR_WIDTH, rfloat(1)]),
    ],
    succ=[
        ['tiles', 'entrance'],
        ['entrance', 'tiles'],
    ],
    prob=[0.5, 0.5]
)

facade2 = Production(
    priority=2,
    pred='facade',
    cond=complement(FACADE_COND),
    scope_modifiers=[identity],
    succ=[['tiles']],
    prob=[1.0]
)

def ENTRANCE_COND(node) :
    ancestors = get_ancs(node)
    anc_axioms = [_.id for _ in ancestors]
    if 'ground_floor' in anc_axioms :   
        return True
    else :
        floor_node = find_anc_with_axiom(node, 'floor')
        subtree = all_nodes(floor_node)
        wall_door_count = len([_ for _ in subtree if _.id == 'wall_door'])
        return wall_door_count <= 1
 
entrance = Production(
    priority=4,
    pred='entrance', 
    cond=ENTRANCE_COND, 
    scope_modifiers=[
        partial(subdiv, axis=0, args=[rfloat(1), DOOR_WIDTH, rfloat(1)]), 
    ],
    succ=[['wall', 'wall_door', 'wall']],
    prob=[1.]
)

entrance2 = Production(
    priority=6,
    pred='entrance',
    cond=ALWAYS, 
    scope_modifiers=[
        identity,
    ],
    succ=[['tiles']],
    prob=[1.0]
)

def BALCONY_WALL_DOOR (node) : 
    ancestors = get_ancs(node)
    anc_axioms = [_.id for _ in ancestors]
    return 'ground_floor' not in anc_axioms
    
wall_door = Production(
    priority=4,
    pred='wall_door',
    cond=BALCONY_WALL_DOOR,
    scope_modifiers=[partial(subdiv, axis=1, args=[DOOR_HEIGHT, rfloat(1)])],
    succ=[['door_balcony', 'wall']],
    prob=[1.0]
)

wall_door2 = Production(
    priority=4,
    pred='wall_door',
    cond=complement(BALCONY_WALL_DOOR),
    scope_modifiers=[partial(subdiv, axis=1, args=[DOOR_HEIGHT, rfloat(1)])],
    succ=[['door_entrance_ext', 'wall']],
    prob=[1.0]
)

@validate_scopes
def door_entrance_ext_scope_mod (scope) : 
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

door_entrance_ext = Production(
    priority=4, 
    pred='door_entrance_ext', 
    cond=ALWAYS, 
    scope_modifiers=[door_entrance_ext_scope_mod], 
    succ=[['door_literal', 'entrance_ext']],
    prob=[1.0]
)       

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

steps_and_slab_literal = Production(
    priority=2,
    pred='steps_and_slab_literal',
    cond=ALWAYS,
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
        partial(subdiv, axis=2, args=[GROUND_BASE, BALCONY_HEIGHT, rfloat(1)]),
    ],
    succ=[
        ['taper_all', 'balcony', 'overhang'], 
        ['taper_all', 'balcony', 'epsilon'],
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

pillar_literal = Production(priority=4, pred='piller_literal')

balcony_railing = Production(
    priority=4, 
    pred='balcony_railing',
    cond=ALWAYS, 
    scope_modifiers=[partial(subscope, tz=-0.5)],
    succ=[['balcony_literal']],
    prob=[1.]
)

balcony_literal = Production(priority=4, pred='balcony_literal')

door_literal = Production(priority=4, pred='door_literal')

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
    cond=ALWAYS, # lambda node: not check_contained(node, list(filter(lambda x : x.id != 'abutments', no_parent(node)))),
    scope_modifiers=[partial(extrude, sz=0.01)],
    succ=[['wall_literal']],
    prob=[1.0]
)

wall_literal = Production(priority=3, pred='wall_literal')
ground_literal = Production(priority=3, pred='ground_literal')

wall2 = Production(
    priority=3,
    pred='wall',
    cond=NEVER, # lambda node: check_contained(node, list(filter(lambda x : x.id != 'abutments', no_parent(node)))),
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
    
FLOOR_FOOTPRINT = Scope(3, np.array([0, 0, 0]), np.eye(3), [FLOOR_WIDTH, FLOOR_LENGTH, GROUND_BASE + FLOOR_HEIGHT])

roof_objects = [
    hipped_roof(1, 1, 1, 0.3), 
    hipped_roof(1, 1, 1, 0.5), 
    hipped_roof(1, 1, 1, 0.1),
    hipped_roof(1, 1, 1, 0.0),
    mansard_roof(1, 1, 1, 0.3)
]

for obj in roof_objects : 
    set_invisible(obj)

geometry_registry = dict(
    steps_and_slab_literal=dict(object=ASSET_DICT['steps_and_slab'][0]['obj']),
    steps_literal=dict(object=ASSET_DICT['steps'][1]['obj']),
    steps_railing_literal=dict(object=ASSET_DICT['steps'][0]['obj']),
    slab_literal=dict(object=ASSET_DICT['slab'][0]['obj']),
    covering=dict(objects=[_['obj'] for _ in ASSET_DICT['covering']], any=True),
    pillar_literal=dict(objects=[_['obj'] for _ in ASSET_DICT['pillar']], fix=True),
    balcony_literal=dict(object=ASSET_DICT['balcony'][0]['obj']),
    balustrade_literal=dict(objects=[_['obj'] for _ in ASSET_DICT['balustrade']], fix=True),
    window_literal=dict(object=ASSET_DICT['window'][0]['obj']),
    door_literal=dict(object=ASSET_DICT['door'][0]['obj']),
    chimney_literal=dict(object=ASSET_DICT['chimney'][0]['obj']), 
    roof_literal=dict(objects=roof_objects, any=True),
)

prods = [single_home, ground_floor, floor, epsilon, abutments, 
        abutment, abutment2, facades, facade, facade2, 
        tiles, tile, tile2, wall, 
        wall2, wall_literal, wall_window, wall_door, wall_door2,
        window, window2, window_literal,
        ground, entrance, entrance2,
        door_balcony, door_literal, 
        balcony_all, overhang, slab_literal, balcony_literal, balcony_railing,
        covering, pillars, pillar_literal, balcony, balustrade, 
        balustrade_literal, taper_all, taper, taper2,
        entrance_ext, door_entrance_ext, steps_and_slab, 
        steps_and_slab_literal, steps, steps_literal, steps_railing_literal,
        roof, chimney_literal, roof_literal, smoke_literal]

def apply_wall_color(pt) :
    i, j, k = [int(127 * _) for _ in pt]
    return clamp_elem_wise_np(NOISE[i, j, k] / 2.0 + WALL_COLOR, 0, 1)

def apply_ground_color (pt) :
    i, j, k = [int(127 * _) for _ in pt]
    return clamp_elem_wise_np(NOISE[i, j, k] / 2.0 + GROUND_COLOR, 0, 1)

def apply_roof_color (pt) :
    i, j, k = [int(127 * _) for _ in pt]
    return clamp_elem_wise_np(FRACTAL_NOISE[i, j, k] / 2.0 + ROOF_COLOR, 0, 1)

clear_all()
node = run_derivation(prods, 'single_home', FOOTPRINT)
leaf_nodes = leaves(node)

all_objs = [_.draw(geometry_registry) for _ in leaf_nodes]

wall_objs = [_ for n, _ in zip(leaf_nodes, all_objs) if n.id == 'wall_literal']
ground_objs = [_ for n, _ in zip(leaf_nodes, all_objs) if n.id == 'ground_literal']
roof_objs = [_ for n, _ in zip(leaf_nodes, all_objs) if n.id == 'roof_literal'] 
smoke_objs = [_ for n, _ in zip(leaf_nodes, all_objs) if n.id == 'smoke_literal'] 

print('Number of smoke objs = ', len(smoke_objs))

for obj in wall_objs : 
    subdivide_n(obj, 3)
    
for obj in ground_objs : 
    subdivide_n(obj, 3)
 
if APPLY_IVY :
    print('Applying ivy on roof')
    for obj in roof_objs :
        ivy.apply(obj)
    
sky_lighting.add_lighting()
 
solid_texture_objects(wall_objs, apply_wall_color)
solid_texture_objects(ground_objs, apply_ground_color)
solid_texture_objects(roof_objs, apply_roof_color)

base = Scope(
    3, 
    np.array([-50, -50, -0.5]),
    np.eye(3),
    [100, 100, 0.5]
)
base_obj = base.draw()
GROUND_MAT.apply(base_obj)
# grass.apply(base_obj)

configure_renderer()
camera_obj = bpy.data.objects['Camera']
current_location = Vector((-35.2287, -5.5878, 24.3083))
radius = np.sqrt(current_location.x ** 2 + current_location.y ** 2)  
thetas = np.linspace(0, 2 * np.pi, 250, endpoint=False)
camera_trajectory = [(2 + radius * np.cos(_), 2 + radius * np.sin(_), current_location.z) for _ in thetas]

for i in range(1, 250) : 
    look_direction = normalized(np.array([2,2,7]) -np.array(camera_trajectory[i]))
    place_camera_insert_key_frame(camera_obj, camera_trajectory[i], look_direction, [0,0,1], i)

if "--" not in sys.argv:
    argv = []  
else:
    argv = sys.argv[sys.argv.index("--") + 1:]  

parser = argparse.ArgumentParser(description='Run script')
parser.add_argument('--out_path', type=str, help='out path')
args = parser.parse_args(argv)
bpy.ops.wm.save_as_mainfile(filepath=args.out_path)
