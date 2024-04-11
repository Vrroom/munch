import bpy 
import numpy as np
import math
from functools import partial 
import bmesh
import mathutils
from importlib import reload
import sys
import os
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

LAYOUT = np.array(Image.open('munch.png').convert('RGB'))
WATER = np.array([0, 183, 239])
HOUSE = np.array([255, 255, 255]) 
ROAD  = np.array([156, 90, 60])
CONNECTOR = np.array([237,  28,  36])

def get_plots (img) : 
    ccs = connected_components(img)
    shapes = []
    for cc in ccs :
        x, X = min([_[0] for _ in cc]), max([_[0] for _ in cc]) + 1
        y, Y = min([_[1] for _ in cc]), max([_[1] for _ in cc]) + 1
        shapes.append(BBox(x, y, X, Y, X - x, Y - y))
    shapes = [s for s in shapes if not s.isDegenerate()]
    return shapes

def draw_dome_enclosing_objects (objs) : 
    mm, MM = bounding_box_object_list(objs)
    radius = 1.2 * max([a - b for a, b in zip(MM, mm)]) 
    center = ((MM[0] - mm[0]) / 2, (MM[1] - mm[1]) / 2, 0)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=center)
    return bpy.context.object

def place_camera (camera_obj, location, look_direction, up) : 
    camera_obj.location = Vector(location) 
    look_direction = Vector(look_direction).normalized()
    up = Vector(up).normalized()
    rot_quat = look_direction.to_track_quat('-Z', 'Y')
    up_quat = Vector((0, 1, 0)).rotation_difference(up)
    final_quat = up_quat @ rot_quat
    camera_obj.rotation_euler = final_quat.to_euler()
    
def place_camera_insert_key_frame (camera_obj, location, look_direction, up, frame) : 
    camera_obj.location = Vector(location) 
    look_direction = Vector(look_direction).normalized()
    up = Vector(up).normalized()
    rot_quat = look_direction.to_track_quat('-Z', 'Y')
    up_quat = Vector((0, 1, 0)).rotation_difference(up)
    final_quat = up_quat @ rot_quat
    camera_obj.rotation_euler = final_quat.to_euler()
    camera_obj.keyframe_insert(data_path="location", frame=frame)
    camera_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

house_bitmap = np.all(LAYOUT == HOUSE, axis=2)
road_bitmap = np.all(LAYOUT == ROAD, axis=2)
connector_bitmap = np.all(LAYOUT == CONNECTOR, axis=2)
water_bitmap = np.ones_like(house_bitmap, dtype=bool)

#walkable_bitmap = road_bitmap | connector_bitmap
#Y, X = np.where(walkable_bitmap)
#sx, sy = X[2], Y[2]
#print(sx, sy)
#def walking_force_field () : 
#    pass

###############################################################
## ROUGH LAYOUT
###############################################################
house_objs = draw_plots_3d(get_plots(house_bitmap), 5)
road_objs = draw_plots_2d(get_plots(road_bitmap))
connector_objs = draw_plots_2d(get_plots(connector_bitmap))
water_objs = draw_plots_2d(get_plots(water_bitmap), -5)

###############################################################
## PARTICLE SYSTEM
###############################################################
[add_particle_system(_) for _ in water_objs]

###############################################################
## ENCLOSING DOME 
###############################################################
noise = generate_fractal_noise_3d(
    (128, 128, 128), (4, 4, 4), 1, tileable=(True, True, True)
)
def apply_noise(pt) : 
    i, j, k = [int(127 * _) for _ in pt]
    return (noise[i, j, k], 0, noise[i, j, k], 1.0)
dome = draw_dome_enclosing_objects(house_objs + road_objs + connector_objs + water_objs)
solid_texture_object(dome, apply_noise, as_emission=True)

#place_camera(bpy.data.objects['Camera'], (0, 0, 0), (1, 0, 0), (0, 1, 0))
clear_all()    


FOOTPRINT = Scope(3, np.array([0, 0, 0]), np.eye(3), [6, 6, 6]) 
ALWAYS = lambda *args, **kwargs: True
def FACING_POS_X (node, *args, **kwargs) : 
    x = np.array([1.0, 0.0, 0.0])
    outward_vec = np.cross(node.scope.global_c[:, 0], node.scope.global_c[:, 1])
    return np.isclose(np.dot(x, outward_vec), 1.0)
DOOR_WIDTH = 1
DOOR_HEIGHT = 1.5
DOOR_DEPTH = .1
WINDOW_SPACING = 1.7
WINDOW_WIDTH = .4
WINDOW_HEIGHT = .8
WINDOW_DEPTH = .1
WALL_DEPTH = .1
WINDOW_OBJ = load_blend_model('assets/window.blend', 'wind_low')
DOOR_OBJ = load_blend_model('assets/door.blend', 'Mansion Door')
BRICK_OBJ = load_blend_model('assets/single_brick.blend', 'single_brick')
ROOF_TILE_OBJ = load_blend_model('assets/roof.blend', 'roof_tile') 
#BRICK_MAT = import_material_from_file('assets/brick.blend', 'brick_material')
BRICK_WIDTH = 0.2
BRICK_HEIGHT = 0.07
BRICK_COLOR = [1.0, 0.69, 0.38, 1.0]
ROOF_TILE_SZ = 1.0

multi_storey = Production(
    id='0',
    priority=0, 
    pred='multi_storey',
    cond=ALWAYS,
    scope_modifiers=[
        partial(repeat, axis=2, val=2)
    ],
    succ=[['ground_floor', 'floor', 'top_floor']],
    prob=[1.0]
)

ground_floor = Production(
    id='1', 
    priority=1,
    pred='ground_floor', 
    cond=ALWAYS, 
    scope_modifiers=[identity], 
    succ=[['ground_floor_facades']], 
    prob=[1.0]
)

def custom_scope_modifier (scope) : 
    return scope, scale_scope(scale_scope(scope, 0.5, 0), 1.5, 1)

floor = Production(
    id='1', 
    priority=1,
    pred='floor', 
    cond=ALWAYS, 
#    scope_modifiers=[identity],
    scope_modifiers=[custom_scope_modifier], 
#    succ=[['facades'],
    succ=[['facades', 'facades']], 
    prob=[1.0]
)

top_floor = Production(
    id='1', 
    priority=1,
    pred='top_floor', 
    cond=ALWAYS, 
    scope_modifiers=[
        partial(subdiv, axis=2, args=[rfloat(0.8), rfloat(0.2)])
    ],
    succ=[['facades', 'roof']], 
    prob=[1.0]
)

#house = Production(
#    id='1', 
#    priority=1,
#    pred='house',
#    cond=ALWAYS,
#    scope_modifiers=[
#        partial(subdiv, axis=2, args=[rfloat(0.95), rfloat(0.05)])
#    ],
#    succ=[['facades', 'roof']],
#    prob=[1.0]
#)

roof = Production(
    id='-1', 
    priority=3, 
    pred='roof', 
    cond=ALWAYS, 
    scope_modifiers=[
#        partial(subscope, sx=rfloat(1.02), sy=rfloat(1.02), sz=rfloat(1.2), tx=-0.06, ty=-0.06, tz=0.0)
    ],
#    succ=[['expanded_roof']],
    succ=[],
    prob=[],#1.0]
)

expanded_roof = Production(
    id='-2',
    priority=3,
    pred='expanded_roof', 
    cond=ALWAYS,
    scope_modifiers=[
        compose(
            partial(repeat, axis=0, val=ROOF_TILE_SZ),
            partial(map, partial(repeat, axis=1, val=ROOF_TILE_SZ)),
            flatten,
            list
        )
    ],
    succ=[['roof_tile']],
    prob=[1.0]
)

roof_tile = Production(
    id='-3',
    priority=3,
    pred='roof_tile', 
    cond=ALWAYS,
    scope_modifiers=[],
    succ=[],
    prob=[]
)             

ground_floor_facades = Production(
    id='1', 
    priority=2, 
    pred='ground_floor_facades', 
    cond=ALWAYS, 
    scope_modifiers=[partial(comp, split_type=ComponentSplitType.SIDE_FACES)],
    succ=[['ground_floor_facade']],
    prob=[1.0]
)

facades = Production(
    id='1', 
    priority=2, 
    pred='facades', 
    cond=ALWAYS, 
    scope_modifiers=[partial(comp, split_type=ComponentSplitType.SIDE_FACES)],
    succ=[['facade']],
    prob=[1.0]
)

ground_floor_facade = Production(
    id='2', 
    priority=2, 
    pred='ground_floor_facade', 
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

ground_floor_facade_2 = Production(
    id='14', 
    priority=2, 
    pred='ground_floor_facade', 
    cond=complement(FACING_POS_X),
    scope_modifiers=[identity],
    succ=[['tiles']],
    prob=[1.0]
)

facade = Production(
    id='14', 
    priority=2, 
    pred='facade', 
    cond=ALWAYS,
    scope_modifiers=[identity],
    succ=[['tiles']],
    prob=[1.0]
)

tiles = Production(
    id='3', 
    priority=2, 
    pred='tiles',
    cond=ALWAYS,
    scope_modifiers=[partial(repeat, axis=0, val=WINDOW_SPACING)],
    succ=[['tile']],
    prob=[1.0],
)

tile = Production(
    id='4',
    priority=2,
    pred='tile',
    cond=ALWAYS,
    scope_modifiers=[partial(subdiv, axis=0, args=[rfloat(1), WINDOW_WIDTH, rfloat(1)])],
    succ=[['wall', 'wall_window', 'wall']],
    prob=[1.0]
)

def OCCLUDED (node, *args, **kwargs) : 
#    return False
    return check_intersect(node, no_parent(node))

window = Production(
    id='5',
    priority=4,
    pred='window',
    cond=complement(OCCLUDED),
    scope_modifiers=[partial(extrude, sz=DOOR_DEPTH)],
    succ=[['window_literal']],
    prob=[1.0],
)

window2 = Production(
    id='213',
    priority=4,
    pred='window',
    cond=OCCLUDED,
    scope_modifiers=[identity],
    succ=[['wall']],
    prob=[1.0],
)


wall = Production(
    id='6',
    priority=2,
    pred='wall',
    cond=ALWAYS,
    scope_modifiers=[partial(extrude, sz=WALL_DEPTH)],
    succ=[['bricks']],
    prob=[1.0]
)

entrance = Production(
    id='7',
    priority=2,
    pred='entrance',
    cond=ALWAYS,
    scope_modifiers=[partial(subdiv, axis=0, args=[rfloat(1), DOOR_WIDTH, rfloat(1)])],
    succ=[['wall', 'wall_door', 'wall']],
    prob=[1.0]
)

wall_window = Production(
    id='8',
    priority=2,
    pred='wall_window',
    cond=ALWAYS,
    scope_modifiers=[partial(subdiv, axis=1, args=[rfloat(2), WINDOW_HEIGHT, rfloat(1)])],
    succ=[['wall', 'window', 'wall']],
    prob=[1.0]
)

wall_door = Production(
    id='9',
    priority=2,
    pred='wall_door',
    cond=ALWAYS,
    scope_modifiers=[partial(subdiv, axis=1, args=[DOOR_HEIGHT, rfloat(1)])],
    succ=[['door', 'wall']],
    prob=[1.0]
)

door = Production(
    id='10', 
    priority=2,
    pred='door',
    cond=ALWAYS, 
    scope_modifiers=[partial(extrude, sz=DOOR_DEPTH)],
    succ=[['door_literal']],
    prob=[1.0],
)

window_literal = Production(
    id='11',
    priority=2,
    pred='window_literal',
    cond=ALWAYS,
    scope_modifiers=[],
    succ=[],
    prob=[],
)

door_literal = Production(
    id='12',
    priority=2,
    pred='door_literal',
    cond=ALWAYS,
    scope_modifiers=[],
    succ=[],
    prob=[],
)

bricks = Production(
    id='13',
    priority=2,
    pred='bricks',
    cond=ALWAYS,
    scope_modifiers=[
        compose(
            partial(repeat, axis=0, val=BRICK_WIDTH),
            partial(map, partial(repeat, axis=1, val=BRICK_HEIGHT)),
            flatten,
            list,
            partial(map, partial(subscope, sx=rfloat(0.95), sy=rfloat(0.95), sz=rfloat(0.95), tx=0, ty=0, tz=0)), 
            flatten,
            list
        )
    ],
    succ=[['brick']],
    prob=[1.0],
)

brick = Production(
    id='15',
    priority=2,
    pred='brick',
    cond=ALWAYS,
    scope_modifiers=[],
    succ=[],
    prob=[],
)

geometry_registry = dict(
    window_literal=dict(object=WINDOW_OBJ),
    door_literal=dict(object=DOOR_OBJ),
    brick=dict(object=BRICK_OBJ),
    roof_tile=dict(object=ROOF_TILE_OBJ),
    roof=dict(object=mansard_roof (10, 10, 2, 0.2, name='mansard'))
#    wall_literal=dict(material=BRICK_MAT)
)

for k in geometry_registry.keys() : 
    if 'object' in geometry_registry[k] : 
        set_invisible(geometry_registry[k]['object'])
 
all_prods = [
    multi_storey,
    floor,
    #ground_floor, floor, top_floor,
    ground_floor_facades, facades,
    ground_floor_facade, ground_floor_facade_2,
    facade,
    tiles, tile,
    window, window2, wall, entrance,
    wall_door, door, wall_window,
    window_literal, door_literal,
#    expanded_roof, roof_tile, roof
    roof
]

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(42)

set_material_color('Procedural Curved Pottery Clay', BRICK_COLOR)

node = run_derivation(all_prods, 'multi_storey', FOOTPRINT)
clear_all()

#print(len(no_parent(node.children[0])))
#FOOTPRINT.draw()
#scale_scope(FOOTPRINT, 0.5, 0).draw()
#print(node)
[_.draw(geometry_registry) for _ in leaves(node)]
