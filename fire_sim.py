from infinigen.assets.lighting import sky_lighting
from infinigen.assets.scatters import grass, ivy
from infinigen.assets.materials import ice, bark_birch, dirt, lava, mud, sand, sandstone
from infinigen.assets.fluid import liquid_particle_material
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

def prepare_vol_material (material_name) : 
    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True

    # Get the node tree for this material
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    # Clear existing nodes
    nodes.clear()

    # Add the Principled Volume node
    principled_volume = nodes.new(type='ShaderNodeVolumePrincipled')
    principled_volume.location = (-200, 200)

    # Set the properties for Principled Volume
    principled_volume.inputs['Color'].default_value = (1.0, 0.0, 0.0, 1)  # red
    principled_volume.inputs['Density'].default_value = 20
    principled_volume.inputs['Anisotropy'].default_value = 0.8
    principled_volume.inputs['Absorption Color'].default_value = (0, 0, 0, 1)  # Black
    principled_volume.inputs['Emission Color'].default_value = (1.0, 0.5, 0.0, 1)  # orange
    principled_volume.inputs['Blackbody Intensity'].default_value = 1.0
    principled_volume.inputs['Blackbody Tint'].default_value = (1, 1, 1, 1)  # White
    principled_volume.inputs['Temperature'].default_value = 1000

    # Add the Material Output node
    material_output = nodes.new(type='ShaderNodeOutputMaterial')
    material_output.location = (200, 200)   

    # Connect the Principled Volume node to the Material Output node (Volume input)
    links.new(principled_volume.outputs['Volume'], material_output.inputs['Volume']) 
    return mat    

def make_fire_in_scope (scope, domain) : 
    # Prepare objects
    side = min(scope.size[0], scope.size[1])
    radius = side * 0.9 / 2
    location = (
        scope.global_x + (scope.global_c[:, 0] * scope.size[0] / 2) \
        + (scope.global_c[:, 1] * scope.size[1] / 2) \
        + (scope.global_c[:, 2] * (radius + 0.01))
    ).tolist()
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius, 
        enter_editmode=False, 
        align='WORLD',
        location=location, 
        scale=(1, 1, 1)
    )
    sphere = bpy.context.active_object
    
    # add fluid modifiers
    with active_object_context(domain) : 
        bpy.ops.object.modifier_add(type='FLUID')
        bpy.context.object.modifiers["Fluid"].fluid_type = 'DOMAIN'
        bpy.context.object.modifiers["Fluid"].domain_settings.use_noise = True
        bpy.context.object.modifiers["Fluid"].domain_settings.cache_type = 'ALL'

    with active_object_context(sphere) :
        bpy.ops.object.modifier_add(type='FLUID')
        bpy.context.object.modifiers["Fluid"].fluid_type = 'FLOW'
        bpy.context.object.modifiers["Fluid"].flow_settings.flow_type = 'BOTH'
        bpy.context.object.modifiers["Fluid"].flow_settings.flow_behavior = 'INFLOW'
        
    # prepare material
    mat = prepare_vol_material('vol_mat')
    
    with active_object_context(domain) : 
        bpy.context.active_object.data.materials.append(mat)
        
    cache_dir = f'/tmp/{random_string()}' 
    
    os.makedirs(cache_dir, exist_ok=True)
    with active_object_context(domain) : 
        fluid_modifier = domain.modifiers["Fluid"]
        fluid_domain_settings = fluid_modifier.domain_settings
        fluid_domain_settings.cache_directory = cache_dir
        fluid_domain_settings.cache_type = 'ALL'
        bpy.ops.fluid.bake_all()

scope = Scope(
    3, 
    np.zeros(3), 
    np.eye(3), 
    [2,2,10]
)
obj = scope.draw()
make_fire_in_scope(scope, obj)
