from infinigen.assets.lighting import sky_lighting
from infinigen.assets.scatters import grass, ivy
from infinigen.assets.materials import ice, bark_birch, dirt, lava, mud, sand, sandstone
from infinigen.assets.fluid import liquid_particle_material
from infinigen.assets.weather import kole_clouds
import bpy
from mathutils import vector, matrix
import numpy as np
import sys
import os
sys.path.append(os.getcwd()) 
import utils
import drawtools
import rooftools
from importlib import reload
reload(utils)
reload(drawtools)
reload(rooftools)
from utils import *
from drawtools import *
from rooftools import *

def duplicate_camera(name, original_camera):
    # copy the camera data
    new_cam_data = original_camera.data.copy()

    # create a new camera object
    new_camera = bpy.data.objects.new(name=name, object_data=new_cam_data)

    # link the new camera to the same collection as the original
    collection = original_camera.users_collection[0]  # assumes the camera is linked to at least one collection
    collection.objects.link(new_camera)
    
    new_camera.matrix_world = original_camera.matrix_world.copy()

    return new_camera
    
def place_camera_insert_key_frame (camera_obj, location, look_direction, up, frame) : 
    y = normalized(np.array(up))
    z = normalized(-np.array(look_direction))
    x = normalized(np.cross(-z, y))
    y = np.cross(z, x)
    mat = np.stack([x, y, z]).t

    euler_angles = rotation_to_euler_angles(mat)
    camera_obj.location = vector(location) 
    camera_obj.rotation_euler = euler_angles
    camera_obj.keyframe_insert(data_path="location", frame=frame)
    camera_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

sky_lighting.add_lighting()

bpy.context.scene.render.engine = 'cycles'
bpy.context.scene.cycles.samples = 1024

#kole_clouds.add_kole_clouds()
mud.apply(bpy.context.active_object)

#hipped = hipped_roof(3, 5, 2, 0.2)
#ivy.apply(bpy.context.active_object)
#wood.apply(bpy.context.active_object)
#liquid_particle_material.apply(bpy.context.active_object)

#camera_obj = bpy.data.objects['camera']
#camera_obj = duplicate_camera('sub_cam', camera_obj) 

#def get_camera_look_direction(camera):
#    # the camera's negative z-axis indicates the direction it is looking
#    look_dir = camera.matrix_world.to_3x3() @ vector((0.0, 0.0, -1.0))
#    return look_dir.normalized()
# 
#def get_camera_up(camera):
#    up = camera.matrix_world.to_3x3() @ vector((0.0, 1.0, 0.0))
#    return up.normalized()

#current_location = camera_obj.location
#radius = np.sqrt(current_location.x ** 2 + current_location.y ** 2)  
#thetas = np.linspace(0, 2 * np.pi, 250, endpoint=false)
#camera_trajectory = [(radius * np.cos(_), radius * np.sin(_), current_location.z) for _ in thetas]

#up = get_camera_up(camera_obj)
#init_look_dir = get_camera_look_direction(camera_obj)

##for i in range(1, 250) :
##    look_direction = normalized(-np.array(camera_trajectory[i]))
##    place_camera_insert_key_frame(camera_obj, camera_trajectory[i], look_direction, [0,0,1], i)
