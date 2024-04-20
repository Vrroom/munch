import bpy
from mathutils import Vector, Matrix
import numpy as np
import sys
import os
from utils import *
from drawTools import *
from roofTools import *

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
    mat = np.stack([x, y, z]).T

    euler_angles = rotation_to_euler_angles(mat)
    camera_obj.location = Vector(location) 
    camera_obj.rotation_euler = euler_angles
    camera_obj.keyframe_insert(data_path="location", frame=frame)
    camera_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

def set_render_engine (engine='CYCLES') :
    bpy.context.scene.render.engine = engine

def set_render_samples (samples=1024):
    bpy.context.scene.cycles.samples = samples

