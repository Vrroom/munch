import bpy
import bmesh
from drawTools import *

def mansard_roof (width, length, height, point_frac, name='mansard') :
    x1 = width * point_frac
    x2 = width * (1 - point_frac)
    y1 = length * point_frac
    y2 = length * (1 - point_frac)
    verts = [
        (0, 0, 0),
        (width, 0, 0),
        (width, length, 0),
        (0, length, 0), 
        (x1, y1, height),
        (x1, y2, height),
        (x2, y2, height),
        (x2, y1, height)
    ]
    faces = [
        [0, 1, 7, 4],
        [0, 4, 5, 3],
        [3, 5, 6, 2],
        [1, 2, 6, 7],
        [4, 5, 6, 7]
    ]
    obj = make_mesh(verts, faces, name=name)
    subdivide_mesh_n(obj)
    return obj
    
def hipped_roof (width, length, height, point_frac, name='hipped') : 
    x1 = length * point_frac
    x2 = length * (1 - point_frac)
    verts = [
        (0, 0, 0),
        (width, 0, 0), 
        (width, length, 0), 
        (0, length, 0),
        (width / 2, x1, height),
        (width / 2, x2, height)
    ]
    faces = [
        [0, 1, 4],
        [2, 5, 3], 
        [0, 4, 5, 3],
        [1, 4, 5, 2]
    ]
    obj = make_mesh(verts, faces, name=name)
    subdivide_mesh_n(obj)
    return obj

def gabled_roof (width, length, height, name='gabled') : 
    return hipped_roof(width, length, height, 0, name=name)

