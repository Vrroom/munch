import bpy
from functools import reduce
import numpy as np
import bmesh
from mathutils import Vector, Matrix
from utils import *
from scope import Scope
from contextlib import contextmanager

def make_mesh (verts, faces, name='mesh') :
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh) 
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.new()
    bm.from_mesh(mesh)
    for vert in verts:
        bm.verts.new(vert)
    bm.verts.ensure_lookup_table()
    for f in faces: 
        bm.faces.new([bm.verts[i] for i in f])
    bpy.ops.object.mode_set(mode='OBJECT')
    bm.to_mesh(mesh)
    bm.free()
    return obj

def min_tuple (ta, tb) : 
    return tuple([min(a, b) for a, b in zip(ta, tb)])

def max_tuple (ta, tb) : 
    return tuple([max(a, b) for a, b in zip(ta, tb)])

def bounding_box_object_list (obj_list) : 
    bbs = [bounding_box(_.data) for _ in obj_list] 
    mms = [a for a, b in bbs] 
    MMs = [b for a, b in bbs] 
    return reduce(min_tuple, mms), reduce(max_tuple, MMs)

def bounding_box (mesh) : 
    mx, my, mz = np.inf, np.inf, np.inf
    Mx, My, Mz = -np.inf, -np.inf, -np.inf
    for poly in mesh.polygons: 
        for idx in poly.loop_indices: 
            vertex_index = mesh.loops[idx].vertex_index
            vertex = mesh.vertices[vertex_index]
            mx, my, mz = min(vertex.co.x, mx), min(vertex.co.y, my), min(vertex.co.z, mz)
            Mx, My, Mz = max(vertex.co.x, Mx), max(vertex.co.y, My), max(vertex.co.z, Mz)
    return (mx, my, mz), (Mx, My, Mz) 

def solid_texture_object(obj, texture_fn=lambda *args, **kwargs : (1.0, 0.0, 0.0, 1.0), as_emission=False): 
    mesh = obj.data 
    if not mesh.vertex_colors: 
        mesh.vertex_colors.new()
    color_layer = mesh.vertex_colors.active 
    m, M = bounding_box(mesh)
    scale_factor = max([a - b for a, b in zip(M, m)])
    for poly in mesh.polygons: 
        for idx in poly.loop_indices: 
            vertex_index = mesh.loops[idx].vertex_index
            vertex = mesh.vertices[vertex_index]
            px = (vertex.co.x, vertex.co.y, vertex.co.z)
            px = [(a - b) / scale_factor for a, b in zip(px, m)] 
            color_layer.data[idx].color = texture_fn(px)
    material_name = f'{obj.name}_mat'
    add_vertex_color_as_material(obj, material_name, as_emission)
    shade_smooth(obj) 
            
def add_vertex_color_as_material (obj, material_name='VertexColorMaterial', as_emission=False) : 
    mat = bpy.data.materials.new(name=material_name)
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    attr_node = nodes.new(type='ShaderNodeAttribute')
    attr_node.attribute_name = 'Col'
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(attr_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    if as_emission : 
        links.new(attr_node.outputs['Color'], bsdf_node.inputs['Emission'])
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

@contextmanager
def active_object_context(obj):
    """ ChatGPT """
    prev_active_obj = bpy.context.view_layer.objects.active
    prev_selected_objs = [o for o in bpy.context.selected_objects]
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        yield  
    finally:
        bpy.ops.object.select_all(action='DESELECT')
        for o in prev_selected_objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = prev_active_obj

def shade_smooth (obj) : 
    with active_object_context(obj) : 
        bpy.ops.object.shade_smooth() 

def clear_all (): 
    for obj in bpy.context.scene.objects: 
        if obj.type not in {'CAMERA', 'LIGHT'}:
            obj.select_set(True)
        else:
            obj.select_set(False)          
    bpy.ops.object.delete()

def draw_plots_2d (plots, z=0) : 
    scopes = [] 
    for plot in plots :
        scopes.append(Scope(
            2,
            np.array([plot.x, plot.y, z]),
            np.eye(3)[:, :2],
            np.array([plot.w, plot.h])
        ))
    objs = []
    for scope in scopes : 
        objs.append(scope.draw())
    return objs

def add_particle_system(obj, show_emitter=False) : 
    with active_object_context(obj) : 
        bpy.ops.object.modifier_add(type='PARTICLE_SYSTEM')
        psys = obj.modifiers[-1].particle_system
        psys.settings.effector_weights.gravity = -1.0
    if not show_emitter :
        emitter_invisible(obj)

def emitter_invisible(obj):
    obj.show_instancer_for_render = False
    obj.show_instancer_for_viewport = False
    obj.data.update()

def draw_plots_3d (plots, height=2) : 
    scopes = [] 
    for plot in plots :
        scopes.append(Scope(
            3,
            np.array([plot.x, plot.y, 0]),
            np.eye(3),
            np.array([plot.w, plot.h, height])
        ))
    objs = []
    for scope in scopes : 
        objs.append(scope.draw())
    return objs
        
def set_visible(obj, scene=bpy.context.scene):
    """ ChatGPT """
    if obj.name in scene.objects:
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.hide_render = False

def set_invisible(obj, scene=bpy.context.scene):
    """ ChatGPT """
    if obj.name in scene.objects:
        obj.hide_set(True)
        obj.hide_viewport = True
        obj.hide_render = True

def load_blend_model(blend_file_path, object_name):
    """ ChatGPT """ 
    for obj in bpy.context.scene.objects:
        if obj and obj.name == object_name:
            return obj

    with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
        if object_name in data_from.objects:
            data_to.objects.append(object_name)

    for obj in data_to.objects:
        if obj and obj.name == object_name:
            bpy.context.collection.objects.link(obj)
            obj.select_set(True)
            return obj

    raise RuntimeError("Didn't find the given object") 

def duplicate_object(name, copy_object):
    """ ChatGPT """ 
    scene = bpy.context.scene
    mesh_data = copy_object.data.copy()
    new_object = bpy.data.objects.new(name, mesh_data)
    scene.collection.objects.link(new_object)
    new_object.matrix_world = copy_object.matrix_world
    return new_object

def get_obj_bounds (obj) : 
    local_bbox_corners = [Vector(_) for _ in obj.bound_box] 
    world_bbox_corners = [obj.matrix_world @ _ for _ in local_bbox_corners] 
    min_coord = Vector((min(_.x for _ in world_bbox_corners),
                        min(_.y for _ in world_bbox_corners),
                        min(_.z for _ in world_bbox_corners)))
    max_coord = Vector((max(_.x for _ in world_bbox_corners),
                        max(_.y for _ in world_bbox_corners),
                        max(_.z for _ in world_bbox_corners)))
    return min_coord, max_coord

def draw_line(location, x, scale, name=None):
    """ 
    Adapted from ChatGPT
    """ 
    name = "line" if name is None else name
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh) 
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    v1 = bm.verts.new(location)
    v2 = bm.verts.new(location + x.reshape(-1) * scale[0])
    bm.edges.new([v1, v2])
    bm.to_mesh(mesh)
    bm.free()
    return obj
    
def draw_cuboid(location, xyz, scale, name=None):
    scale = [_ / 2 for _ in scale]
    corner = (xyz @ (np.array([-a for a in scale]).reshape(3, 1))).reshape(-1)
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.rotation_euler = rotation_to_euler_angles(xyz)
    obj.scale = scale
    obj.location = (-corner + np.array(location)).tolist()
    obj.name = name if name is not None else 'Cube'
    return obj

def draw_rectangle (location, xy, scale, name=None) : 
    name = "rectangle" if name is None else name
    loc = np.array(location)
    coords = [
        loc, 
        loc + scale[0] * xy[:, 0], 
        loc + scale[0] * xy[:, 0] + scale[1] * xy[:, 1], 
        loc + scale[1] * xy[:, 1]
    ]
    
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    bm = bmesh.new()
    bm.from_mesh(mesh)
    
    for coord in coords:
        bm.verts.new(coord.tolist())
        
    bm.verts.ensure_lookup_table()
    
    bm.edges.new([bm.verts[0], bm.verts[1]])
    bm.edges.new([bm.verts[1], bm.verts[2]])
    bm.edges.new([bm.verts[2], bm.verts[3]])
    bm.edges.new([bm.verts[3], bm.verts[0]])
    
    bm.faces.new([bm.verts[0], bm.verts[1], bm.verts[2], bm.verts[3]])
    
    bm.to_mesh(mesh)
    bm.free()
    
    return obj

def apply_matrix_to_mesh_obj (obj, mat) : 
    """ mat is 4 by 4 homogeneous matrix """ 
    if obj.type != "MESH": 
        raise ValueError(f'Don\'t know what to do with object type = {obj.type}')

    blender_mat = Matrix(mat.tolist())
    for vert in obj.data.vertices: 
        vert.co = blender_mat @ vert.co
    obj.data.update()

def fit_in_scope(obj, scope):
    v_min, v_max = get_obj_bounds(obj) 
    delta = v_max - v_min 

    o_dx = np.array([delta[0], delta[1], delta[2]])
    s_dx = np.array([_ for _ in scope.size])

    apply_matrix_to_mesh_obj(obj, homogenize_scale(s_dx / o_dx))

    apply_matrix_to_mesh_obj(obj, homogenize_rotation_matrix(scope.global_c))

    apply_matrix_to_mesh_obj(obj, homogenize_translation(scope.global_x))

    return obj

def import_material_from_file (file_name, material_name) : 
    if material_name in bpy.data.materials : 
        return bpy.data.materials[material_name]

    with bpy.data.libraries.load(file_name) as (data_from, data_to):
        data_to.materials = data_from.materials

    return bpy.data.materials[material_name]

def apply_material_on_obj (obj, mat) : 
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


def set_brick_spacing (mat, val) :
    mat.node_tree.nodes['Brick Texture'].inputs['Scale'].default_value = val

def set_material_color (material_name, color) : 
    bpy.data.materials[material_name].node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = color

#xyz = np.array([
#    [0.707106, -0.707106, 0],
#    [0.707106, 0.707106, 0],
#    [0., 0., 1.],
#])
#xy = np.array([
#    [0.707106, -0.707106],
#    [0.707106, 0.707106],
#    [0., 0.],
#])
#obj = draw_rectangle((1, 1, 1), xy, (3, 2))
#obj.matrix_world = [[3, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
#print(obj.matrix_world)
#obj.rotation_euler = [0, 0, math.pi/4]
#Rotation.from_matrix(
#print(obj.location)
##print(obj.rotation_euler)
#print(obj.matrix_world)
#print(obj.scale)
#sqt_2 = 1 / math.sqrt(2)
