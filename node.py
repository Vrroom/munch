from more_itertools import flatten
import random
from drawTools import * 
from uuid import uuid4
import numpy as np

class Node () :
    
    def __init__ (self, id, scope) : 
        self.parent = None
        self.children = []
        self.id = id 
        self.unique_id = uuid4()
        self.scope = scope
        
    def __lt__ (self, that) : 
        return False
    
    def __eq__ (self, that) : 
        return True

    def is_root(self) : 
        return self.parent is None
    
    def is_active (self) : 
        return len(self.children) == 0  
    
    def is_leaf (self) : 
        return self.is_active()

    def __repr__ (self, level=0) : 
        ret = "\t" * level + repr(self.id) + "\n"
        for child in self.children: 
            ret += child.__repr__(level + 1)
        return ret

    def draw(self, geometry_registry=None, **kwargs) : 
        if self.id == 'epsilon' : 
            # Dummy axiom that means don't do anything
            return None
        material = None
        if geometry_registry is not None: 
            if self.id in geometry_registry : 
                if 'object' in geometry_registry[self.id]  :
                    obj = geometry_registry[self.id]['object']
                    copy = duplicate_object(self.id, obj)
                    set_visible(copy)
                    fit_in_scope(copy, self.scope)
                    return copy
                elif 'objects' in geometry_registry[self.id] : 
                    if 'any' in geometry_registry[self.id]: 
                        id = random.randint(0, len(geometry_registry[self.id]['objects']) - 1)
                        obj = geometry_registry[self.id]['objects'][id]
                        copy = duplicate_object(self.id, obj)
                        set_visible(copy)
                        fit_in_scope(copy, self.scope)
                    elif 'fix' in geometry_registry[self.id] :
                        L = len(geometry_registry[self.id]['objects'])
                        # hack to cache the literal we are placing
                        if 'id' in geometry_registry[self.id] : 
                            id = geometry_registry[self.id]['id']
                        else :
                            id = random.randint(0, L - 1)
                            geometry_registry[self.id]['id'] = id
                        obj = geometry_registry[self.id]['objects'][id]
                        copy = duplicate_object(self.id, obj)
                        set_visible(copy)
                        fit_in_scope(copy, self.scope)
                    else :
                        scope_aspect = self.scope.size[1] / self.scope.size[0]
                        cand_bounds = [bounding_box(_.data) for _ in geometry_registry[self.id]['objects']]
                        cand_aspects = [(M[1] - m[1]) / (M[0] - m[0]) for m, M in cand_bounds]
                        diffs = [abs(a - scope_aspect) for a in cand_aspects]
                        obj = geometry_registry[self.id]['objects'][np.argmin(diffs)]
                        copy = duplicate_object(self.id, obj)
                        set_visible(copy)
                        fit_in_scope(copy, self.scope)
                    return copy

                if 'material' in geometry_registry[self.id] :
                    material = geometry_registry[self.id]['material']
        obj = self.scope.draw(self.id, **kwargs)
        if material is not None: 
            apply_material_on_obj(obj, material)
        return obj
    
def leaves(root) : 
    if root.is_leaf() : 
        return [root]
    else : 
        return list(flatten(map(leaves, root.children)))

def all_nodes (root) : 
    return [root] + list(flatten(map(all_nodes, root.children)))
    
def add_links (parent, child) : 
    parent.children.append(child)
    child.parent = parent

def no_parent (node) : 
    """ returns all nodes that are not an ancestor """ 
    ancs = []
    root = node
    while not root.is_root() : 
        ancs.append(root)
        root = root.parent
    ancs.append(root) 
    nodes = all_nodes(root)
    node_ids = [n.unique_id for n in ancs]
    no_ancs = [n for n in nodes if n.unique_id not in node_ids]
    return no_ancs

def check_intersect (node, node_list) : 
    for that in node_list : 
        if node.scope.approx_intersect(that.scope) and that.id != 'epsilon': 
            return True
    return False

def check_contained (node, node_list) : 
    for that in node_list : 
        if node.scope.contained_in(that.scope) and that.id != 'epsilon': 
            return True
    return False

def find_root (node) : 
    while not node.is_root() : 
        node = node.parent
    return node

def get_ancs (node) : 
    ancs = []
    root = node
    while not root.is_root() : 
        ancs.append(root)
        root = root.parent
    ancs.append(root) 
    return ancs
