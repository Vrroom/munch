from enum import Enum 
from itertools import cycle
from copy import deepcopy 
import numpy as np 
import math
from more_itertools import flatten
from functools import reduce, wraps
from drawTools import *
from utils import *

EPS = 1e-5

def rotate_scope (scope, R) :
    new_scope = deepcopy(scope)
    new_scope.global_c = R @ scope.global_c
    return new_scope

def scope_center (scope)  :
    return ((scope.global_c @ np.diag(scope.size)).sum(axis=1) / 2.) + scope.global_x

def scale_scope (scope, scale_factor, axis) :
    assert scope.dim > axis, (
        f'Mismatch in dim and scaling axis '
        f'dim = {scope.dim}, axis = {axis}'
    )
    new_scope = deepcopy(scope)
    new_scope.global_x = new_scope.global_x + (new_scope.global_c[:, axis] * (1. - scale_factor) * new_scope.size[axis] / 2.)
    new_scope.size[axis] *= scale_factor
    return new_scope

class ComponentSplitType(Enum):
    FACES = 'faces'
    EDGES = 'edges'
    VERTICES = 'vertices'
    SIDE_FACES = 'side faces'

class rfloat (float) : 
    pass

def type_check (arr) : 
    for i in range(len(arr)) : 
        if isinstance(arr[i], int) : 
            arr[i] = float(arr[i]) 
    assert all(isinstance(_, float) or isinstance(_, rfloat) for _ in arr), (
        f'Found wierd type in arr, ',
        f'{[type(_) for _ in arr]}'
    )

def is_abs (num) :
    return isinstance(num, float) and not isinstance(num, rfloat)

def is_rel (num) :
    return isinstance(num, rfloat)

def xz (c) : 
    nc = np.copy(c)
    return nc[:, [0, 2]] 

def mxz (c) : 
    nc = np.copy(c)
    ans = nc[:, [0, 2]] 
    ans[:, 0] = -ans[:, 0]
    return ans

def xy (c) : 
    nc = np.copy(c)
    return nc[:, [0, 1]] 

def mxy (c) : 
    nc = np.copy(c)
    ans = nc[:, [0, 1]] 
    ans[:, 0] = -ans[:, 0]
    return ans

def yz (c) : 
    nc = np.copy(c)
    return nc[:, [1, 2]] 

def myz (c) : 
    nc = np.copy(c)
    ans =  nc[:, [1, 2]] 
    ans[:, 0] = -ans[:, 0]
    return ans

class Scope () : 
    
    def __init__ (self, dim, global_x, global_c, size, index=None) : 
        self.dim = dim 
        self.global_x = global_x # shape [3]
        self.global_c = global_c # shape [3, dim]
        self.size = size
        assert self.dim == self.global_c.shape[1], (
            f'Mismatch in dim and shape of global coordinate system, '
            f'dim = {self.dim}, global_c shape = {self.global_c.shape}'
        )
        assert self.dim == len(self.size), (
            f'Mismatch in dim and size, '
            f'dim = {self.dim}, size length = {len(self.size)}'
        )
        self.index = index
        self.set_aabb()

    def __repr__(self):
        return (f'Scope(dim={self.dim}, '
                f'global_x={self.global_x}, '
                f'global_c={self.global_c}, '
                f'size={self.size},'
                f'aabb={self.aabb})')
                
    def draw (self, name=None, **kwargs) : 
        if self.dim == 3 :
            return draw_cuboid(self.global_x, self.global_c, self.size, name=name, **kwargs)
        elif self.dim == 2 :
            return draw_rectangle(self.global_x, self.global_c, self.size, name=name, **kwargs)
        else :
            return draw_line(self.global_x, self.global_c, self.size, name=name, **kwargs)

    def validate (self) : 
        assert self.dim == self.global_c.shape[1], (
            f'Mismatch in dim and shape of global coordinate system, '
            f'dim = {self.dim}, global_c shape = {self.global_c.shape}'
        )
        assert self.dim == len(self.size), (
            f'Mismatch in dim and size, '
            f'dim = {self.dim}, size length = {len(self.size)}'
        )
        if self.dim == 3: 
            det = np.linalg.det(self.global_c)
            assert np.isclose(det, 1.0), (
                f'The scope is not positively oriented. It has determinant = {det}'
            )

    def __and__(self, that):
        self.set_aabb()
        that.set_aabb()
        for i in range(3): 
            if self.aabb[0][i] > that.aabb[1][i] or self.aabb[1][i] < that.aabb[0][i]:
                return False
        return True

    def shrink_aabb(self, scale_factor) : 
        self.set_aabb()
        mm, MM = self.aabb
        center = (mm + MM) / 2
        mm = center + (mm - center) * scale_factor
        MM = center + (MM - center) * scale_factor
        return mm, MM 

    def approx_intersect (self, that, scale_factor=0.9) : 
        self.set_aabb()
        that.set_aabb()
        aabb_self = self.shrink_aabb(scale_factor)
        aabb_that = that.aabb # shrink_aabb(1.)
        for i in range(3): 
            if aabb_self[0][i] > aabb_that[1][i] or aabb_self[1][i] < aabb_that[0][i]:
                return False
        return True

    def contained_in (self, that, scale_factor=0.9) : 
        self.set_aabb()
        that.set_aabb()
        aabb_self = self.shrink_aabb(scale_factor)
        aabb_that = that.shrink_aabb(1.)
        for i in range(3) : 
            ms, Ms = aabb_self[0][i], aabb_self[1][i]
            mt, Mt = aabb_that[0][i], aabb_that[1][i]
            if not (mt <= ms <= Ms <= Mt) :
                return False
        return True

    def set_aabb (self) : 
        p1 = self.global_x
        p2 = (self.global_c @ np.diag(self.size)).sum(axis=1) + self.global_x
        mm = np.minimum(p1, p2)
        MM = np.maximum(p1, p2)
        self.aabb = (mm, MM)

class ShapeScope (Scope) : 

    def __init__ (self, shape, dim, global_x, global_c, size, index=None) : 
        """
        A shape scope with dim = 2 is a planar scope. A shape scope with dim = 3 is a volume
        scope. The cross section in both remains the same 
        """
        assert dim in [2, 3], f'Can\'t create ShapeScope with dim = {dim}'
        super().__init__(dim, global_x, global_c, size, index=index)
        self.shape = shape

        assert all(x <= size[0] for x, y in shape), f'shape goes beyond x bounds'
        assert all(y <= size[1] for x, y in shape), f'shape goes beyond y bounds'

    @classmethod
    def create_from_scope (cls, shape, scope) : 
        return cls(shape, scope.dim, scope.global_x, scope.global_c, scope.size, scope.index)

    def draw (self, name=None, **kwargs) :
        if self.dim == 3 :
            verts, faces = [], []
            L = len(self.shape)
            for (x, y) in self.shape : 
                pt1 = self.global_x + (x * self.global_c[:, 0] + y * self.global_c[:, 1])
                verts.append(pt1)

            for (x, y) in self.shape :
                pt2 = self.global_x + (self.size[2] * self.global_c[:, 2]) + (x * self.global_c[:, 0] + y * self.global_c[:, 1])
                verts.append(pt2)

            for i in range(L) : 
                faces.append([i, (i + 1) % L, L + (i + 1) % L, L + i])
                faces.append([0, (i + 1) % L, (i + 2) % L])
                faces.append([L + 0, L + (i + 1) % L, L + (i + 2) % L])

            faces = [[_ % (2 * L) for _ in f] for f in faces]
            faces = [f for f in faces if len(set(f)) == len(f)]
            return make_mesh (verts, faces, name='polytope')
        elif self.dim == 2 :
            verts, faces = [], []
            L = len(self.shape)
            for (x, y) in self.shape : 
                pt1 = self.global_x + (x * self.global_c[:, 0] + y * self.global_c[:, 1])
                verts.append(pt1)

            for i in range(L) : 
                faces.append([0, (i + 1) % L, (i + 2) % L])

            faces = [[_ % (2 * L) for _ in f] for f in faces]
            faces = [f for f in faces if len(set(f)) == len(f)]
            return make_mesh (verts, faces, name='polytope')

def validate_scopes(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not isinstance(result, list):
            raise ValueError("Output is not a list")
        for _ in result :
            _.validate()
        return result
    return wrapper

def compose (*args) :
    """ 
    compose(A, B)(x) = B(A(X))
    """
    return lambda x : reduce(lambda a, fn : fn(a), args, x)

def compose_scope_modifiers (*args) :
    if len(args) == 0 :
        return identity

    def new_mod (x) :
        new_scopes = args[0](x)
        for i in range(1, len(args)) : 
            new_scopes = list(flatten(map(args[i], new_scopes)))
        return new_scopes

    return new_mod

def two_level_compose (s_mod1, s_mod_list) :
        
    def new_mod(x) : 
        print(x)
        out = s_mod1(x) 
        assert len(out) == len(s_mod_list), f'the output of first modifier should be same length as second'
        new_scopes =  []
        for os, s_mod in zip(out, s_mod_list) : 
            new_scopes.extend(s_mod(os))
        return new_scopes

    return new_mod

@validate_scopes
def flip_z (scope) : 
    new_scope = deepcopy(scope) 
    new_scope.global_x += new_scope.global_c[:, 2] * new_scope.size[2]
    new_scope.global_x += new_scope.global_c[:, 0] * new_scope.size[0]
    new_scope.global_c[:, 2] = - new_scope.global_c[:, 2]
    new_scope.global_c[:, 0] = - new_scope.global_c[:, 0]
    return [new_scope]

@validate_scopes
def identity (scope) : 
    return [scope]

@validate_scopes
def copy_n (scope, n) : 
    return [deepcopy(scope) for _ in range(n)]

@validate_scopes 
def four_corner (scope, sx=rfloat(0.5), sy=rfloat(0.5)) : 
    """
    Creates a four corner scope useful for creating pillars. For 3D scopes only

    Will cut in x y only
    """
    assert scope.dim == 3, f'Scope with dim = {scope.dim} found'  
    assert not isinstance(scope, ShapeScope), f'four_corner does not work on ShapeScope'
    x_sz = float(sx * scope.size[0]) if is_rel(sx) else sx
    y_sz = float(sy * scope.size[1]) if is_rel(sy) else sy

    new_scopes = [deepcopy(scope) for i in range(4)]

    for ns in new_scopes : 
        ns.size[0] = x_sz
        ns.size[1] = y_sz
    
    deltas = [
        (0, 0), 
        (scope.size[0] - x_sz, 0), 
        (0, scope.size[1] - y_sz), 
        (scope.size[0] - x_sz, scope.size[1] - y_sz)
    ]

    for ns, (dx, dy) in zip(new_scopes, deltas) :
        ns.global_x += dx * ns.global_c[:, 0]
        ns.global_x += dy * ns.global_c[:, 1]

    return new_scopes

@validate_scopes
def split_into_faces_using_shape (scope, shape, top_and_bottom=False) : 
    """ 
    Assume that the two d shape is in the x y plane in the coordinate axis given by
    x = scope.global_c[:, 0], y = scope.global_c[:, 1]. The shape is also assumed to be 
    inside the x and y bounds of the scope.

    Works only for 3D scopes
    shape = list of (x, y) tuples
    """
    assert scope.dim == 3, f'Scope with dim = {scope.dim} found'  
    new_scopes = [] 
    for ((x1, y1), (x2, y2)) in zip(shape, shape[1:] + shape[:1]) :
        pt1 = scope.global_x + (x1 * scope.global_c[:, 0] + y1 * scope.global_c[:, 1])
        pt2 = scope.global_x + (x2 * scope.global_c[:, 0] + y2 * scope.global_c[:, 1])
        new_x = normalized(pt2 - pt1)
        new_y = scope.global_c[:, 2]
        combined = np.stack([new_x, new_y]).T
        new_scopes.append(
            Scope(
                2, 
                pt1, 
                combined, 
                [np.linalg.norm(pt2 - pt1), scope.size[2]]
            )
        )

    if top_and_bottom : 
        new_scopes.append(
            ShapeScope(
                shape, 
                2, 
                scope.global_x, 
                scope.global_c[:, :2],
                scope.size[:2],
                index=scope.index
            )
        )
        new_scopes.append(
            ShapeScope(
                shape, 
                2, 
                scope.global_x + (scope.global_c[:, 2] * scope.size[2]), 
                scope.global_c[:, :2],
                scope.size[:2],
                index=scope.index
            )
        )


    return new_scopes

@validate_scopes
def subscope (scope, sx=rfloat(1), sy=rfloat(1), sz=rfloat(1), tx=0, ty=0, tz=0) : 
    """
    Creates a new scope from the properties of the old scope. For 3D scopes only 
    """ 
    assert scope.dim == 3, f'Scope with dim = {scope.dim} found'  
    new_scope = deepcopy(scope)
    new_scope.global_x[0] += tx
    new_scope.global_x[1] += ty
    new_scope.global_x[2] += tz
    new_scope.size[0] = float(sx * new_scope.size[0]) if is_rel(sx) else sx
    new_scope.size[1] = float(sy * new_scope.size[1]) if is_rel(sy) else sy
    new_scope.size[2] = float(sz * new_scope.size[2]) if is_rel(sz) else sz
    return [new_scope]  

@validate_scopes
def scale_center (scope, sx=rfloat(1), sy=rfloat(1), sz=rfloat(1)) :
    """ scale a 3d scope about center """ 
    assert scope.dim == 3, f'Scope with dim = {scope.dim} found'  
    center = scope_center(scope)
    new_scope = deepcopy(scope)
    new_scope.size[0] = float(sx * new_scope.size[0]) if is_rel(sx) else sx
    new_scope.size[1] = float(sy * new_scope.size[1]) if is_rel(sy) else sy
    new_scope.size[2] = float(sz * new_scope.size[2]) if is_rel(sz) else sz
    new_scope.global_x = center - ((new_scope.global_c @ np.diag(new_scope.size)).sum(axis=1) / 2.)

    if isinstance(new_scope, ShapeScope) : 
        assert is_rel(sx) and is_rel(sy), "Not sure what to do when absolute scales are given"
        new_scope.shape = [(float(sx * x), float(sy * y)) for x, y in scope.shape]

    return [new_scope]

@validate_scopes
def translate (scope, tx=0, ty=0, tz=0) : 
    """
    Creates a new scope from the properties of the old scope. 
    """ 
    assert scope.dim == 3, f'Scope with dim = {scope.dim} found'  
    new_scope = deepcopy(scope)
    delta = (new_scope.global_c @ (np.array([tx, ty, tz]).reshape(-1, 1))).reshape(-1)
    new_scope.global_x += delta
    return [new_scope]  

@validate_scopes
def extrude (scope_2d, sz, d=1) : 
    """ 
    extrudes a 2d scope to 3d using the cross product of the 2d scope 

    if d = -1, we have to take negative of the cross product
    """ 
    if d == -1 :
        raise NotImplementedError

    if is_rel(sz) : 
        # if sz is relative, we multiply with the min dim
        sz = min(*scope.size) * sz

    new_scope = deepcopy(scope_2d) 
    new_scope.dim = 3
    new_scope.global_c = add_cross_product(new_scope.global_c)
    new_scope.size.append(sz)
    return [new_scope]

@validate_scopes
def subdiv (scope, axis, args) : 
    """ 
    Subdivides a scope into sub scopes. 

    axis can be 0, 1, 2

    *args is a list of floats/rfloats

    New scopes are created with different sizes and different global_x
    """

    assert implies(isinstance(scope, ShapeScope) , axis==2), f'Works on ShapeScopes only for z axis. Found axis={axis}'

    type_check(args)

    div_ax_sz = scope.size[axis] 
    abs_sum = sum(_ for _ in args if is_abs(_))
    rel_sum = sum(_ for _ in args if is_rel(_))
    
    assert div_ax_sz > abs_sum - EPS, (
        f'Couldn\'t subdivide because absolute sum ({abs_sum}) is greater '
        f'than axis ({axis}) size ({div_ax_sz})'
    )

    if rel_sum == 0 : 
        r = 1.0
    else :
        r = (div_ax_sz - abs_sum) / rel_sum

    new_szs = [_ if is_abs(_) else r * _ for _ in args]

    new_scopes = [] 
    cum_sz = 0
    for i, sz in enumerate(new_szs) : 
        new_scope = deepcopy(scope) 
        new_scope.size[axis] = sz
        new_scope.global_x = cum_sz * scope.global_c[:, axis].reshape(-1) + scope.global_x
        new_scope.index = i
        new_scopes.append(new_scope)
        cum_sz += sz

    return new_scopes 

@validate_scopes
def repeat (scope, axis, val) : 
    """ 
    Breaks the scope down into multiple scopes with repeats
    of approximately size val
    """ 
    assert implies(isinstance(scope, ShapeScope) , axis==2), f'Works on ShapeScopes only for z axis. Found axis={axis}'
    reps = math.ceil(scope.size[axis] / val)
    ac_val = scope.size[axis] / reps
    return subdiv(scope, axis, ([ac_val] * reps))

@validate_scopes
def repeat_n (scope, axis, reps) : 
    """ 
    creates reps repeats
    """ 
    assert implies(isinstance(scope, ShapeScope) , axis==2), f'Works on ShapeScopes only for z axis. Found axis={axis}'
    ac_val = scope.size[axis] / reps
    return subdiv(scope, axis, ([ac_val] * reps))
    
@validate_scopes
def comp (scope, split_type) : 
    """ 
    Splits a scope based on the split type
    """
    assert not isinstance(scope, ShapeScope), 'Does not work for ShapeScopes'
    scopes = [] 
    if split_type == ComponentSplitType.FACES or split_type == ComponentSplitType.SIDE_FACES: 
        assert scope.dim == 3, f'You tried to split a scope of dim = {scope.dim}'
        # F
        sz = scope.size
        scopes.append(
            Scope(
                2, 
                scope.global_x, 
                xz(scope.global_c), 
                [sz[0], sz[2]]
            )
        )
        # B
        scopes.append(
            Scope(
                2, 
                scope.global_x + sz[1] * scope.global_c[:, 1] + sz[0] * scope.global_c[:, 0], 
                mxz(scope.global_c), 
                [sz[0], sz[2]]
            )
        )
        # L 
        scopes.append(
            Scope(
                2,
                scope.global_x + sz[1] * scope.global_c[:, 1],
                myz(scope.global_c), 
                [sz[1], sz[2]]
            )
        )
        # R
        scopes.append(
            Scope(
                2,
                scope.global_x + sz[0] * scope.global_c[:, 0], 
                yz(scope.global_c), 
                [sz[1], sz[2]]
            )
        )
        if split_type == ComponentSplitType.FACES : 
            # B
            scopes.append(
                Scope(
                    2,
                    scope.global_x + sz[0] * scope.global_c[:, 0],
                    mxy(scope.global_c), 
                    [sz[0], sz[1]]
                )
            )
            # T
            scopes.append(
                Scope(
                    2,
                    scope.global_x + sz[2] * scope.global_c[:, 2],
                    xy(scope.global_c), 
                    [sz[0], sz[1]]
                )
            )
    elif split_type == ComponentSplitType.EDGES :
        pass
    elif split_type == ComponentSplitType.VERTICES : 
        pass
    else :
        raise ValueError(f'Unhandled split_type = {split_type}') 
    return scopes

