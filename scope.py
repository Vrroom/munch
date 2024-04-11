from enum import Enum 
from copy import deepcopy 
import numpy as np 
import math
from functools import reduce, wraps
from drawTools import *
from utils import *

EPS = 1e-5

def scope_center (scope)  :
    return ((scope.global_c @ np.diag(scope.size)).sum(axis=1) / 2.) + scope.global_x

def scale_scope (scope, scale_factor, axis) :
    assert scope.dim > axis, (
        f'Mismatch in dim and scaling axis '
        f'dim = {scope.dim}, axis = {axis}'
    )
    new_scope = deepcopy(scope)
    new_scope.global_x = new_scope.global_x + (new_scope.global_c[axis, :] * (1. - scale_factor) * new_scope.size[axis] / 2.)
    new_scope.size[axis] *= scale_factor
    print(new_scope.global_x)
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
                
    def draw (self, name=None) : 
        if self.dim == 3 :
            return draw_cuboid(self.global_x, self.global_c, self.size, name=name)
        elif self.dim == 2 :
            return draw_rectangle(self.global_x, self.global_c, self.size, name=name)
        else :
            return draw_line(self.global_x, self.global_c, self.size, name=name)

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

    def set_aabb (self) : 
        p1 = self.global_x
        p2 = (self.global_c @ np.diag(self.size)).sum(axis=1) + self.global_x
        mm = np.minimum(p1, p2)
        MM = np.maximum(p1, p2)
        self.aabb = (mm, MM)

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

@validate_scopes
def identity (scope) : 
    return [scope]

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
    reps = math.ceil(scope.size[axis] / val)
    ac_val = scope.size[axis] / reps
    return subdiv(scope, axis, ([ac_val] * reps))
    
@validate_scopes
def comp (scope, split_type) : 
    """ 
    Splits a scope based on the split type
    """
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
            # T
            scopes.append(
                Scope(
                    2,
                    scope.global_x + sz[2] * scope.global_c[:, 2],
                    xy(scope.global_c), 
                    [sz[0], sz[1]]
                )
            )
            # B
            scopes.append(
                Scope(
                    2,
                    scope.global_x + sz[0] * scope.global_c[:, 0],
                    mxy(scope.global_c), 
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

