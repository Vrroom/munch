from scipy.spatial.transform import Rotation  
import numpy as np
import random
import bpy
import string

def random_string() :
    return ''.join(random.choices(string.ascii_lowercase, k=20))

def rotation_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def euler_xyz_rotation_matrix(rx, ry, rz):
    Rx = rotation_x(rx)
    Ry = rotation_y(ry)
    Rz = rotation_z(rz)
    return Rz @ Ry @ Rx 

def normalized (x) :
    return x / np.linalg.norm(x)

def is_object (x) : 
    return isinstance(x, bpy.types.Object)

def list_dict_flatten (obj) : 
    if isinstance(obj, list) : 
        for item in obj :
            yield from list_dict_flatten(item)
    elif isinstance(obj, dict) : 
        for k, v in obj.items() : 
             yield from list_dict_flatten(v)
    else :
        yield obj

def clamp (x, a, b) : 
    return max(a, min(x, b))

def clamp_elem_wise_np (x, a, b) : 
    return np.maximum(a, np.minimum(x, b))

def complement(pred) : 
    return lambda x : not pred(x)

def rotation_to_euler_angles (rot, degrees=False) : 
    """ xyz is the only ordering I'm going to use """
    return Rotation.from_matrix(rot).as_euler('xyz', degrees=degrees)

def euler_angles_to_rotation(euler_angles, degrees=False):
    return Rotation.from_euler('xyz', euler_angles, degrees=degrees).as_matrix()

def sample_idx (probs) : 
    return random.choices(range(len(probs)), weights=probs)[0]
    
def add_cross_product(matrix):
    if matrix.shape != (3, 2):
        raise ValueError("Input matrix must be 3x2.")
    x = matrix[:, 0]
    y = matrix[:, 1]
    x_cross_y = np.cross(x, y)
    if x_cross_y.ndim == 1:
        x_cross_y = x_cross_y.reshape(-1, 1)
    result = np.column_stack((matrix, x_cross_y))
    return result

def homogenize_rotation_matrix(rot_matrix):
    if rot_matrix.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix")
    hom_matrix = np.eye(4)
    hom_matrix[:3, :3] = rot_matrix
    return hom_matrix

def homogenize_translation(tx):
    hom_matrix = np.eye(4)
    hom_matrix[:3, 3] = tx
    return hom_matrix

def homogenize_scale(sx):
    hom_matrix = np.eye(4)
    hom_matrix[[0, 1, 2], [0, 1, 2]] = sx
    return hom_matrix

def seed_everything(seed: int):
    import random, os
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
def generate_ngon (n, radius, theta_i=0.0) : 
    thetas = theta_i + np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = radius + radius * np.cos(thetas)
    ys = radius + radius * np.sin(thetas)
    return list(zip(xs.tolist(), ys.tolist()))

def implies (a, b) :
    return (not a) or b
