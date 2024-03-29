import numpy as np
import torch

"""
Camera utils and projections

"""

# -------------------------------------------------------------------------------------------------

def transform_clip(mvp, pos):
    """
    Transform vertex coordinates to clip space.

    :param mvp: Modelview * Projection matrix
    :param pos: Tensor of vertex positions
    :return: Tensor of transformed vertex positions
    """
    t_mtx = torch.from_numpy(mvp).cuda(device='cuda') if isinstance(mvp, np.ndarray) else mvp
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda(device='cuda')], axis=1)
    ret = torch.matmul(posw, t_mtx.t())[None, ...]
    return ret

# -------------------------------------------------------------------------------------------------

def intrinsic_to_projection(intr=None, zn=0.01, zf=200):
    """
    Get OpenGL projection matrix from intrinsic camera parameters.

    :param intr: 3x3 intrinsic camera matrix specifying focal length
    and principal point in pixels (+ skew)
    :param zn: Distance to front clipping plane
    :param zf: Distance to back clipping plane
    :return: 4x4 projection matrix
    """

    return np.array([[intr[0, 0]/intr[0, 2], 0, 0, 0],
                     [0, intr[1, 1]/intr[1, 2], 0, 0],
                     [0, 0, -(zf + zn) / (zf - zn), -(2 * zf * zn) / (zf - zn)],
                     [0, 0, -1, 0]]).astype(np.float32)

# -------------------------------------------------------------------------------------------------


def extrinsic_to_modelview(rmat=None, tvec=None):
    """
    Get OpenGL modelview matrix from extrinsic camera parameters.

    :param rmat: 3x3 camera rotation matrix w.r.t. world origin
    :param tvec: 1x3 camera translation matrix w.r.t world origin
    :return: 4x4 modelview matrix
    """

    rt = np.append(rmat, tvec, axis=1)
    br = np.array([0, 0, 0, 1], dtype=np.float32)
    mdv = np.vstack((rt, br))
    mdv[1, 0] *= -1
    mdv[1, 1] *= -1
    mdv[1, 2] *= -1
    mdv[2, 0] *= -1
    mdv[2, 1] *= -1
    mdv[2, 2] *= -1
    mdv[1, 3] *= -1
    mdv[2, 3] *= -1
    return mdv

# -------------------------------------------------------------------------------------------------

def default_projection(xn=1.0, xf=50.0, x=0.1):
    return np.array([[xn/x,    0,            0,              0],
                     [  0, xn/-x,            0,              0],
                     [  0,    0, -(xf+xn)/(xf-xn), -(2*xf*xn)/(xf-xn)],
                     [  0,    0,           -1,              0]]).astype(np.float32)

# -------------------------------------------------------------------------------------------------


def default_modelview(zoffset=-30):
    return np.array([[1,  0, 0, 0],
                     [0,  1, 0, 0],
                     [0, 0, 1, zoffset],
                     [0,  0, 0, 1]]).astype(np.float32)

# -------------------------------------------------------------------------------------------------


def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]]).astype(np.float32)

# -------------------------------------------------------------------------------------------------


def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0],
                     [0,  c, s, 0],
                     [0, -s, c, 0],
                     [0,  0, 0, 1]]).astype(np.float32)

# -------------------------------------------------------------------------------------------------


def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]).astype(np.float32)

# -------------------------------------------------------------------------------------------------


def translate_tensor(vec):
    mat = torch.zeros((4,4), dtype=torch.float32, device='cuda')
    mat[(0, 1, 2, 3), (0, 1, 2, 3)] = torch.ones(4, dtype=torch.float32, device='cuda')
    # assign xyz translation values into mat to preserve gradients
    # print(f"vec: {vec.shape}\nreshape: {vec.reshape((3,1)).shape}")
    mat[(0, 1, 2), (3, 3, 3)] = torch.flatten(vec)
    return mat

# -------------------------------------------------------------------------------------------------


def rigid_grad(tvec, rotmat):
    rt = torch.cat((rotmat, torch.reshape(tvec, (3, 1))), 1)
    br = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device='cuda')
    transform = torch.cat((rt, torch.reshape(br, (1, 4))), 0)
    return transform
