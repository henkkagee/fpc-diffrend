import numpy as np
import torch

# camera utils and projections

# ------------------------------------------------------------------------------------


def transformClip(mvp, pos):
    """
    Transform vertex coordinates to clip space.

    :param mvp: Modelview * Projection matrix
    :param pos: Tensor of vertex positions
    :return: Tensor of transformed vertex positions
    """
    t_mtx = torch.from_numpy(mvp).cuda() if isinstance(mvp, np.ndarray) else mvp
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

# ------------------------------------------------------------------------------------

def intrinsicToProjection(intr, zn=0.01, zf=1000):
    """
    Get OpenGL projection matrix from intrinsic camera parameters.

    :param intr: 3x3 intrinsic camera matrix specifying focal length
    and principal point in pixels (+ skew)
    :param zn: Distance to front clipping plane
    :param zf: Distance to back clipping plane
    :return: 4x4 projection matrix
    """
    n = 1.0
    x = 0.1
    f = 100.0
    return np.array([[n/x,    0,            0,              0],
                     [  0, n/-x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)

    return np.array([[(2 * intr[0,0])/(2 * intr[0,2]), 0, 0, 0],
                     [0, (2 * intr[1,1])/(2 * intr[1,2]), 0, 0],
                     [0, 0, -(zf + zn) / (zf - zn), -(2 * zf * zn) / (zf - zn)],
                     [0, 0, -1, 0]]).astype(np.float32)

# ------------------------------------------------------------------------------------


def extrinsicToModelview(rmat, tvec):
    """
    Get OpenGL modelview matrix from extrinsic camera parameters.

    :param rmat: 3x3 camera rotation matrix w.r.t. world origin
    :param tvec: 1x3 camera translation matrix w.r.t world origin
    :return: 4x4 modelview matrix
    """

    return np.array([[1,  0, 0, 0],
                     [0,  -1, 0, 0],
                     [0, 0, -1, -30],
                     [0,  0, 0, 1]]).astype(np.float32)

    rt = np.append(rmat, tvec, axis=1)
    br = np.array([0, 0, 0, 1], dtype=np.float32)
    mdv = np.vstack((rt, br))
    """mdv[0, 1] *= -1
    mdv[0, 2] *= -1
    mdv[1, 0] *= -1
    mdv[2, 0] *= -1"""
    """mdv[1, 0] *= -1
    mdv[1, 1] *= -1
    mdv[1, 2] *= -1
    mdv[2, 0] *= -1
    mdv[2, 1] *= -1
    mdv[2, 2] *= -1
    mdv[1, 3] *= -1
    mdv[2, 3] *= -1"""
    return mdv

# ------------------------------------------------------------------------------------

