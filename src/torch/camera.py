import numpy as np
import torch

# camera utils and projections

# ------------------------------------------------------------------------------------


def transform_clip(mvp, pos):
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

def intrinsicToProjection(intr, zn=0.1, zf=100):
    """
    Get OpenGL projection matrix from intrinsic camera parameters.

    :param intr: 3x3 intrinsic camera matrix specifying focal length
    and principal point in pixels (+ skew)
    :param zn: Distance to front clipping plane
    :param zf: Distance to back clipping plane
    :return: 4x4 projection matrix
    """
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
    return np.r_[np.c_[rmat, tvec], np.array([0, 0, 0, 1], dtype=np.float32)]
