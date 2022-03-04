import numpy as np
import torch

# camera utils and projections

class Projection():

    def __init__(self):
        self.intrinsic = np.ndarray((3, 4))
        self.distortion = np.ndarray((2, 2))
        self.extrinsic = np.ndarray((4, 4))

# Transform vertex positions to clip space
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]