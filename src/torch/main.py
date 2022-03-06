# builtin
import os

# 3rd party
import numpy as np

# local
from src.torch.data import MeshData
import fit
import camera

# ------------------------------------------------------------


def run(mdir, nbl, bmesh, imdir, it, calib):
    # blendshapes
    bl_vtx = np.empty(nbl)
    bl_tri = np.empty(nbl)
    for i, file in enumerate(os.listdir(mdir)):
        if file.endswith(".obj"):
            mesh = MeshData(file)
            bl_vtx[i] = mesh.vertices
            bl_tri[i] = mesh.faces

    # basemesh
    bm = MeshData(bmesh)
    bm_vtx = np.asarray(bm.vertices)
    bm_tri = np.asarray(bm.faces)

    # setup

    # dr
    fit.fitTake()

# -----------------------------------------------------------

# Define args here
meshdir = ""
imagedir = ""
num_bl = 624
basemesh = ""
iters = 1000
calibration = ""

if __name__ == "__main__":
    run(meshdir, num_bl, basemesh, imagedir, iters, calibration)
