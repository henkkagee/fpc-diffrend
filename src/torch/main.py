# builtin
import os

# 3rd party
import numpy as np

# custom
from data import MeshData
import dr
import camera as c

# ------------------------------------------------------------


def run(mdir, nbl, bmesh, imdir, it):
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

    # images
    # imgs[range][pos][1-3][img] ?
    imgs = []

    proj = c.Projection()

    for range in os.listdir(imdir):
        for cam in os.listdir(range):
            if "top" in cam:
                if "pod2" in cam:
                    proj.extrinsic =
                    proj.distortion =
                    proj.intrinsic = 
            elif "bottom" in cam:
                pass
            elif "colour" in cam:
                pass

    # setup

    # dr
    fit_mesh()

# -----------------------------------------------------------

# Define args here
meshdir = ""
imagedir = ""
num_bl = 624
basemesh = ""
iters = 1000

if __name__ == "__main__":
    run(meshdir, num_bl, basemesh, imagedir, iters)
