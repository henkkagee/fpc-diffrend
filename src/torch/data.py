import numpy as np
import torch

# -------------------------------------------------------------


class MeshData:
    """
    Convenience class for reading Wavefront .obj files.

    """
    def __init__(self, obj):
        vertices = []
        uv = []
        faces = []
        fuv = []
        with open(obj, "r") as f:
            lines = f.readlines()
            for line in lines:
                # vertex positions x,y,z,x,...,z
                if line.startswith("v "):
                    vertices.extend([float(x) for x in line.strip().split(" ")[1:]])

                # texture coordinates
                elif line.startswith("vt "):
                    uv.append([float(x) for x in line.strip().split(" ")[1:]])

                # triangles and face-uv mapping
                elif line.startswith("f "):
                    idxs = [l.split("/") for l in line.strip().split(" ")[1:]]
                    assert len(idxs) == 3
                    # obj vertices are indexed from 1, in opengl from 0
                    faces.append([int(x[0])-1 for x in idxs])
                    fuv.append([int(x[1])-1 for x in idxs])

        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.uv = np.asarray(uv, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.fuv = np.asarray(fuv, dtype=np.int32)

# -------------------------------------------------------------------------------------------------


def vertex_neighbours(faces, n_vertices):
    """
    Get vertex neighbours from a triangular mesh (for computing the mesh laplacian regularization term)
    :param faces: MeshData.faces list of int triplets (vertices) defining triangles in a mesh
    :return vertlist: 2-dimensional list of vertex neighbours by vertex number (0, 1, ..., n_vertices-1)
    """
    print("Computing vertex neighbours...")
    # a nested tensor would be nice but that's only available in an old, incompatible dev release
    vertlist = [[] for i in range(n_vertices)]

    for i, face in enumerate(faces):
        vertlist[face[0]].extend([face[1], face[2]])
        vertlist[face[1]].extend([face[0], face[2]])
        vertlist[face[2]].extend([face[0], face[1]])

    # remove duplicates and pad to length 8 (max number of possible neighbours)
    # pytorch tensors do not support variable-sized column dimensions
    ll = len(vertlist)
    for i, v in enumerate(vertlist):
        vertlist[i] = list(set(v))
        vertlist[i] += [-1] * (8 - len(vertlist[i]))

    print("Done.")
    return vertlist

# -------------------------------------------------------------------------------------------------


def get_vertex_coordinates(vtx_pos, idxs):
    """
    Get tensor of vertex coordinates by vertex index. Vertex neighbour tensors padded with -1
    to avoid variable-sized columns.
    :param vtx_pos: Tensor of shape (n_vertices*3) of vertex coordinates
    :param idxs: List of vertex indices for which vertices to get
    :return:
    """
    size = len(idxs)
    tensor = torch.zeros(size, dtype=torch.float32, device='cuda')
    # atm I don't think there's any other way to do this... this is slow and not nice
    for i in range(size):
        tensor[i] = vtx_pos[idxs[i]]
    return tensor
