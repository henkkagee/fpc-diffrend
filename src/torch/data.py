import numpy as np

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

def vert_neighbours(faces):
    """
    Get vertex neighbours from a triangular mesh (for computing the mesh laplacian regularization term)

    :param faces: MeshData.faces list of int triplets (vertices) defining triangles in a mesh
    :return vertdict: dict vertex neighbours by vertex number
    """

    vertdict = {}
    for face in faces:
        if not face[0] in vertdict:
            vertdict[face[0]] = []
        if not face[1] in vertdict:
            vertdict[face[1]] = []
        if not face[2] in vertdict:
            vertdict[face[2]] = []
        vertdict[face[0]].extend([face[1], face[2]])
        vertdict[face[1]].extend([face[0], face[2]])
        vertdict[face[2]].extend([face[0], face[1]])

    for vertex, neighs in vertdict.items():
        vertdict[vertex] = list(set(neighs))

    return vertdict