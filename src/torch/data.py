import numpy as np

class MeshData():

    def __init__(self, obj):

        vertices = []
        uv = []
        faces = []
        fuv = []
        with open(obj, "r") as f:
            lines = f.readlines()
            for line in lines:
                # vertex positions
                if line.startswith("v "):
                    vertices.extend([float(x) for x in line.strip().split(" ")[1:]])

                # texture coordinates
                if line.startswith("vt "):
                    uv.extend([float(x) for x in line.strip().split(" ")[1:]])

                # triangles and face-uv mapping
                elif line.startswith("f "):
                    idxs = [l.split("/") for l in line.strip().split(" ")[1:]]
                    assert len(idxs) == 3
                    faces.extend([x[0] for x in idxs])
                    fuv.extend([x[1] for x in idxs])

        self.vertices = np.asarray(vertices)
        self.uv = np.asarray(uv)
        self.faces = np.asarray(faces)
        self.fuv = np.asarray(fuv)
