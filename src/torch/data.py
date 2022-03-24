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
                # vertex positions
                if line.startswith("v "):
                    vertices.append([float(x) for x in line.strip().split(" ")[1:]])

                # texture coordinates
                if line.startswith("vt "):
                    uv.append([float(x) for x in line.strip().split(" ")[1:]])

                # triangles and face-uv mapping
                elif line.startswith("f "):
                    idxs = [l.split("/") for l in line.strip().split(" ")[1:]]
                    assert len(idxs) == 3
                    faces.append([int(x[0]) for x in idxs])
                    fuv.append([int(x[1]) for x in idxs])

        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.uv = np.asarray(uv, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.fuv = np.asarray(fuv, dtype=np.int32)
