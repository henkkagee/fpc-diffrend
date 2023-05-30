import src.torch.data as data

def run():
    UprLp_L = r"W:\thesis\blendshapes\UprLp_L.obj"
    BrInRs_00_R = r"W:\thesis\blendshapes\BrInRs_00_R.obj"
    basemeshpath = r"W:\thesis\basemesh.obj"

    basemesh = data.MeshData(basemeshpath)
    UprLp_Lmesh = data.MeshData(UprLp_L)
    BrInRs_00_Rmesh = data.MeshData(BrInRs_00_R)

    uprlp_diff = abs(basemesh.vertices - UprLp_Lmesh.vertices)
    BrInRs_diff = abs(basemesh.vertices - BrInRs_00_Rmesh.vertices)

    newMeshVerts = basemesh.vertices + uprlp_diff + BrInRs_diff

    with open(r"w:/thesis/blended1.obj", mode="w") as f:
        v = 0
        while v < newMeshVerts.shape[0]:
            f.write(f"v {newMeshVerts[v]} {newMeshVerts[v + 1]} {newMeshVerts[v + 2]}\n")
            v += 3
        #
        #f.write(f"vt {u[0]} {u[1]}\n")
        #f.writelines(basemesh.faces)

run()
