import os

olddir = os.path.join(os.getcwd(), "ilkvil_blendshapes")
newdir = os.path.join(os.getcwd(), "ilkvil_blendshapes_eyes_plusnormals")

with open(os.path.join(os.getcwd(), "basemesh_patched_correctedvertices.obj"), 'r') as newbasemesh:
    vt = []
    f = []
    vn = []
    for line in newbasemesh:
        if line.startswith("vt "):
            vt.append(line)
        elif line.startswith("vn "):
            vn.append(line)
        elif line.startswith("f "):
            f.append(line)


for i, objfile in enumerate(os.listdir(olddir)):
    print(i)
    with open(os.path.join(olddir, objfile), 'r') as oldmesh:
        with open(os.path.join(newdir, objfile), 'w') as newmesh:
            for line in oldmesh:
                if line.startswith("v "):
                    newmesh.write(line)
                else:
                    break
            newmesh.writelines(vn)
            newmesh.writelines(vt)
            newmesh.writelines(f)
