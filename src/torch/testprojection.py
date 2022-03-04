import cv2
from data import MeshData
import numpy as np

import matplotlib.pyplot as plt

# basemesh
bmesh = "C:/Users/Henkka/Projects/invrend-fpc/data/basemesh.obj"
bm = MeshData(bmesh)
bm_vtx = bm.vertices

obj = []
"""for i in range(len(bm_vtx)//3):
    if i % 10 == 0:
        obj.append([bm_vtx[i*3], bm_vtx[i*3+1] - 170, bm_vtx[i*3+2]])"""
obj = [[4.379, 7.452, 14.691], [-4.379, 7.452, 14.691], [0, 3.113, 18.978],
       [-2.954, -0.328, 15.730], [2.954, -0.328, 15.730]]
objectPoints = np.asarray(obj, dtype=np.float64)

cameraMatrix = np.asarray([[35, 0, 800], [0, 35, 600], [0, 0, 1]], dtype=np.float64)
# cameraMatrix = np.asarray([[6769.82, 0, 701.915], [0, 6782.57, 495.164], [0, 0, 1]], dtype=np.float64)
print(cameraMatrix)
rmat = np.asarray([[0.9998071, -0.0055418, -0.0188460],
        [-0.0054683, -0.9999773,  0.0039444],
        [-0.0188674, -0.0038405, -0.9998146]], dtype=np.float64)
"""rmat = np.asarray([  [0.9999822, -0.0059635, -0.0001459],
   [0.0049483,  0.8155873,  0.5786129],
  [-0.0033315, -0.5786033,  0.8156023] ], dtype=np.float64)"""
rvec = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
cv2.Rodrigues(rmat, rvec)
tvec = np.asarray([0.0549075, 0.0538474, 1.46651], dtype=np.float64)
distCoeffs = np.asarray([0.0928989, 2.02773, -0.0252613, 0.00817519], dtype=np.float64)

pts = [[0.0, 0.0] for i in range(len(objectPoints))]
imagePoints = np.asarray(pts, dtype=np.float64)

print(f"objpts: {objectPoints.shape}\nrmat: {rmat.shape}")
imagePoints, _ = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)
print(f"imagePoints: {imagePoints}")
print(f"imagePoints: {imagePoints[0]}")

x = [i[0,0] for i in imagePoints if i[0,0] < 1600 and i[0,1] < 1600 and i[0,0] > -100 and i[0,1] > -100]
y = [i[0,1] for i in imagePoints if i[0,0] < 1600 and i[0,1] < 1600 and i[0,0] > -100 and i[0,1] > -100]
plt.scatter(x, y)
plt.show()
