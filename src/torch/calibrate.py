import os
import cv2
import numpy as np
import codecs
import json

# ------------------------------------------------------------------------


def calibrate(objpoints, imgpoints, img):
    """
    Calibrate camera with a set of calibration images. Parameters from:
    https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d

    :param objpoints:
    :param imgpoints:
    :param img:
    :return:
    """
    imgpoints = np.asarray(imgpoints, dtype=np.float32)
    assert imgpoints.shape[0] == objpoints.shape[0]
    assert img is not None
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    print(f"success? {ret}")
    print(f"mtx: {mtx}")
    print(f"dist: {dist}")
    print(f"rvecs: {rvecs[0]}")
    print(f"tvecs: {tvecs[0]}")
    rmat = np.asarray([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]], dtype=np.float64)

    cv2.Rodrigues(rvecs[0], rmat)
    print(f"rmat: {rmat}")
    if ret:
        return {"intrinsic": mtx.tolist(), "rotation": rmat.tolist(),
                                  "translation": tvecs[0].tolist(), "distortion": dist.tolist()}

# ------------------------------------------------------------------------


# 3d object points are known: 10x10 circle grid target, 2cm offsets. 3D origin is at the target's center crosshair.
objpoints = []  # 3d point in real world space
for y in range(9, -10, -2):
    x = [x for x in range(-9, 10, 2)]
    objpoints.append(list([list(a) for a in zip(x, [y] * 10, [0] * 10)]))
objpoints = [i for sub in objpoints for i in sub]
objpoints = [objpoints] * 9
objpoints = np.asarray(objpoints, dtype=np.float32)

# Blob detector for better circle center detection
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 1
params.minCircularity = 0.05
params.minConvexity = 0.50
blobdetector = cv2.SimpleBlobDetector_create(params)

calibdict = {}
imgpoints = []  # 2d points in image plane.
img = None
prevcamname = "pod1bottom"
path = "C:/Users/Henkka/Projects/invrend-fpc/data/calibration/2018-11-15/extracted"
images = os.listdir(path)

for fname in images:
    camname = fname.split("_")[0]
    print(f"cam: {fname}")

    # assume images are processed in camera order
    if camname != prevcamname:
        # all (9) images from one camera have been processed
        realcamname = prevcamname.replace("bottom", "primary")
        realcamname = realcamname.replace("top", "secondary")
        realcamname = realcamname.replace("colour", "texture")
        calibdict[realcamname] = calibrate(objpoints, imgpoints, img)
        imgpoints = []

    # read image as grayscale
    # invert, blur, and threshold filter for easier circle detection
    img = cv2.imread(f"{path}/{fname}",flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)
    kernel = np.ones((5, 5), np.float32) / 25
    img = cv2.filter2D(img, -1, kernel)
    ret, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)

    # Find the circle centers
    # treat pod1bottom_0009 separately as it's tricky to automatically detect due to angle
    # EDIT: one could also do blobDetector.detect() and drawKeypoints() before findCirclesGrid() for easier detection
    if "pod1bottom_0009" in fname:
        ret, centers = cv2.findCirclesGrid(img, np.asarray([10, 10]),
                                           blobDetector=blobdetector,
                                           flags=cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING)
    else:
        ret, centers = cv2.findCirclesGrid(img, np.asarray([10, 10]))

    # If found, add center points and draw them
    if ret:
        imgpoints.append(centers)
        cv2.drawChessboardCorners(img, (10,10), centers, ret)
        cv2.imshow('img', img)
        cv2.waitKey(200)
    else:
        raise Exception(f"No centers found for image {path}/{fname}")

    prevcamname = camname

realcamname = prevcamname.replace("bottom", "primary")
realcamname = realcamname.replace("top", "secondary")
realcamname = realcamname.replace("colour", "texture")
calibdict[realcamname] = calibrate(objpoints, imgpoints, img)

# save calibration file
json.dump(calibdict, codecs.open("C:/Users/Henkka/Projects/invrend-fpc/data/calibration/2018-11-15/calibration.json",
                                 'w', encoding='utf-8'),
          separators=(',', ':'),
          sort_keys=True,
          indent=4)

cv2.destroyAllWindows()
