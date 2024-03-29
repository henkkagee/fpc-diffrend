import os
import cv2
import numpy as np
import codecs
import json

import xml.etree.ElementTree as et
# tree = et.parse("W:\\2021-12-07-01.xml")
# root = tree.getroot()
# for cam in root:
# ...


"""
Calibrate cameras with cv2 with a 10x10 circular camera calibration pattern

"""

# ------------------------------------------------------------------------

def changeCamName(camname):
    """
    Change camera names to match actual image data.

    :param camname: camera name with "bottom", "top" or "colour"
    :return:
    """
    realcamname = camname.replace("bottom", "primary")
    realcamname = realcamname.replace("top", "secondary")
    return realcamname.replace("colour", "texture")

# ------------------------------------------------------------------------


def calibrate(objpoints, imgpoints, img, xml_cam):
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

    # initial guess for intrinsic matrix and distCoeffs
    intrmatrix = np.array([[6700.0, 0.0, 800.0], [0.0, 6700.0, 600.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    # intrmatrix = np.array([float(x[1]) for x in xml_cam[0].getchildren()[0].items()], dtype=np.float32).reshape((3,3))
    # distCoeffs = np.array([float(x[1]) for x in xml_cam[0].getchildren()[1].items()] + [0.0], dtype=np.float32)
    distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], intrmatrix, distCoeffs,
                                                       flags= cv2.CALIB_ZERO_TANGENT_DIST |
                                                        cv2.CALIB_USE_INTRINSIC_GUESS |
                                                        cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
                                                        | cv2.CALIB_FIX_K3)
    print(f"mtx: {mtx}")
    print(f"dist: {dist}")
    print(f"rvecs: {rvecs[1]}")
    print(f"tvecs: {tvecs[1]}")
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
print(f"objpoints:\n{objpoints}")
objpoints = [objpoints] * 18
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
prevcamname = "pod1primary"
path = "C:/Users/Henrik/fpc-diffrend/calibration/combined/extracted"
# path = "C:/Users/Henrik/fpc-diffrend/calibration/2021-07-01"
# path = r"\\rmd.remedy.fi\Capture\System\RAW\Calibrations\2021-12-07"
images = os.listdir(path)

# DI xmls
tree = et.parse(r"C:\Users\Henrik\fpc-diffrend\data\cube\20220310\2021-12-07-01.dicx")
xml_cams = tree.getroot()[3]

# different threshold values to try to account for reflections in the calibration target
thresholds = [200, 190, 180, 170, 160, 150, 140]

# for root, dirs, files in os.walk(path):
for fname in images:
    camname = fname.split("_")[0]
    # print(f"cam: {fname}")

    # assume images are processed in camera order
    if camname != prevcamname:
        # all images from one camera have been processed
        realcamname = changeCamName(prevcamname)
        print("Calibrating...")
        calibdict[realcamname] = calibrate(objpoints, imgpoints, img, [x for x in xml_cams if x.get('name') == camname+"_0001"][0])
        imgpoints = []

    # read image as grayscale
    # invert, blur, and threshold filter for easier circle detection
    img = cv2.imread(f"{path}/{fname}", flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)
    kernel = np.ones((3, 3), np.float32) / 25
    preimg = cv2.filter2D(img, -1, kernel)

    # Find the circle centers
    # TODO: one could also do blobDetector.detect() and drawKeypoints() before findCirclesGrid() for easier detection
    for thres in thresholds:
        ret, img = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)
        cv2.imshow('thresh', img)
        cv2.waitKey(100)
        ret, centers = cv2.findCirclesGrid(img, np.asarray([10, 10]))

        if not ret:
            ret, centers = cv2.findCirclesGrid(img, np.asarray([10, 10]),
                                           blobDetector=blobdetector,
                                           flags=cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING)
        if ret:
            break


    # If found, add center points and draw them
    if ret:
        print(f"{fname}")
        imgpoints.append(centers)
        cv2.drawChessboardCorners(img, (10,10), centers, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)
    else:
        # raise Exception(f"No centers found for image {path}/{fname}")
        print(f"No centers found for image {path}/{fname}")

    prevcamname = camname

# handle the last camera
realcamname = changeCamName(prevcamname)
calibdict[realcamname] = calibrate(objpoints, imgpoints, img, [x for x in xml_cams if x.get('name') == camname+"_0001"][0])

# save calibration file
json.dump(calibdict, codecs.open("C:/Users/Henrik/fpc-diffrend/calibration/combined/calibration_noblur.json",
                                 'w', encoding='utf-8'),
          separators=(',', ':'),
          sort_keys=True,
          indent=4)

cv2.destroyAllWindows()
