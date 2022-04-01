import cv2
import json
import numpy as np
import os
from PIL import Image

import xml.etree.ElementTree as et
# tree = et.parse("W:\\2021-12-07-01.xml")
# root = tree.getroot()
# for cam in root:
# ...

path = "W:/git/invrend-fpc/calibration.json"
with open(path, 'r') as json_file:
    calib = json.load(json_file)

if not os.path.isdir('W:/git/fpc-diffrend/data/img_undistort'):
    os.mkdir('W:/git/fpc-diffrend/data/img_undistort')

take = r"\\rmd.remedy.fi\Capture\BigFish\RAW\20191106\ML\ROM\ilkvil\take0001"
newtake = "W:/git/fpc-diffrend/data/img_undistort/take0001"
if not os.path.isdir(newtake):
    os.mkdir(newtake)
fullres = os.path.join(newtake, 'fullres')
if not os.path.isdir(fullres):
    os.mkdir(fullres)

for range in os.listdir(os.path.join(take, 'fullres')):
    fr = os.path.join(fullres, range)
    if not os.path.isdir(fr):
        os.mkdir(fr)
    for cam in os.listdir(os.path.join(take, 'fullres', range)):
        frc = os.path.join(fullres, range, cam)
        if not os.path.isdir(frc):
            os.mkdir(frc)
        for frame in os.listdir(os.path.join(take, 'fullres', range, cam)):
            image = np.array(Image.open(os.path.join(take, 'fullres', range, cam, frame)))
            dist = np.asarray(calib[cam.split('_')[-1]]['distortion'], dtype=np.float32)
            intrinsic = np.asarray(calib[cam.split('_')[-1]]['intrinsic'], dtype=np.float32)
            img = cv2.undistort(image, intrinsic, dist)
            resimg = Image.fromarray(img)
            resimg.save(os.path.join(fullres, range, cam, frame))