import cv2
import json
import numpy as np
import os
from PIL import Image



path = r"C:\Users\Henkka\Projects\invrend-fpc/data/calibration/2021-07-01/calibration_DIlens.json"
with open(path, 'r') as json_file:
    calib = json.load(json_file)

if not os.path.isdir(r'C:\Users\Henkka\Projects\invrend-fpc/data/img_undistort/take0001'):
    os.mkdir(r'C:\Users\Henkka\Projects\invrend-fpc/data/img_undistort/cube')

take = r"C:\Users\Henkka\Projects\invrend-fpc\data\reference\neutral/20191106_ilkvil_ML_neutral_take0001_pod2colour_range04_regression"
newtake = r"C:\Users\Henkka\Projects\invrend-fpc/data/img_undistort/ilkvil/neutral/take0001"
if not os.path.isdir(newtake):
    os.mkdir(newtake)
fullres = os.path.join(newtake, 'fullres')
if not os.path.isdir(fullres):
    os.mkdir(fullres)

for range in os.listdir(take):
    fr = os.path.join(fullres, range)
    if not os.path.isdir(fr):
        os.mkdir(fr)
    for cam in os.listdir(take):
        frc = os.path.join(fullres, cam)
        if not os.path.isdir(frc):
            os.mkdir(frc)
        for frame in os.listdir(os.path.join(take, cam)):
            image = np.array(Image.open(os.path.join(take, cam, frame)))
            dist = np.asarray(calib[cam.split('_')[-1]]['distortion'], dtype=np.float32)
            intrinsic = np.asarray(calib[cam.split('_')[-1]]['intrinsic'], dtype=np.float32)
            img = cv2.undistort(image, intrinsic, dist)
            resimg = Image.fromarray(img)
            resimg.save(os.path.join(fullres, cam, frame))