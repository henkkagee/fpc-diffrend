import cv2
import numpy as np

def test_calibJSON():
    import json
    path = "W:/git/fpc-diffrend/calibration.json"
    with open(path, 'r') as json_file:
        data = json.load(json_file)
        for cam in data:
            rot = np.array(data[cam]['rotation'], dtype=np.float32)
            rotvec = np.zeros(3, dtype=np.float32)
            cv2.Rodrigues(rot, rotvec)
            data[cam]['rotvec'] = rotvec.tolist()
        with open("W:/git/fpc-diffrend/calibration2.json", 'w') as wfile:
            json.dump(data, wfile)
            

test_calibJSON()