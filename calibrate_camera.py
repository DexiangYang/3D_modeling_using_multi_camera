import cv2 as cv
import numpy as np
obj_point = np.zeros((6 * 9, 3), np.float32)
obj_point[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

objpoints = [] 
imgpoints = [] 

video = cv.VideoCapture("videos/test.mp4")
success, frame = video.read()
cnt = 0
while success:
    cnt += 1
    if 0 == cnt % 15:
        # continue
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (9,6), None)
        if True == ret:
            objpoints.append(obj_point)
            imgpoints.append(corners)
    success, frame = video.read()
print("read finsh")
print(len(objpoints))
ret, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.save("npies/mtx.npy", mtx)
np.save("npies/dist.npy", dist)
print(mtx, dist)
video.release()
