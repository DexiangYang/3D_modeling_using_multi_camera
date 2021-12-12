# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import sys
import cv2
import time
import numpy as np
import ncnn
from ncnn.model_zoo import get_model
from ncnn.utils import draw_detection_objects
from matplotlib import pyplot as plt

def get_xy_from_yolo(image, class_names, objects, cnt = 0, min_prob=0.0):
    rslt_array = []
    for obj in objects:
        if obj.prob < min_prob:
            continue
        if obj.label == 0:
            y, x, h, w = obj.rect.y, obj.rect.y + obj.rect.h, obj.rect.x, obj.rect.x + obj.rect.w
            return (obj.rect.y, obj.rect.y + obj.rect.h, obj.rect.x, obj.rect.x + obj.rect.w)
        else:
            return (0, 0, 0, 0)

def my_draw_pose(image, keypoints, x_shift, y_shift, cnt):
    points = np.zeros((17, 2))
    points_set = set()
    cnt = -1
    for keypoint in keypoints:
        cnt += 1
        # print("%.2f %.2f = %.5f" % (keypoint.p.x + x_shift, keypoint.p.y, keypoint.prob))
        if keypoint.prob < 0.2:
            continue
        points_set.add(cnt)
        points[cnt, 0] = keypoint.p.x + x_shift
        points[cnt, 1] = keypoint.p.y + y_shift
    return points_set, points
    #     cv2.circle(image, (int(keypoint.p.x + x_shift), int(keypoint.p.y + y_shift)), 3, (0, 255, 0), -1)
    # cv2.imwrite(f"tmp/{cnt}.jpg", image)
if __name__ == "__main__":

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    mtx = np.load("npies/mtx.npy")
    dist = np.load("npies/dist.npy")

    obj_point = np.zeros((6 * 9, 3), np.float32)
    obj_point[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    print(mtx, dist)
    mat_inv = np.linalg.inv(mtx)

    # video = cv.VideoCapture("videos/test.mp4")
    # success, frame = video.read()
    # cnt = 0
    # num_of_pictures = 200

    # # config the matrix saving the paramter
    # # matrix_A = np.zeros((num_of_pictures * 3, num_of_pictures + 3))
    # # matrix_B = np.zeros((num_of_pictures * 3, 1))
    # matrix_E = np.array([   [1, 0, 0], 
    #                         [0, 1, 0],
    #                         [0, 0, 1]])
    
    yolo_net = get_model(
        "yolov5s",
        target_size=640,
        prob_threshold=0.25,
        nms_threshold=0.45,
        num_threads=4,
        use_gpu=True,
    )
    
    pose_net = get_model("simplepose", 
        num_threads=32, 
        use_gpu=True)

    cap = cv2.VideoCapture("videos/VID_20211207_232129.mp4")
    ret, img = cap.read()
    cnt = 0
    # numpy.zeros()
    points_array = {}
    for i in range(17):
        points_array[i] = {"A" : [], "B" : []}
    step = 1
    while ret and cnt < 300: 
        if cnt % step == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                ret, r_vec, t_vec = cv2.solvePnP(obj_point, corners2, mtx, dist)
                r_mat, _ = cv2.Rodrigues(r_vec)
                r_mat_inv = np.linalg.inv(r_mat)
                try:
                    objects = yolo_net(img)
                    y, y_, x, x_  = get_xy_from_yolo(img, yolo_net.class_names, objects, cnt)
                    keypoints = pose_net(img[int(y): int(y_), int(x): int(x_)].copy())
                    points_set, points = my_draw_pose(img, keypoints, x, y, cnt)
                    print(points, points_set)
                    for i in points_set:
                        
                        x, y = points[i][0], points[i][1]
                        # print(corners2[0, 0, 0])
                        # x, y = corners2[1, 0, 0], corners2[1, 0, 1]
                        cv2.circle(img, (int(x), int(y)), 8, (255,0,0), -1)
                        cv2.putText(img, f"{i}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        A = np.matmul(r_mat_inv, np.matmul(mat_inv, np.array([[x], [y], [1]])))
                        B = np.matmul(r_mat_inv, t_vec)
                        points_array[i]["A"].append(A)
                        points_array[i]["B"].append(B)
                except Exception as e:
                    print(e)
        cv2.imwrite(f"tmp/{cnt // step}.jpg", img)
        ret, img = cap.read()
        cnt += 1
matrix_E = np.array([   [1, 0, 0], 
                        [0, 1, 0],
                        [0, 0, 1]])
from numpy.linalg import lstsq
ax = plt.axes(projection = "3d")
for i in range(17):
    num_of_pictures = len(points_array[i]["A"])
    matrix_A = np.zeros((num_of_pictures * 3, num_of_pictures + 3))
    matrix_B = np.vstack(points_array[i]["B"])
    for j in range(num_of_pictures):
        matrix_A[3 * j : 3 * (j + 1), num_of_pictures : ] = - matrix_E
        matrix_A[3 * j : 3 * (j + 1), j : j + 1] = points_array[i]["A"][j]
    # print(lstsq(matrix_A, matrix_B, rcond=None)[0][-3:])
    x, y, z = lstsq(matrix_A, matrix_B, rcond=None)[0][-3:]
    print(f"({x[0]}, {y[0]}, {z[0]}),")
    ax.scatter3D(x, y, z)
plt.savefig('3d.jpg', bbox_inches = 'tight')
    
