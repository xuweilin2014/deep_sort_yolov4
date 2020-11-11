#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default="./test_video/TownCentreXVID.avi")
ap.add_argument("-c", "--class", help="name of class", default="person")
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")


# list = [[] for _ in range(100)]

def main(yolo):
    start = time.time()
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    counter = []
    # deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    find_objects = ['person']
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    write_video_flag = True
    # VideoCapture()中参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
    video_capture = cv2.VideoCapture(args["input"])

    if write_video_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./output/output.avi', fourcc, 15, (w, h))
        list_file = open('detection_rslt.txt', 'w')
        frame_index = -1

    fps = 0.0

    while True:
        # cap.read()按帧读取视频，ret,frame是获cap.read()方法的两个返回值。其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
        ret, frame = video_capture.read()  # frame shape 640 * 480 * 3
        if not ret:
            break
        t1 = time.time()

        # Image.fromarray 就是将数组转变为图像
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxes, confidence, class_names = yolo.detect_image(image)
        features = encoder(frame, boxes)
        # score to 1.0 here
        # 根据 yolo 检测到的检测框和特征向量，得到 Detection 类对象
        detections = [Detection(bbox, conf, feature) for bbox, conf, feature in zip(boxes, confidence, features)]
        # 获取到 frame 图像中每一个检测框的坐标
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        # 进行非极大值抑制
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        # 根据最新得到的检测框，进行级联匹配，对轨迹进行更新，可能新增 track，也可能删除掉一些 track
        tracker.update(detections)

        i = int(0)
        indexIDs = []

        for det in detections:
            bbox = det.to_tlbr()
            # 在 frame 上画出黑色的检测框
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

        for track in tracker.tracks:
            # 遇到以下两种情况，不在 frame 上显示出追踪的轨迹
            # 1.track 的状态不为 confirmed，也就是说要么处于 deleted 或者 tentative
            # 2.track 所追踪的目标没有匹配上 detection
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            list_file.write(str(frame_index) + ',')
            list_file.write(str(track.track_id) + ',')

            # cv2.rectangle 在图像上显示矩形框，使用对角线，来画矩形
            # cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (rgb color), 3)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)

            b0 = str(bbox[0])  # bbox.left
            b1 = str(bbox[1])  # bbox.top
            b2 = str(bbox[2] - bbox[0])  # bbox.right
            b3 = str(bbox[3] - bbox[1])  # bbox.bottom

            list_file.write(str(b0) + ',' + str(b1) + ',' + str(b2) + ',' + str(b3))
            list_file.write('\n')

            # 在检测框上方显示物体的追踪 id
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 150, (color), 2)

            # 在检测框上方显示物体的类别
            if len(class_names) > 0:
                class_name = class_names[0]
                cv2.putText(frame, str(class_names[0]), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150, (color), 2)

            i += 1

            # 算出检测框的中心点 (x,y) 坐标
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            pts[track.track_id].append(center)

            thickness = 5

            # cv2.circle 根据给定的圆心和半径等画园
            # 前面求出了检测框的中心点，在检测框上以中心点为圆心，画一个半径为 1 的圆
            cv2.circle(frame, (center), 1, color, thickness)

            # pts 中保存了检测框前一段时间的中心点的坐标，因此遍历 pts，并且在每两个点之间画出一个条线，据此在 frame 上
            # 显示出物体的移动轨迹
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), (color), thickness)

        count = len(set(counter))
        cv2.putText(frame, "Total Pedestrian Counter: " + str(count), (int(20), int(120)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "Current Pedestrian Counter: " + str(i), (int(20), int(80)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "FPS: %f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)
        cv2.namedWindow("YOLO4_Deep_SORT", 0)
        cv2.resizeWindow('YOLO4_Deep_SORT', 1024, 768)
        cv2.imshow('YOLO4_Deep_SORT', frame)

        if write_video_flag:
            # 将经过处理（画上了检测框和追踪框）的 frame 保存到 output.avi 中
            out.write(frame)
            frame_index = frame_index + 1

        fps = (fps + (1. / (time.time() - t1))) / 2
        out.write(frame)
        frame_index = frame_index + 1

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(" ")
    print("[Finish]")
    end = time.time()

    if len(pts[track.track_id]) is not None:
        print(args["input"][43:57] + ": " + str(count) + " " + str(class_name) + ' Found')

    else:
        print("[No Found]")
    # print("[INFO]: model_image_size = (960, 960)")
    video_capture.release()
    if write_video_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
