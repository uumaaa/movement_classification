import numpy as np
import cv2
import os
from KalmanFilter import KalmanFilter
from Detector import detect

classes = ['GolfSwing/','']
for class_name in classes:
    for video in os.listdir('dataset/test/'+class_name):
        cap = cv2.VideoCapture('dataset/test/'+class_name+video)
        KF = KalmanFilter(0.1, 1, 1, 1, 0.4,0.4)
        kernel = np.ones((5,5),dtype=np.uint8)
        fgbg = cv2.createBackgroundSubtractorMOG2()
        max_width_bbox = -1
        max_height_bbox = -1
        while (True):
            ret, frame = cap.read()
            if not ret:
                break
            imgs_thresh,center = detect(frame,fgbg,kernel)
            if (center[0] != -1):
                (x, y) = KF.predict()
                (x1, y1) = KF.update(np.array([center[0]+center[2]/2,center[1]+center[3]/2]))
                if((center[0] < x1 < center[0] + center[2]) and (center[1] < y1 < center[1] + center[3])):
                    max_width_bbox = max(max_width_bbox,int(center[2]))
                    max_height_bbox = max(max_height_bbox,int(center[3]))
        cap.release()
        cap = cv2.VideoCapture('dataset/test/'+class_name+video)
        ret, frame = cap.read()
        x1 = int(frame.shape[0]/2)
        y1 = int(frame.shape[1]/2)
        mid_width = int(max_width_bbox/2)
        mid_height = int(max_height_bbox/2)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            imgs_thresh, center = detect(frame, fgbg, kernel)
            if center[0] != -1:
                (x, y) = KF.predict()
                (x1, y1) = KF.update(np.array([center[0] + center[2] / 2, center[1] + center[3] / 2]))
                y1 = int(y1)
                x1 = int(x1)
                y1_start = max(0, y1 - mid_height)
                y1_end = min(frame.shape[0], y1 + mid_height)
                x1_start = max(0, x1 - mid_width)
                x1_end = min(frame.shape[1], x1 + mid_width)
                movement_frame = frame[y1_start:y1_end, x1_start:x1_end]
            else:
                y1 = int(y1)
                x1 = int(x1)
                y1_start = max(0, y1 - mid_height)
                y1_end = min(frame.shape[0], y1 + mid_height)
                x1_start = max(0, x1 - mid_width)
                x1_end = min(frame.shape[1], x1 + mid_width)
                movement_frame = frame[y1_start:y1_end, x1_start:x1_end]
            cv2.imshow('Objects Tracked',cv2.resize(movement_frame,(224,224)))
            k = cv2.waitKey(101)
            if k == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        