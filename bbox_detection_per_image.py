
import numpy as np
import cv2
import os
from KalmanFilter import KalmanFilter
from Detector import detect

#classes = ['GolfSwing/','Archery/','HandstandPushups/','HandstandWalking/','Drumming/','WritingOnBoard/','Pullups/','JumpingJack/','PlayingGuitar']
classes = ['PlayingGuitar/']
output_base_path = 'cleaned_dataset/'

for class_name in classes:
    class_path = 'dataset/train/' + class_name
    for video in os.listdir(class_path):
        input_video_path = class_path + video
        output_video_path = os.path.join(output_base_path,'train', class_name, video)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        cap = cv2.VideoCapture(input_video_path)
        KF = KalmanFilter(0.1, 1, 1, 1, 0.6, 0.6)
        kernel = np.ones((5, 5), dtype=np.uint8)
        fgbg = cv2.createBackgroundSubtractorMOG2()
        max_width_bbox = -1
        max_height_bbox = -1

        # First pass to determine the max width and height of the bounding box
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            imgs_thresh, center = detect(frame, fgbg, kernel)
            if center[0] != -1:
                (x, y) = KF.predict()
                (x1, y1) = KF.update(np.array([center[0] + center[2] / 2, center[1] + center[3] / 2]))
                if (center[0] < x1 < center[0] + center[2]) and (center[1] < y1 < center[1] + center[3]):
                    max_width_bbox = max(max_width_bbox, int(center[2]))
                    max_height_bbox = max(max_height_bbox, int(center[3]))

        cap.release()

        cap = cv2.VideoCapture(input_video_path)
        ret, frame = cap.read()
        frame_height, frame_width = frame.shape[:2]
        x1 = frame_width // 2
        y1 = frame_height // 2
        mid_width = max_width_bbox // 2
        mid_height = max_height_bbox // 2

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (224, 224))

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
                y1_end = min(frame_height, y1 + mid_height)
                x1_start = max(0, x1 - mid_width)
                x1_end = min(frame_width, x1 + mid_width)
                movement_frame = frame[y1_start:y1_end, x1_start:x1_end]
            else:
                y1 = int(y1)
                x1 = int(x1)
                y1_start = max(0, y1 - mid_height)
                y1_end = min(frame_height, y1 + mid_height)
                x1_start = max(0, x1 - mid_width)
                x1_end = min(frame_width, x1 + mid_width)
                movement_frame = frame[y1_start:y1_end, x1_start:x1_end]
            if(movement_frame.shape[0] != 0 and movement_frame.shape[1] != 0):
                movement_frame = cv2.resize(movement_frame, (224, 224))
                cv2.imshow('result',movement_frame)
                out.write(movement_frame)

            k = cv2.waitKey(1)
            if k == 27:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
