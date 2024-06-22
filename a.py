import numpy as np
import cv2
import os
from KalmanFilter import KalmanFilter
from Detector import detect

'''def get_videos_from_classes(dataset_path, classes_to_include):
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d in classes_to_include]

    all_videos = []
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        video_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        for video_file in video_files:
            all_videos.append((os.path.join(class_path, video_file), class_name))

    return all_videos

def load_videos(video_list):
    X = []
    y = []

    for video_path, class_name in video_list:
        # Read the video using cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            continue

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cv2tColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame_gray)

        cap.release()

        X.append(frames)
        y.append(class_name)

    return X, y


dataset_path = 'dataset/test'
classes_to_include = ['Skiing', 'Surfing', 'PullUps']  
video_list = get_videos_from_classes(dataset_path, classes_to_include)
X, y = load_videos(video_list)'''

class_name = 'GolfSwing/'
for video in os.listdir('dataset/test/'+class_name):
    cap = cv2.VideoCapture('dataset/test/'+class_name+video)
    ControlSpeedVar = 1  #Lowest: 1 - Highest:100
    HiSpeed = 100
    KF = KalmanFilter(0.1, 1, 1, 1, 0.4,0.4)
    debugMode=1
    kernel = np.ones((5,5),dtype=np.uint8)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    i = 0
    height = -1
    width = -1
    while(1):
        ret, frame = cap.read()
        if not ret:
            break
        if(i == 0):
            movement_frame = frame[:][:]
            height = frame.shape[1]
            width = frame.shape[0]
            max_width_bbox = -1
            max_height_bbox = -1
        imgs_thresh,center = detect(frame,fgbg,kernel,debugMode)
        if (center[0] != -1):
            cv2.rectangle(frame, (int(center[0]), int(center[1])), (int(center[0] + center[2]), int(center[1] + center[3])), (0,191,255), 2)
            (x, y) = KF.predict()
            (x1, y1) = KF.update(np.array([center[0]+center[2]/2,center[1]+center[3]/2]))
            if((center[0] < x1 < center[0] + center[2]) and (center[1] < y1 < center[1] + center[3])):
                max_width_bbox = max(max_width_bbox,int(center[2]))
                max_height_bbox = max(max_height_bbox,int(center[3]))
            cv2.circle(frame, (int((x1)), int(y1)),15, (0, 0, 255), 2)
            cv2.putText(frame, "Estimated Position", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Measured Position", (int(center[0] + 15), int(center[1] - 15)), 0, 0.5, (0,191,255), 2)
        i+=1
        cv2.imshow('Objects Tracked', np.hstack((frame,cv2.resize(movement_frame,(height,width)),cv2.cvtColor(imgs_thresh,cv2.COLOR_GRAY2RGB))))
        k = cv2.waitKey(HiSpeed-ControlSpeedVar+1)
        if k == 27:
            break
    print(max_width_bbox)
    print(max_height_bbox)
    cap.release()
    cv2.destroyAllWindows()