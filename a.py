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


cap = cv2.VideoCapture('dataset/test/Archery/v_Archery_g25_c06.avi')
ControlSpeedVar = 1  #Lowest: 1 - Highest:100
HiSpeed = 100
KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)
debugMode=1
kernel = np.ones((3,3),dtype=np.uint8)
fgbg = cv2.createBackgroundSubtractorMOG2()
while(1):
    ret, frame = cap.read()
    if not ret:
        break
    imgs_thresh,center = detect(frame,fgbg,kernel,debugMode)
    if (center[0] != -1):
        cv2.rectangle(frame, (int(center[0]), int(center[1])), (int(center[0] + center[2]), int(center[1] + center[3])), (0,191,255), 2)

        # Predict
        (x, y) = KF.predict()
        cv2.circle(frame, (int((x)), int(y)),15, (255, 0, 0), 2)

        # Update
        (x1, y1) = KF.update(np.array([(center[0]+center[2])/2,center[1]+center[3]/2],dtype=np.uint16))

        # Draw a rectangle as the estimated object position
        #cv2.rectangle(frame, (int(x1 - 15), int(y1 - 15)), (int(x1 + 15), int(y1 + 15)), (0, 0, 255), 2)

        #cv2.putText(frame, "Estimated Position", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Predicted Position", (int(x + 15), int(y)), 0, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "Measured Position", (int(center[0] + 15), int(center[1] - 15)), 0, 0.5, (0,191,255), 2)


    cv2.imshow('Objects Tracked', np.hstack((frame,cv2.cvtColor(imgs_thresh[0],cv2.COLOR_GRAY2RGB),cv2.cvtColor(imgs_thresh[1],cv2.COLOR_GRAY2RGB),cv2.cvtColor(imgs_thresh[2],cv2.COLOR_GRAY2RGB))))
    k = cv2.waitKey(HiSpeed-ControlSpeedVar+1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()