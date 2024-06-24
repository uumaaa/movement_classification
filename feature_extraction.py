
import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def apply_temporal_gradient(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        return None

    prev_bgr = cv2.split(prev_frame)
    temporal_gradient_frames = {ch: [] for ch in ['b', 'g', 'r']}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bgr = cv2.split(frame)
        for i, ch in enumerate(['b', 'g', 'r']):
            temporal_gradient = cv2.absdiff(bgr[i], prev_bgr[i])
            temporal_gradient_frames[ch].append(temporal_gradient)
        prev_bgr = bgr

    cap.release()
    return temporal_gradient_frames

def apply_lbp(channel_frame):
    radius = 1
    n_points = 8 * radius
    lbp_frame = local_binary_pattern(channel_frame, n_points, radius, method='uniform')
    return lbp_frame

def process_videos(input_folder):
    for video_name in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_name)
        if not os.path.isfile(video_path):
            continue

        # Apply Temporal Gradient
        temporal_gradient_frames = apply_temporal_gradient(video_path)

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            continue

        # Create window to display results
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            bgr = cv2.split(frame)
            gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)] * 3
            temp_grad_frames = [temporal_gradient_frames[ch].pop(0) if temporal_gradient_frames[ch] else np.zeros_like(bgr[i]) for i, ch in enumerate(['b', 'g', 'r'])]
            lbp_frames = [apply_lbp(temp_grad_frame) for temp_grad_frame in temp_grad_frames]

            combined_frame = np.vstack((
                np.hstack([cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR) for ch in bgr]),                 # Gray frames
                np.hstack([cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR) for ch in temp_grad_frames]),   # Temporal Gradient frames
                np.hstack([cv2.cvtColor(np.uint8(lbp_frame), cv2.COLOR_GRAY2BGR) for lbp_frame in lbp_frames]) # LBP frames
            ))

            cv2.imshow('BGR Channels | Temporal Gradient | LBP', combined_frame)
            k = cv2.waitKey(30)
            if k == 27:  # Press 'ESC' to exit
                break

        cap.release()
    cv2.destroyAllWindows()

# Define the input folder containing the videos
input_folder = 'cleaned_dataset/train/HandstandPushups'

# Process the videos
process_videos(input_folder)
