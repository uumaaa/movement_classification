'''
    File name         : detectors.py
    Description       : Object detector used for detecting the objects in a video /image
    Author            : Rahmad Sadli
    Date created      : 20/02/2020
    Python Version    : 3.7
'''

# Import python libraries
import numpy as np
import cv2


def detect(frame,fgbg,kernel,debugMode=0,minArea = 100):
    x, y, w, h = -1, -1, -1, -1
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    if (debugMode):
        cv2.imshow('mask', fgmask)


    img_thresh_gaussian = cv2.adaptiveThreshold(fgmask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv2.THRESH_BINARY_INV,13,4)
    contours, _ = cv2.findContours(img_thresh_gaussian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if(len(contours) >=1):
        contour = max(contours, key=cv2.contourArea)
        if (cv2.contourArea(contour) > minArea):
            x, y, w, h = cv2.boundingRect(contour)
    return img_thresh_gaussian,np.array([[x],[y],[w],[h]])



