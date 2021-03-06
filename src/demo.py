#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--no_camera", action="store_true", help="Process this program on local video")
parser.add_argument("--scaleFactor",  type=float, default=1.3, help="The scaleFactor of detectMultiScale")
parser.add_argument("--minNeighbors",  type=int, default=2, help="The minNeighbors of detectMultiScale")
parser.add_argument("--minSize",  type=int, default=20, help="The minSize of detectMultiScale")
args = parser.parse_args()

face_cascade = cv2.CascadeClassifier('../opencv_classifier/haarcascade_frontalface_default.xml')
model = load_model('../model/my_model.h5')

def filter(image, mode):
    image_copy = np.copy(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 2, minSize = (20, 20))
    
    for (x,y,w,h) in faces:
        roi_color = image_copy[y:y+h, x:x+w]
        resize_roi_gray = cv2.resize(image_gray[y:y+h, x:x+w], (96, 96)) / 255.
        landmarks = np.squeeze(model.predict(np.expand_dims(np.expand_dims(resize_roi_gray, axis=-1), axis=0)))
        
        if mode==1:
            sunglasses = cv2.imread('../filter_image/sunglasses.png', cv2.IMREAD_UNCHANGED)
            resize_sunglasses = cv2.resize(sunglasses, (w, w//3), interpolation = cv2.INTER_CUBIC)
            ind = np.argwhere(resize_sunglasses[:,:,3] > 0)
            for i in range(3):
                roi_color[ind[:,0]+int((landmarks[19]+1)*roi_color.shape[1]/2),ind[:,1],i] = resize_sunglasses[ind[:,0],ind[:,1],i]
            image_copy[y:y+h,x:x+w] = roi_color
        
        elif mode==2:
            cv2.circle(image_copy, (int((landmarks[0]+1)*roi_color.shape[0]/2+x),int((landmarks[1]+1)*roi_color.shape[1]/2+y)), w//4, (41, 36, 33), -1)
            cv2.circle(image_copy, (int((landmarks[0]+1)*roi_color.shape[0]/2+x),int((landmarks[1]+1)*roi_color.shape[1]/2+y)), max(1, w//25), (255, 255, 255), -1)
            cv2.circle(image_copy, (int((landmarks[2]+1)*roi_color.shape[0]/2+x),int((landmarks[3]+1)*roi_color.shape[1]/2+y)), w//4, (41, 36, 33), -1)
            cv2.circle(image_copy, (int((landmarks[2]+1)*roi_color.shape[0]/2+x),int((landmarks[3]+1)*roi_color.shape[1]/2+y)), max(1, w//25), (255, 255, 255), -1)
        
        elif mode==3:
            Takeshi = cv2.imread('../filter_image/Takeshi.png', cv2.IMREAD_UNCHANGED)
            resize_Takeshi = cv2.resize(Takeshi, (w, h), interpolation = cv2.INTER_CUBIC)
            ind = np.argwhere(resize_Takeshi[:,:,3] > 0)
            for i in range(3):
                roi_color[ind[:,0],ind[:,1],i] = resize_Takeshi[ind[:,0],ind[:,1],i]
            image_copy[y:y+h,x:x+w] = roi_color
        
        elif mode==4:
            Bamboo = cv2.imread('../filter_image/Bamboo.png', cv2.IMREAD_UNCHANGED)
            resize_Bamboo = cv2.resize(Bamboo, (4*w//5, w//4), interpolation = cv2.INTER_CUBIC)
            ind = np.argwhere(resize_Bamboo[:,:,3] > 0)
            for i in range(3):
                roi_color[ind[:,0]+int((landmarks[27]+1)*roi_color.shape[1]/2)-w//8,ind[:,1]+w//8,i] = resize_Bamboo[ind[:,0],ind[:,1],i]
            image_copy[y:y+h,x:x+w] = roi_color
        
        elif mode==0:
            for i in range(0, len(landmarks), 2):
                cv2.circle(image_copy, (int((landmarks[i]+1)*roi_color.shape[0]/2+x),int((landmarks[i+1]+1)*roi_color.shape[1]/2+y)), 1, (0,255,0), -1)
    
    if mode==1:
        image_copy = cv2.cvtColor(image_copy, 72)
    elif mode==2:
        image_copy = cv2.cvtColor(image_copy, 53)
    elif mode==3:
        image_copy = cv2.cvtColor(image_copy, 32)
    elif mode==4:
        image_copy = cv2.cvtColor(image_copy, 2)
    elif mode==0:
        image_copy = cv2.cvtColor(image_copy, 1)
    return image_copy

if __name__ == '__main__':
    cv2.namedWindow('face detection activated', cv2.WINDOW_KEEPRATIO)
    vc = cv2.VideoCapture('../video/sample_video.mp4') if (args.no_camera) else cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    assert vc.isOpened()
    
    #keypoint, sunglasses, eyes, Takeshi, Bamboo
    Num_mode = 4
    mode = 0
    while 1:
        rval, frame = vc.read()
        frame = filter(frame, mode)
        cv2.imshow('face detection activated', frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key & 0xFF == ord('w'):
            mode = Num_mode if mode==0 else mode-1
        elif key & 0xFF == ord('e'):
            mode = 0 if mode==Num_mode else mode+1
        elif key & 0xFF == ord('a'):
            mode = 0
        elif key & 0xFF == ord('s'):
            mode = 1
        elif key & 0xFF == ord('d'):
            mode = 2
        elif key & 0xFF == ord('f'):
            mode = 3
        elif key & 0xFF == ord('g'):
            mode = 4
    