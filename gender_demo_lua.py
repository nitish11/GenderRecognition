# @author : https://github.com/MohanaRC
# @description : Python wrapper for Detection of gender from camera using Lua

'''Uses lutorpy to call lua processes within a python function and finds the result of the gender code'''

import lutorpy
import numpy as np
import cv2
import time
import os

cv=require("cv") 
require ("loadcaffe")
require ("image")


#cv2 Haar Face detector
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

fx=0.5
M=227
gender_net = loadcaffe.load('./models/deploy_gender.prototxt', './models/gender_net.caffemodel')._float()
img_mean = torch.load('./models/age_gender_mean.t7')._permute(3,1, 2)._float()

def process_lua_code(image_path):
    if image_path!=None:
        loadType = cv.IMREAD_GRAYSCALE
        
        frame = cv2.imread(image_path)
        im = cv2.resize(frame,(256, 256), interpolation=cv2.INTER_AREA)
        im = np.reshape(im,(256,3,256))

        im2=im-img_mean
        im2 = np.reshape(im2,(256,256,3))

        I = cv2.resize(im2, (M, M),interpolation=cv2.INTER_AREA)
        I =np.reshape(I, (3, 227, 227))
        I=torch.fromNumpyArray(I)
        I=I._clone()

        gender_out = gender_net._forward(I)
        gender = gender_out[0] > gender_out[1] and 'Male' or 'Female'
        
        return gender




#Getting prediction from live camera
cap = cv2.VideoCapture(0)

while True:    
    ret,frame = cap.read()
    if ret is True:
        start_time = time.time()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

        #Finding the largest face
        if len(rects) >= 1:
            rect_area = [rects[i][2]*rects[i][3] for i in xrange(len(rects))]
            rect = rects[np.argmax(rect_area)]
            x,y,w,h = rect
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_color = frame[y:y+h, x:x+w]

            #Resizing the face image
            crop = cv2.resize(roi_color, (256,256))
            cv2.imwrite('temp.jpg',crop)
            
            
            #Getting the prediction
            start_prediction = time.time()
            gender = process_lua_code("temp.jpg")
            print("Time taken by DeepNet model: {}").format(time.time()-start_prediction)
            print gender
            cv2.putText(frame,gender,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
        
            print("Total Time taken to process: {}").format(time.time()-start_time)
        #Showing output
        cv2.imshow("Gender Detection",frame)
        cv2.waitKey(1) 

#Delete objects
cap.release()
cv2.killAllWindows()

