#Caffe implementation source:
#http://nbviewer.jupyter.org/url/www.openu.ac.il/home/hassner/projects/cnn_agegender/cnn_age_gender_demo.ipynb

import os
import numpy as np
import sys
import cv2
import time
import caffe


#Models root folder
models_path = "./models"

#Loading the mean image
mean_filename=os.path.join(models_path,'./mean.binaryproto')
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean_image  = caffe.io.blobproto_to_array(a)[0]

#Loading the gender network
gender_net_pretrained=os.path.join(models_path,'./gender_net.caffemodel')
gender_net_model_file=os.path.join(models_path,'./deploy_gender.prototxt')
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained)

#Reshaping mean input image
mean_image = np.transpose(mean_image,(2,1,0))

#Gender labels
gender_list=['Male','Female']

#cv2 Haar Face detector
face_cascade = cv2.CascadeClassifier(os.path.join(models_path,'haarcascade_frontalface_default.xml'))

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
            
            #Subtraction from mean file
            input_image = crop -mean_image

            #Getting the prediction
            start_prediction = time.time()
            prediction = gender_net.predict([input_image]) 
            gender = gender_list[prediction[0].argmax()]
            print("Time taken by DeepNet model: {}").format(time.time()-start_prediction)
            print prediction,gender
            cv2.putText(frame,gender,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
        
            print("Total Time taken to process: {}").format(time.time()-start_time)
        #Showing output
        cv2.imshow("Gender Detection",frame)
        cv2.waitKey(1) 

#Delete objects
cap.release()
cv2.killAllWindows()
