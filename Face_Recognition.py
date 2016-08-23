#Face Recognition source
#https://github.com/cmusatyalab/openface

import argparse
import cv2
import os
import pickle
import dlib

import numpy as np
np.set_printoptions(precision=2)
import openface


#Setting up models directory path
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
classiferModelDir = os.path.join(modelDir, 'classifier')

networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.ascii.t7')
classifierModel = os.path.join(classiferModelDir,'classifier.pkl')
dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")

#Variables for openface model 
imgDim = 96
cuda = False


#Loading models

#Setting up openface libraries
align = openface.AlignDlib(dlibFacePredictor)
net = openface.TorchNeuralNet(networkModel, imgDim=imgDim, cuda=cuda)

#Setting up dlib variables for face detection and keypoints extraction
dlib_predictor_path = dlibFacePredictor
dlib_predictor = dlib.shape_predictor(dlib_predictor_path)
dlib_detector = dlib.get_frontal_face_detector()

#Load classification model
with open(classifierModel, 'r') as f:
    (le, clf) = pickle.load(f)


def get_representations(alignedFace):
    '''
    Given, network model and alignedFace
    Returns representations of the Face from forward pass neural network
    '''
    #Get represntations from trained neural net model for the aligned face
    rep = net.forward(alignedFace)
    print "Representations : ",rep
    return rep.reshape(1, -1)


def getAlignFace(img):
    '''
    Given image, detect largest face bouding box
    Returns aligned image 
    '''
    global align
    bgrImg = img
    if bgrImg is None:
        print("Unable to load image")
        return None
    else:
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        #Get largest face bounding box from the base image
        bb = align.getLargestFaceBoundingBox(rgbImg)
        if bb is None:
            print("Unable to find a face")
            return None
        else:
            #Get aligned face from 
            alignedFace = align.align(imgDim, rgbImg, bb,
                                         landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                print("Unable to align image")
                return None
            else:
                return alignedFace



def face_recognition(frame):
    '''
    Given frame, get the aligned image
    Returns predicted person 
    '''
    #Get Aligned face from the frame
    alignedFace = getAlignFace(frame)
    
    
    if alignedFace is None:
        return
    else:
        #Get represntations from trained neural net model for the alignedFace
        rep = get_representations(alignedFace)

        #Get prediction from the trained classification model
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]

	print("--------",person,confidence)
	cv2.imwrite("frame.jpg",frame)
        
        if confidence > 0.5:
            return person
        else:
            return "Unknown"


def test_model_camera():
    '''
    Given arguments of the trained mode, get images from the live camera 
    Return perdiction of the face
    '''
    
    #Starting the camera
    vid = cv2.VideoCapture(0)
    
    while(vid.isOpened()):
        ret, frame = vid.read()

        if ret==True:
            face_recognition(frame)        
            
    vid.release()


#Live camera implementation
test_model_camera()