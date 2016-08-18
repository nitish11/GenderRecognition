# @author : github.com/nitish11
# @description : Python wrapper for Detection of gender from camera using Lua

import time
import subprocess
import cv2

def detect_gender(img_path):
    output = subprocess.check_output('th gender_detection.lua '+img_path, shell=True)
    gender = output.split('\n')[-2].split('\t')[0]
    return gender


if __name__ == "__main__":
    img_path = "tmp.jpg"

    #Starting the camera
    vid = cv2.VideoCapture(0)

    while(vid.isOpened()):
        start_time = time.time()
        ret, frame = vid.read()

        if ret==True:
            cv2.imwrite("tmp.jpg",frame)
            gender = detect_gender(img_path)
            print gender
            if gender == 'Male' or gender == 'Female':
                cv2.putText(frame,gender,(100,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
            print("Time taken = ",time.time()-start_time)
        cv2.imshow("Gender",frame)
        cv2.waitKey(1)
