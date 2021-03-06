import cv2
import mediapipe as mp 
import numpy as np
import time
from poseModule import PoseDetector

###Webcam
cap = cv2.VideoCapture(0)
###Flower
# cap = cv2.VideoCapture('videos/flower.mp4')
pTime = 0
###Time

detector = PoseDetector()


while True:
    success,img = cap.read()    
    img = detector.findPose(img) #takes as input image, makes image with marks as output
    LmList = detector.findPosition(img) #image as input, list of data as output

    try:
        print(LmList[14],end='\n')
    except:
        ("Empty LmList")
    #measure fps    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime 
    #put fps
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(0,255,90))

    #show image
    cv2.imshow("Image",img)
    cv2.waitKey(1) #Increment to reduce 
