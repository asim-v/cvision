import cv2
import mediapipe as mp 
import numpy as np

cap = cv2.VideoCapture('videos/guy.mp4')
if (cap.isOpened()== False):
    print("Error opening video stream or file")

 

while cap.isOpened():
    sucess,img = cap.read()
    if sucess:
        cv2.imshow("Image",img)
        cv2.waitKey(1)
    else:break

cap.release()
cv2.destroyAllWindows()