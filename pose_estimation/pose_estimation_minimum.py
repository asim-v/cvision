import cv2
import mediapipe as mp 
import numpy as np
import time


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose() 
    # static_image_mode:
        # True: Detection -> Tracking
        # False: Tracking
    # upper_body_pose
        # 33 vs 25
    # smooth_landmarks


###Flower
cap = cv2.VideoCapture('Videos/flower.mp4')
pTime = 0
###Time

while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #mediapipe uses rgb and the image is on bgr
    results = pose.process(imgRGB) #Returns results object which has the tuple of tuples landmarks with x,y,z,visibility
    if results.pose_landmarks:  
        # Organize so that it is in a list
        # Landmarks: https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape #Height and width of the image
            print(lm,id,end='\n')  #Print landmark

            cx,cy = int(lm.x*w),int(lm.y*h)  #Multiply result with width and height, how to access attributes

            ##Show color based on visibility
            def rgb(minimum, maximum, value):
                minimum, maximum = float(minimum), float(maximum)
                ratio = 2 * (value-minimum) / (maximum - minimum)
                b = int(max(0, 255*(1 - ratio)))
                r = int(max(0, 255*(ratio - 1)))
                g = 255 - b - r
                return r, g, b
            color = rgb(0,1,lm.visibility)
            cv2.circle(img,(cx,cy),10,color,cv2.FILLED,15)  #Draw points

            # #Only draw points
            # cv2.circle(img,(cx,cy),10,(0,255,80),cv2.FILLED,10)  #Draw points

        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS) #Will fill the connections

    #measure fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime 

    #put fps
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(0,255,90))





    #show image
    cv2.imshow("Image",img)
    cv2.waitKey(1) #Increment to reduce 