import handTrackingModule as htm
import time
import cv2
import math
import pycaw
import numpy as np

#######
wCam, hCam = 640, 480
#######

#####Pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
bar_volume = 0
#####

cap = cv2.VideoCapture(0)
cap.set(3,wCam) #set size of the detector
cap.set(4,hCam)


detector = htm.handDetector(min_detection_confidence=0.9)

while True:
    succes,img = cap.read()

    
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    

    if len(lmList) != 0:
        # cv2.putText(img,str(result[:3]), (10,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,70),3)#put text
        #print(lmList[4],lmList[4][2])

        x1,y1 = lmList[4][1],lmList[4][2]
        x2,y2 = lmList[8][1],lmList[8][2]
        cv2.circle(img,(x1,y1),5,(0,280,70),3) # index dot
        cv2.circle(img,(x2,y2),5,(0,280,100),3) # thumb dot
        cv2.line(img,(x1,y1),(x2,y2),(0,255,70),3) #line of index and thumb

        cx,cy = (x1 + x2)//2,(y1 + y2)//2 #dot in between
        cv2.circle(img,(cx,cy),5,(0,255,80),cv2.FILLED)
        


        lenght = math.hypot(x2-x1,y2-y1)
        cv2.putText(img,str(lenght), (10,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,70),3)#show lenght

        # Hand range = 50,300
        # Volume range = -65 -0
        volume_int = np.interp(lenght,[50,190],[minVol,maxVol])  #convert range
        print(volume_int)
        
        #updates the volume
        volume.SetMasterVolumeLevel(volume_int, None)

        #adds a bar
        cv2.rectangle(img,(50,150),(85,400),(0,240,80),3) #Img / Initial position / ending position / color / Thicness
        #fills the bar
        cv2.rectangle(img,(50,int(bar_volume)),(85,400),(0,240,80),cv2.FILLED)
        bar_volume = np.interp(lenght,[50,300],[400,0]) #definition goes below so that program can start at bar_volume=0

        if lenght<50:                        
            cv2.circle(img,(cx,cy),5,(0,55,240),cv2.FILLED)            
            


        
    
    #display
    cv2.imshow("Image",img)
    cv2.waitKey(1) 