import handTrackingModule as htm
import time
import cv2


cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    succes,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        result = lmList[4]
        cv2.putText(img,str(result), (10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0))#put text
        

    
    #display
    cv2.imshow("Image",img)
    cv2.waitKey(1) 