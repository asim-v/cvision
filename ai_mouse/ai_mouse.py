import handTrackingModule as htm
import cv2
import autopy 
import numpy as np
import math

#############
wCam,hCam = 640,480
wScr,hScr = autopy.screen.size() #Size and height of the screen
frameR = 150 #Reduction of detection frame
mode = 'STABLE' #STABLE or PARALLEL
smoothening = 5
#############



print("Screen width:",wScr," Screen height:",hScr)

# This values get updated each iteration to smoothen mouse
plocX,plocY = 0,0 #previous location
clocX,clocY = 0,0 #current location 



cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

detector = htm.handDetector(max_num_hands=1)

while True:
    # Steps for project
    # 1. Find landmarks
    succes,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(0,255,90),5)#Bound box 
    if len(lmList) != 0:
        # 2. Get the tip of the index and middle fingers 
        x1,y1 = lmList[8][1:3] #INDEX FINGER
        x2,y2 = lmList[12][1:3] #MIDDLE FINGER
        # x3,y3 = lmList[4][1:3] #Thumb
        
    
        # 3. Check which of the fingers are up
        fingers = detector.fingersUp()        
        
        if mode == "STABLE":

            # 4. Only index finger is up        
            if fingers[1] == 1 and fingers[2] == 0:

                # Makes bounding box limit smaller
                x3 = int(np.interp(x1,(frameR,wCam-frameR),(0,wScr))) #Convert range of x1
                y3 = int(np.interp(y1,(frameR,hCam-frameR),(0,hScr))) #Convert range of y1
                cv2.circle(img,(x1,y1),10,(0,255,80),5) #Center needs to be int

                #Make movement soft
                clocX = plocX+(x3-plocX)/smoothening
                clocY = plocY+(y3-plocY)/smoothening
                autopy.mouse.move(wScr-clocX,clocY) #Inversed
                #Update location values
                plocX,plocY = clocX,clocY

            # Both fingers are up (CLICKIGN MODE)        
            if fingers[1] == 1 and fingers[2] == 1:
                lenght,img,lineInfo = detector.findDistance(8,12,img) #Modifies the image so that there's a distance between the two
                
                if lenght<40:
                    cv2.circle(img,(lineInfo[4],lineInfo[5]),10,(255,255,255),5) #Center needs to be int #LineInfo[4] is the middle dot returned from the findDistance function
                    autopy.mouse.click()
                    

        if mode == "PARALLEL":
            # 4. Only index finger is up        
            if fingers[1] == 1:

                # Makes bounding box limit smaller
                x3 = int(np.interp(x1,(frameR,wCam-frameR),(0,wScr))) #Convert range of x1
                y3 = int(np.interp(y1,(frameR,hCam-frameR),(0,hScr))) #Convert range of y1
                cv2.circle(img,(x1,y1),10,(0,255,80),5) #Center needs to be int

                #Make movement soft
                clocX = plocX+(x3-plocX)/smoothening
                clocY = plocY+(y3-plocY)/smoothening
                autopy.mouse.move(wScr-clocX,clocY) #Inversed
                #Update location values
                plocX,plocY = clocX,clocY

                # Both fingers are up (CLICKIGN MODE)        
                if fingers[1] == 1 and fingers[2] == 1:
                    lenght,img,lineInfo = detector.findDistance(8,12,img) #Modifies the image so that there's a distance between the two
                    
                    if lenght<40:
                        cv2.circle(img,(lineInfo[4],lineInfo[5]),10,(255,255,255),5) #Center needs to be int #LineInfo[4] is the middle dot returned from the findDistance function
                        autopy.mouse.click()
                        


        


            
        
            
    
 
    



    
    #display
    cv2.imshow("Image",img)
    cv2.waitKey(1) 