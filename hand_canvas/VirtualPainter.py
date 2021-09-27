import cv2
import numpy as np 
import time 
import os
import handTrackingModule as htm

#########Open Files
folderPath = "header"
myList = os.listdir(folderPath)
#print(myList) # Prints the contents of the path relative to the file
overlay_list = [] #In this list, every content of the list gets saved
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}') #Read image from path FOLDER + IMAGE
    overlay_list.append(image)

print(len(overlay_list))#Check if read correctly
header = overlay_list[0]
#########Open Files

cap = cv2.VideoCapture(0)
cap.set(3,1280) #Set feature id 3 (width) to value 1280
cap.set(4,720) #Set height exactly to 720

#For landmarks
detector = htm.handDetector(min_detection_confidence = 0.90)


while True:
    # (1st step) Import image
    success,img = cap.read() 
    img = cv2.flip(img,1) #Flip image to have the same direction

    # (2nd step) Find Landmarks
    img = detector.findHands(img) 
    lmList = detector.findPosition(img,draw=False)
    if len(lmList) != 0:
        #print(lmList)
        
        # (2.5 step) Index and middle
        x1,y1 = lmList[8][1:3]#Tip of index finger         
        x2,y2 = lmList[12][1:3] #Top of middle finger

        # (3rd step) Check which fingers are up,  one finger up for drawing and two fingers up for not drawing
        fingers = detector.fingersUp()
        #print(fingers)

        # (4rd step) If selection node -> Two fingers are up 
        if fingers[1] and fingers[2]:
            #cv2.rectangle(img,(x1,y1-15),(x2,y2+15),(105, 66, 245),cv2.FILLED) #Draw square
            cv2.line(img,(x1,y1),(x2,y2),(105, 66, 245),3)
            print("Selection mode")            
        

        # (5th step) Drawing mode, index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img,(x1,y1),10,(255, 255, 255),cv2.FILLED) #Draw dot
            print("Drawing Mode")
        

    #(Normal Header) Overlay image -> Since it's a matrix we just need to define it's location
    img[0:125,0:1280] = header   #We just define the matrix content of img
    



    #Show video feed  
    cv2.imshow("Image",img)
    cv2.waitKey(1)