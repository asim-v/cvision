import cv2
import numpy as np 
import time 
import os
import handTrackingModule as htm


#######
brush_thickness = 15
eraser_thickness = brush_thickness*4
#######


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

#Initial draw color
drawColor = (0,0,0)

cap = cv2.VideoCapture(0)
cap.set(3,1280) #Set feature id 3 (width) to value 1280
cap.set(4,720) #Set height exactly to 720

#For landmarks
detector = htm.handDetector(min_detection_confidence = 0.60)

#For drawing
xp,yp = 0,0


imgCanvas = np.zeros((720,1280,3),np.uint8) #uint = 0-255 values


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
            xp,yp = 0,0 #This fixes the bug where the line was drawn on switch

            #cv2.rectangle(img,(x1,y1-15),(x2,y2+15),(105, 66, 245),cv2.FILLED) #Draw square
            cv2.line(img,(x1,y1),(x2,y2),drawColor,3)
            print("Selection mode")            
            
            #SELECT SOMETHIGN
            #We need to check if we're at the top of the image so we can change the 
            if y1 < 125: #If we're in the header
                header = overlay_list[0]
                if 250 < x1 < 450 : #It means it is clicking the first button
                    header = overlay_list[1] #The header is the first image
                    drawColor = (10,163,252)
                elif 550 < x1 < 750 : #It means it is clicking the first button
                    header = overlay_list[2] #The header is the second image
                    drawColor = (126, 217, 87)
                elif 800 < x1 < 950 : #It means it is clicking the first button
                    header = overlay_list[3] #The header is the third image
                    drawColor = (10, 10, 252)
                elif 1050 < x1 < 1200 : #It means it is clicking the first button
                    header = overlay_list[4] #The header is the fourth image
                    drawColor = (0,0,0)            

        # (5th step) Drawing mode, index finger is up
        if fingers[1] and fingers[2] == False:  
            cv2.circle(img,(x1,y1),5,drawColor,cv2.FILLED,5) #Draw dot BGR
            print("Drawing Mode")

            if xp==0 and yp==0:
                #first frame
                xp,yp = x1,y1

            #Drawing circles would have gaps, instead we'll draw lines
            # cv2.line(img,(xp,yp),(x1,y1),drawColor,brush_thickness)

            if drawColor == (0,0,0): #If eraser, then use big canvas
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraser_thickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraser_thickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brush_thickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brush_thickness)
            xp,yp = x1,y1

    #Converting the imgcanvas into a grey image
    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY) 
    #Convert into a binary image and also inverting it
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    #The idea is
    #   (1) Convert image into black and white
    #   (2) Wherever there's black i want it to be white
    #   (3) Wherever there's a color image i want it to be black
    #   (4) Will create a mask and with white except the drawings as black
    #   (5) In the original image, will make all of the previous image black 
    #   (6) The merged options are black 
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)  #cONVERTING IT BACK (Cannot add grey image to color image)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    ##Explication Due




    #(Normal Header) Overlay image -> Since it's a matrix we just need to define it's location
    img[0:125,0:1280] = header   #We just define the matrix content of img
    
    

    ###ADD BOTH IMAGES TOGETHER
    # img = cv2.addWeighted(img, 0.5,imgCanvas,0.5, 0)
    



    #Show video feed  
    cv2.imshow("Image",img)
    # cv2.imshow("Canvas",imgCanvas)
    cv2.waitKey(1)