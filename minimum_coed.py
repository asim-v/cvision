
#Palm detection
#Land landmarks (21)


import cv2
import mediapipe as mp
import time


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils #Draws the points

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)

while True:
    succes,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Converts img to RGB
    results = hands.process(imgRGB) #Processes the information  returns the hand landmarks and handedness of each detected hand
    
    
    print(results.multi_hand_landmarks) #to check if something is detected or not

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # mpDraw.draw_landmarks(img,handLms) #Original image + Single Hand
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS) #Original image + Single Hand + HandConnections
            
    cTime = time.time() #current time
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70),cv2.FONT_HERSHEY_DUPLEX,3,(255,0,255))


    #display
    cv2.imshow("Image",img)
    cv2.waitKey(1)