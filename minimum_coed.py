
#Palm detection
#Land landmarks (21)


import cv2
import mediapipe as mp
import time


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils #Draws the points

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
            
    cv2.imshow("Image",img)
    cv2.waitKey(1)