
#Palm detection
#Land landmarks (21)

#LANDMARKS : https://drive.google.com/file/d/1KgnKtC1lvsSIj0a-Of2d5FL4MuZgOWgJ/view?usp=sharing

import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self,
               static_image_mode=False,
               max_num_hands=2,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode,self.max_num_hands,self.min_detection_confidence)
        self.mpDraw = mp.solutions.drawing_utils #Draws the points
    
        self.tipIds = [4,8,12,16,20] #TipIds

    def findHands(self,img, draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Converts img to RGB
        self.results = self.hands.process(imgRGB) #Processes the information  returns the hand landmarks and handedness of each detected hand
        
        
        #print(results.multi_hand_landmarks) #to check if something is detected or not

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # mpDraw.draw_landmarks(img,handLms) #Original image + Single Hand
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS) #Original image + Single Hand + HandConnections
        return img

    def findPosition(self,img,handNo=0,draw=False): #img parameter needed for weight and height
        self.lmList = []
        if self.results.multi_hand_landmarks:#if something is detected
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):# check their index number
                #print(id,lm) #prints the id of the landmark and the ratio of the image
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h) #turn  ratio into pixels
                #print(id, cx,cy)
                self.lmList.append([id,cx,cy,lm.x,lm.y])
                
                if draw:
                    cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)
        
        return self.lmList

    def fingersUp(self):
        fingers = []
        

        #Thumb -> Checks if tip of thumb is on the right or the left says if thumb is open or closed
        if self.lmList[self.tipIds[0]][1]>self.lmList[self.tipIds[0] -1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        #Fingers -> If the tip of the finger is above the other landmark (2 landmarks below)
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
        
    def findDistance(self, p1, p2, img, draw=True,r=5, t=3):
        '''
        P1 = Integer of initial point
        P2 = Integer of final point
        Returns:
        lenght,image,list with initial,end,middle dots
        '''

        x1, y1 = self.lmList[p1][1:3]
        x2, y2 = self.lmList[p2][1:3]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 100, 0), t)
            cv2.circle(img, (x1, y1), r, (255, 100, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 100, 0), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 100, 0), cv2.FILLED)
            
    
        return length, img, [x1, y1, x2, y2, cx, cy]
            
        
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        succes,img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time() #current time
        fps = 1/(cTime-pTime) #fps
        pTime = cTime

        #put text
        cv2.putText(img,str(int(fps)), (10,70),cv2.FONT_HERSHEY_DUPLEX,3,(255,0,255))
        # img / what text / positon / font / scale/ color / thicness

        #display
        cv2.imshow("Image",img)
        cv2.waitKey(1) 
    
if __name__ == "__main__":
    main()