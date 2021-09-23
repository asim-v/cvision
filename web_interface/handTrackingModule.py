
#Palm detection
#Land landmarks (21)


import cv2
import mediapipe as mp
import time


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

    def findPosition(self,img,handNo=0,draw=True): #img parameter needed for weight and height
        lmList = []
        if self.results.multi_hand_landmarks:#if something is detected
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):# check their index number
                #print(id,lm) #prints the id of the landmark and the ratio of the image
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h) #turn  ratio into pixels
                #print(id, cx,cy)
                lmList.append([id,cx,cy,lm.x,lm.y])
                
                if draw:
                    cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)
        
        return lmList

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