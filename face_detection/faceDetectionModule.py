import time
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0) #Webcam
# cap = cv2.VideoCapture("videos/guy.mp4")

class faceDetector(object):
    def __init__(self,minDetectionConfidence = 0.75):
        self.minDetectionConfidence = minDetectionConfidence
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionConfidence)

    def findFaces(self,img,draw=True):
        #Returns 
        #   bounding box information  (X,Y,W,H)
        #   id number
        #   score

        

        #Convert BGR TO RGB
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []


        if self.results.detections:
            for id,detection in enumerate(self.results.detections): #Find for a particular detection

                
                #print(id,detection)
                #print(detection.score)
                
                bboxC = detection.location_data.relative_bounding_box #bounding box coming from the class
                ih,iw,ic = img.shape
                bbox = (int(bboxC.xmin * iw),int(bboxC.ymin * ih),int(bboxC.width * iw),int(bboxC.height * ih)) 
                
                bboxs.append((id,bbox,detection.score[0]))
                
                img = self.fancyDraw(img,bbox)

                if draw:
                    self.mpDraw.draw_detection(img, detection)

                    # cv2.rectangle(img,bbox,(0,255,50),2)
                    cv2.putText(img,str(int(detection.score[0]*100)),(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,2,(0,255,80),2)
        
        return img,bboxs
    
    def fancyDraw(self,img,bbox,l=30,t=10):
        x,y,w,h = bbox
        x1,y1 = x+w,y+h
        cv2.line(img, (x,y),(x+l,y),(0,255,0),t) #corner
        cv2.line(img, (x,y),(x,y+l),(0,255,0),t) #corner
        return img
def main():
    
    detector = faceDetector()
    while True:
        succes,img = cap.read()
        img,bbox = detector.findFaces(img)
        print(bbox)
        #display
        cv2.imshow("Image",img)
        cv2.waitKey(1)            

if __name__ == "__main__":
    main()