import time
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0) #Webcam
#cap = cv2.VideoCapture("videos/guy.mp4")

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()


while True:
    succes,img = cap.read()
    #Convert BGR TO RGB
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)
    
        

    if results.detections:
        for id,detection in enumerate(results.detections): #Find for a particular detection

            mpDraw.draw_detection(img, detection)
            
            #print(id,detection)
            #print(detection.score)
            
            bboxC = detection.location_data.relative_bounding_box #bounding box coming from the class
            ih,iw,ic = img.shape
            bbox = int(bboxC.xmin * iw),int(bboxC.ymin * ih),\
                int(bboxC.width * iw),int(bboxC.height * ih), 
            

            cv2.rectangle(img,bbox,(0,255,50),2)
    
    #display
    cv2.imshow("Image",img)
    cv2.waitKey(1)