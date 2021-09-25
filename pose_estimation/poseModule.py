import cv2
import mediapipe as mp 
import numpy as np
import time



  



class PoseDetector(object):
    def __init__(self,
               static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
    
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode,self.model_complexity,self.smooth_landmarks,self.enable_segmentation,self.smooth_segmentation,self.min_detection_confidence,self.min_tracking_confidence) 
    """Initializes a MediaPipe Pose object.

    Args:
      static_image_mode: Whether to treat the input images as a batch of static
        and possibly unrelated images, or a video stream. See details in
        https://solutions.mediapipe.dev/pose#static_image_mode.
      model_complexity: Complexity of the pose landmark model: 0, 1 or 2. See
        details in https://solutions.mediapipe.dev/pose#model_complexity.
      smooth_landmarks: Whether to filter landmarks across different input
        images to reduce jitter. See details in
        https://solutions.mediapipe.dev/pose#smooth_landmarks.
      enable_segmentation: Whether to predict segmentation mask. See details in
        https://solutions.mediapipe.dev/pose#enable_segmentation.
      smooth_segmentation: Whether to filter segmentation across different input
        images to reduce jitter. See details in
        https://solutions.mediapipe.dev/pose#smooth_segmentation.
      min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for person
        detection to be considered successful. See details in
        https://solutions.mediapipe.dev/pose#min_detection_confidence.
      min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
        pose landmarks to be considered tracked successfully. See details in
        https://solutions.mediapipe.dev/pose#min_tracking_confidence.
    """
    def findPose(self,img,draw=True):

        ##Show color based on visibility
        def rgb(minimum, maximum, value):
            minimum, maximum = float(minimum), float(maximum)
            ratio = 2 * (value-minimum) / (maximum - minimum)
            b = int(max(0, 255*(1 - ratio)))
            r = int(max(0, 255*(ratio - 1)))
            g = 255 - b - r
            return r, g, b

        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #mediapipe uses rgb and the image is on bgr
        self.results = self.pose.process(imgRGB) #Returns results object which has the tuple of tuples landmarks with x,y,z,visibility
        if self.results.pose_landmarks:  
            if draw:                
                # Organize so that it is in a list
                # Landmarks: https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png
                for id,lm in enumerate(self.results.pose_landmarks.landmark):
                    h,w,c = img.shape #Height and width of the image
                    #print(lm,id,end='\n')  #Print landmark

                    cx,cy = int(lm.x*w),int(lm.y*h)  #Multiply result with width and height, how to access attributes

                    color = rgb(0,1,lm.visibility) #use draw function to generate rgb for cv2
                    cv2.circle(img,(cx,cy),10,color,cv2.FILLED,15)  #Draw points                
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS) #Will fill the connections
        return img

    def findPosition(self,img):
        lmList = []
        if self.results.pose_landmarks:            
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape #Height and width of the image
                #print(lm,id,end='\n')  #Print landmark

                cx,cy = int(lm.x*w),int(lm.y*h)  #Multiply result with width and height, how to access attributes
                lmList.append([id,cx,cy,lm.x,lm.y,lm.z,lm.visibility])
        return lmList
                
    

def main():
    ###Flower
    cap = cv2.VideoCapture('Videos/guy.mp4')
    pTime = 0
    ###Time

    detector = PoseDetector()


    while True:
        success,img = cap.read()    
        img = detector.findPose(img) #takes as input image, makes image with marks as output
        LmList = detector.findPosition(img) #image as input, list of data as output


        #measure fps
        print(LmList[14],end='\n')
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime 
        #put fps
        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(0,255,90))

        #show image
        cv2.imshow("Image",img)
        cv2.waitKey(1) #Increment to reduce 



if __name__ == "__main__":
    main()