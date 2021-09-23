#!/usr/bin/env python

import handTrackingModule as htm
import time
import cv2
import itertools
import time
from flask import Flask, Response, redirect, request, url_for


cap = cv2.VideoCapture(0)
detector = htm.handDetector()
succes,img = cap.read()
img = detector.findHands(img)
lmList = detector.findPosition(img)


app = Flask(__name__)

@app.route('/')
def index():
    if request.headers.get('accept') == 'text/event-stream':
        def events():
            while True:
                succes,img = cap.read()
                img = detector.findHands(img)
                lmList = detector.findPosition(img)
                if len(lmList) != 0:                
                    result = lmList[8]
                    x = int(round(result[3],2)*100)
                    y = int(round(result[4],2)*100)
                    # print(x,y)
                    yield "data: %s %d\n\n" % (x,y)
                    
                    
                else:
                    yield "data: %s %d\n\n" % (0,0)

                #display
                cv2.imshow("Image",img)
                cv2.waitKey(1)
                    
                
            # for i, c in enumerate(itertools.cycle('\|/-')):
            #     yield "data: %s %d\n\n" % (c, i)
            #     time.sleep(.1)  # an artificial delay


        return Response(events(), content_type='text/event-stream')
    return redirect(url_for('static', filename='index.html'))

if __name__ == "__main__":
    app.run(host='localhost', port=23423)