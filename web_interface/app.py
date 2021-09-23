#!/usr/bin/env python

import handTrackingModule as htm
import time
import cv2
import itertools
import time
from flask import Flask, Response, redirect, request, url_for


cap = cv2.VideoCapture(0)
detector = htm.handDetector()



app = Flask(__name__)

@app.route('/')
def index():
    if request.headers.get('accept') == 'text/event-stream':
        def events():
            for i in range(10000):
                succes,img = cap.read()
                img = detector.findHands(img)
                lmList = detector.findPosition(img)
                if len(lmList) != 0:                
                    result = lmList[8]
                                        
                    yield "data: %s %d\n\n" % (result[1],result[2])
                    
                    time.sleep(.1)  # an artificial delay
                else:
                    yield "no hand detected"
                
            # for i, c in enumerate(itertools.cycle('\|/-')):
            #     yield "data: %s %d\n\n" % (c, i)
            #     time.sleep(.1)  # an artificial delay

        return Response(events(), content_type='text/event-stream')
    return redirect(url_for('static', filename='index.html'))

if __name__ == "__main__":
    app.run(host='localhost', port=23423)