# Face Recognition

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # load the cascade for the face.
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml') # load the cascade for the eyes.

def detect(gray, frame): 
    '''a function that takes as input the image in black and white (gray) and the original image (frame),
      and that will return the same image with the detector rectangles. '''
    faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.3, minNeighbors=5) # apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # paint a rectangle around the face.
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w] 
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor= 1.5, minNeighbors=15)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color,(sx, sy),(sx+sw, sy+sh), (0, 255, 0), 2) 
    return frame # return the image with the detector rectangles.

video_capture = cv2.VideoCapture(0) # turn the webcam on.

while True: 
    _, frame = video_capture.read() # get the last frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # do colour transformations.
    canvas = detect(gray, frame) # use detect function
    cv2.imshow('Video', canvas) # display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break # stop the loop if type "q".

video_capture.release() # turn the webcam off.
cv2.destroyAllWindows() # destroy all the windows inside which the images were displayed.