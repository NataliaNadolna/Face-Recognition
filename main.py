import numpy as np
import cv2

class Recognition:

    def __init__(self):
        self.model = cv2.CascadeClassifier('models/haarcascade_frontalface_alt2.xml')
        #self.model = cv2.CascadeClassifier('models/haarcascade_eye.xml.xml')
        #self.model = cv2.CascadeClassifier('models/haarcascade_smile.xml')

    def onCamera(self):
        cap = cv2.VideoCapture(0)
 
        while(True):
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.model.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
 
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(190,0,0),2)
 
            cv2.imshow('img',img)
            if cv2.waitKey(30) & 0xFF == ord('q'): 
                break
 
        cap.release()
        cv2.destroyAllWindows()

    def onImage(self, imgPath: str): # nie działa
        img = cv2.imread(imgPath)
        faces = self.model.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5)
 
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(190,0,0),2)
 
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def onVideo(self, videoPath: str):
        cap = cv2.VideoCapture(videoPath)
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
            return

        (sucess, img) = cap.read()

        while sucess:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.model.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
 
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(190,0,0),2)
 
            cv2.imshow('img',img)
            if cv2.waitKey(30) & 0xFF == ord('q'): 
                break
 
        cap.release()
        cv2.destroyAllWindows()

recognise = Recognition()
recognise.onCamera()
#recognise.onImage("twarze\Natalia\363278905_331444039240104_4662878192679284882_n.jpg")
#recognise.onVideo("film.mp4")
 
