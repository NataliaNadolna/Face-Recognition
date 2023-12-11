import numpy as np
import cv2
 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_eye.xml.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

class Recognition:
    def onCamera(self):
        cap = cv2.VideoCapture(0)
 
        while(True):
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
 
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(190,0,0),2)
 
            cv2.imshow('img',img)
            if cv2.waitKey(30) & 0xFF == ord('q'): 
                break
 
        cap.release()
        cv2.destroyAllWindows()

    def onImage(self, imgPath: str): # nie działa
        img = cv2.imread(imgPath)
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5)
 
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
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
 
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(190,0,0),2)
 
            cv2.imshow('img',img)
            if cv2.waitKey(30) & 0xFF == ord('q'): 
                break
 
        cap.release()
        cv2.destroyAllWindows()



    
recognise = Recognition()
#recognise.onCamera()
recognise.onImage("twarze\Natalia\363278905_331444039240104_4662878192679284882_n.jpg")
#recognise.onVideo("film.mp4")
 
