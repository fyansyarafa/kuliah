 # -*- coding: utf-8 -*-

# Face Recognition

# Importing the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
happy_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Defining a function that will do the detections
def detect(gray, frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # 1.3 : size of the image will be reduced 1.3 times 
    # in order for a zone of pixels to be accepted, we all need to have at least a certain number of neighbor zones that are also accepted.
    # 5 : in order for a zone of pixels to be accepted, at least 5 neighbor zone must also tobe accepted
    #cv2.putText(frame,'Ekspresi:',(0,20), font, 0.5, (255,255,255)) 
    for (x,y,w,h) in faces:
        segiempat=cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 3) #rectangle untuk deteksi wajah
        #frame, gambar asli
        #(x+y), sudut kiri atas dari rectangle
        #argumen 3, sudut kanan bawah
        #argumen 4, warna (BGR)
        #argumen 5, ketebalan rectangle
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        happy = happy_cascade.detectMultiScale(roi_gray, 1.7, 22)
        cv2.putText(frame,'Wajah',(x+5,y-10), font, 0.5, (255,255,255)) 
        for (ex, ey, ew, eh) in happy:
            cv2.putText(frame,'Bahagia',(x+5,y+15), font, 0.5, (0, 255, 0)) 
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        #cv2.putText(frame,'Wajah',(x+5,y+15), font, 0.5, (255,255,255)) 
        #cv2.imshow('Face having name', frame)
         
    return frame
 
# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)  #argumen berisi 0, jika webcam bawaan pc, 1, jika webcam external
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # mengubah gambar berwarna menjadi grayscale.
    canvas = detect(gray, frame) # output dari fungsi detect
    
    
    cv2.imshow('Tugas Akhir 2', canvas) # displaying output
    if cv2.waitKey(1) & 0xFF == ord('q'): # keyword untuk meematikan proses
        break #  stop  loop.

video_capture.release() # mematikan webcam
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.
        
        
    