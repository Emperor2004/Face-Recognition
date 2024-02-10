# Required Modules
import cv2

# Variables to store the cascade data of HaarCascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)           # Variable to store video data

#COLORS Octal Values
blue = (0, 0, 255)
magenta = (255, 0, 255)

if cam.isOpened:
    while True:
        ret, frame = cam.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for x,y,w,h in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), magenta, 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
        cv2.imshow("Frame", frame)
        
        if (cv2.waitKey(30) & 0xff) == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()