import numpy
import cv2
import face_recognition

train = face_recognition.load_image_file('images/modi.jpg')
train = cv2.cvtColor(train, cv2.COLOR_BGR2RGB)

face = face_recognition.face_locations(train)[0]

train_encode = face_recognition.face_encodings(train)[0]



test = face_recognition.load_image_file('images/modi2.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

test_encode = face_recognition.face_encodings(test)[0]
isSame = face_recognition.compare_faces([train_encode], test_encode)
print(isSame)
if isSame:
    cv2.rectangle(train, (face[3], face[0]), (face[1], face[2]), (255, 0, 255), 1)
    cv2.imshow('Modi', train)
    cv2.waitKey(0)