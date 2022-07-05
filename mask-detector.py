import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade\haarcascade_frontalface_default.xml")
mask_detection = tf.keras.models.load_model("data-training\mask_detection1.h5")

txt_mask = "MASK ON"
txt_no_mask = "MASK OFF"
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.8

def predict(image):
    face_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_frame = cv2.resize(face_frame, (224, 224))
    face_frame = img_to_array(face_frame)
    face_frame = np.expand_dims(face_frame, axis = 0)
    face_frame = preprocess_input(face_frame)
    prediction = mask_detection.predict(face_frame)
    return prediction[0][0]

def detector(gray_image, frame):
    wajah = face_cascade.detectMultiScale(gray_image, 1.1, 5)
    for (x, y, w, h) in wajah:
        roi_color = frame[y:y+h, x:x+w]
        mask = predict(roi_color)
    
        if mask > 0.5:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, text = txt_mask, org = (x+50, y-100), fontFace = font, fontScale = scale, color = (0, 255, 0), thickness = 2)
        
        elif mask < 0.5:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, text = txt_no_mask, org = (x+50, y-100), fontFace = font, fontScale = scale, color = (0, 0, 255), thickness = 2)
    return frame

video_capture = cv2.VideoCapture(1)

while True:
    ret, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    detect = detector(gray_frame, frame)

    cv2.imshow("Mask Detector", detect)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv2.destroyAllWindows()