from ultralytics import YOLO
from imutils.video import VideoStream
from keras.models import load_model
import cv2 as cv
import cvzone
import math
import skimage
import pandas as pd
import numpy as np

#==== BACA MODEL YOLO ==== (GANTI PATH MODEL best.pt KE MODEL PUNYA LU)
modelYOLO = YOLO('best.pt')

#Generate Class (Ganti classnames dari A sampai Z)
classnames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

cap = VideoStream(src=0).start()
while True :
    frame = cap.read()
    frame = cv.resize(frame, (640, 480))

    # Prediksi gambar
    result = modelYOLO(source=frame, stream=True)

    # Menampilkan
    for info in result :
        boxes = info.boxes
        for box in boxes :
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 20 :
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5, thickness=2)

    cv.imshow('Klasifikasi', frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv.destroyAllWindows()
cap.stop()