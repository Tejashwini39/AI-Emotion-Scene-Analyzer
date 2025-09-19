import cv2
import numpy as np
from keras.models import load_model
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array
import tensorflow as tf

# Load emotion recognition model (pretrained FER2013 CNN model)
# You can also use 'fer' package directly, but here we show manual
emotion_model = load_model("emotion_model.h5")  # <-- need to download or train
emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# Load MobileNetV2 for scene/object recognition
scene_model = MobileNetV2(weights="imagenet")

# Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Emotion prediction
        roi = cv2.resize(roi_gray, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_model.predict(roi)[0]
        emotion_idx = np.argmax(preds)
        emotion_text = f"{emotion_labels[emotion_idx]} ({preds[emotion_idx]*100:.1f}%)"

        # Draw rectangle + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # Scene recognition (MobileNetV2) on whole frame
    resized = cv2.resize(frame, (224, 224))
    arr = img_to_array(resized)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    preds_scene = scene_model.predict(arr)
    label_scene = decode_predictions(preds_scene, top=1)[0][0][1]

    cv2.putText(frame, f"Scene: {label_scene}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Emotion & Scene Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
