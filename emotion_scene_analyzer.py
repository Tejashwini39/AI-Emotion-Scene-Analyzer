import cv2
from fer import FER
import tensorflow as tf
import numpy as np
scene_model = tf.keras.applications.MobileNetV2(weights="imagenet")
detector = FER(mtcnn=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    emotions = detector.detect_emotions(frame)
    if emotions:
        top_emotion, score = detector.top_emotion(frame)
        text = f"Emotion: {top_emotion} ({score*100:.1f}%)"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    resized = cv2.resize(frame, (224, 224))
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(resized, axis=0)
    )
    preds = scene_model.predict(img_array, verbose=0)
    scene_label = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0][1]
    scene_prob = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0][2]

    cv2.putText(frame, f"Scene: {scene_label} ({scene_prob*100:.1f}%)",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Emotion & Scene Analyzer", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

