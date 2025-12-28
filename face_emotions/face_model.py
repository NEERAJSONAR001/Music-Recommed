import cv2
import numpy as np
from tensorflow.keras.models import load_model

from collections import deque



model = load_model("face_emotions/face_emotion_cnn_model.h5")

emotion_labels = [
    'angry',     # 0
    'disgust',   # 1
    'fear',      # 2
    'happy',     # 3
    'neutral',   # 4
    'sad',       # 5
    'surprise'   # 6
]

emotion_queue = deque(maxlen=10)

face_cascade = cv2.CascadeClassifier(
    "face_emotions/haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

print("Webcam started. Press 'q' to quit.")




while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        prediction = model.predict(face)
        emotion_queue.append(prediction[0])
        avg_prediction = np.mean(emotion_queue, axis=0)
        emotion_index = np.argmax(avg_prediction)
        emotion = emotion_labels[emotion_index]

        

        # Draw rectangle and emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

    cv2.imshow("Real-Time Face Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
