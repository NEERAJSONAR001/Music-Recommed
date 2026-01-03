import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from collections import deque


model = load_model("face_emotions/face_emotion_cnn_model.h5")

emotion_labels = [
    'angry', 'disgust', 'fear',
    'happy', 'neutral', 'sad', 'surprise'
]


songs_df = pd.read_csv("data/songs.csv")


def regulate_mood(emotion):
    regulation_map = {
        "sad": "calm",
        "fear": "calm",
        "angry": "calm",
        "neutral": "happy",
        "happy": "happy",
        "surprise": "happy"
    }
    return regulation_map.get(emotion, "calm")

def get_all_songs(emotion):
    target_mood = regulate_mood(emotion)
    filtered = songs_df[songs_df["emotion"] == target_mood]
    return filtered, target_mood


emotion_queue = deque(maxlen=10)

face_cascade = cv2.CascadeClassifier(
    "face_emotions/haarcascade_frontalface_default.xml"
)


cap = cv2.VideoCapture(0)
print("\n🎥 Webcam started. Press 'q' to quit.\n")

last_printed_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        prediction = model.predict(face, verbose=0)
        emotion_queue.append(prediction[0])

        avg_prediction = np.mean(emotion_queue, axis=0)
        emotion = emotion_labels[np.argmax(avg_prediction)]

        # Get songs
        songs, target_mood = get_all_songs(emotion)

        # Print songs ONLY when emotion changes
        if emotion != last_printed_emotion:
            print(f"\nDetected Emotion: {emotion}")
            print(f"Target Mood    : {target_mood}")
            print("Recommended Songs:")

            if songs.empty:
                print("  No songs found.")
            else:
                for _, row in songs.iterrows():
                    print(f"  - {row['song_name']} by {row['artist']}")

            print("-" * 40)
            last_printed_emotion = emotion

        # Pick ONE song for webcam display
        display_song = "No song"
        if not songs.empty:
            display_song = f"{songs.iloc[0]['song_name']} - {songs.iloc[0]['artist']}"

        # Draw UI
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Song: {display_song}", (x, y+h+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        break  # one face only

    cv2.imshow("Face Emotion Based Music Recommendation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()