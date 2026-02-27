# import cv2
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model
# from collections import deque
# # from yt_player import play_queue_on_youtube
# import webbrowser

# youtube_opened = False

# def play_queue_on_youtube(songs):
#     """
#     Opens recommended songs as a YouTube search queue.
#     """

#     if songs is None or songs.empty:
#         print("No songs to play.")
#         return

#     query = ""

#     for _, row in songs.iterrows():
#         query += f"{row['song_name']} {row['artist']} "

#     youtube_url = f"https://www.youtube.com/results?search_query={query}"

#     print("\n🎵 Opening YouTube Queue...\n")
#     webbrowser.open(youtube_url)



# # model he 
# model = load_model("face_emotions/face_emotion_mobilenetv2_70.h5")

# # iska order same hona chahiye jese train me tha
# emotion_labels = [
#     "surprise",
#     "happy",
#     "sad",
#     "angry",
#     "neutral"
# ]

# songs_df = pd.read_csv("data/songs.csv")

# def regulate_mood(emotion):
#     regulation_map = {
#         "sad": "calm",
#         "angry": "calm",
#         "neutral": "happy",
#         "surprise": "happy",
#         "happy": "happy"
#     }
#     return regulation_map.get(emotion, "calm")

# def get_all_songs(emotion):
#     target_mood = regulate_mood(emotion)
#     filtered = songs_df[songs_df["emotion"] == target_mood]
#     return filtered, target_mood


# emotion_queue = deque(maxlen=10)

# face_cascade = cv2.CascadeClassifier(
#     "face_emotions/haarcascade_frontalface_default.xml"
# )

# cap = cv2.VideoCapture(0)
# print("\n🎥 Webcam started. Press 'q' to quit.\n")

# last_printed_emotion = None

# # last_emotion = None


# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         face = frame[y:y+h, x:x+w]             
#         face = cv2.resize(face, (96, 96))     
#         face = face / 255.0
#         face = np.expand_dims(face, axis=0)    

#         prediction = model.predict(face, verbose=0)
#         emotion_queue.append(prediction[0])

#         avg_prediction = np.mean(emotion_queue, axis=0)
#         emotion = emotion_labels[np.argmax(avg_prediction)]

      
#         songs, target_mood = get_all_songs(emotion)
        

#         if emotion != last_printed_emotion:
#             print(f"\nDetected Emotion: {emotion}")
#             print(f"Target Mood    : {target_mood}")
#             print("Recommended Songs:")

#     if songs.empty:
#         print("  No songs found.")
#     else:
#         for _, row in songs.iterrows():
#             print(f"  - {row['song_name']} by {row['artist']}")

#         # 🔥 OPEN YOUTUBE QUEUE HERE
#         play_queue_on_youtube(songs)

#         print("-" * 40)
#         last_printed_emotion = emotion


           

#         display_song = "No song"
#         if not songs.empty:
#             display_song = f"{songs.iloc[0]['song_name']} - {songs.iloc[0]['artist']}"

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, f"Emotion: {emotion}", (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
#         cv2.putText(frame, f"Song: {display_song}", (x, y+h+30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

#         break  

#     cv2.imshow("Face Emotion Based Music Recommendation", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

#27/feb/2026


# import cv2
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model
# from collections import deque
# import webbrowser
# import os
# from ytapi import play_smart_playlist

# def play_queue_on_youtube(songs):
#     if songs is None or songs.empty:
#         print("No songs to play.")
#         return

#     query = ""
#     for _, row in songs.iterrows():
#         query += f"{row['song_name']} {row['artist']} "

#     youtube_url = f"https://www.youtube.com/results?search_query={query}"

#     print("\n🎵 Opening YouTube Queue...\n")
#     webbrowser.open(youtube_url)

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(BASE_DIR, "face_emotion_mobilenetv2_70.h5")

# print("Loading model from:", model_path)

# model = load_model(model_path, compile=False)

# emotion_labels = [
#     "surprise",
#     "happy",
#     "sad",
#     "angry",
#     "neutral"
# ]

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(BASE_DIR)

# csv_path = os.path.join(PROJECT_ROOT, "data", "songs.csv")

# print("Loading songs from:", csv_path)

# songs_df = pd.read_csv(csv_path)


# def regulate_mood(emotion):
#     regulation_map = {
#         "sad": "calm",
#         "angry": "calm",
#         "neutral": "happy",
#         "surprise": "happy",
#         "happy": "happy"
#     }
#     return regulation_map.get(emotion, "calm")

# def get_all_songs(emotion):
#     target_mood = regulate_mood(emotion)
#     filtered = songs_df[songs_df["emotion"] == target_mood]
#     return filtered, target_mood

# emotion_queue = deque(maxlen=10)
# face_cascade = cv2.CascadeClassifier(
#     "face_emotions/haarcascade_frontalface_default.xml"
# )

# cap = cv2.VideoCapture(0)
# print("\n🎥 Webcam started. Press 'q' to quit.\n")

# last_emotion = None
# youtube_opened = False   

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         face = frame[y:y+h, x:x+w]
#         face = cv2.resize(face, (96, 96))
#         face = face / 255.0
#         face = np.expand_dims(face, axis=0)

#         prediction = model.predict(face, verbose=0)
#         emotion_queue.append(prediction[0])

#         avg_prediction = np.mean(emotion_queue, axis=0)
#         emotion = emotion_labels[np.argmax(avg_prediction)]

#         songs, target_mood = get_all_songs(emotion)

#         # PRINT & OPEN YOUTUBE ONLY IF EMOTION CHANGED
#         if emotion != last_emotion:

#             print(f"\nDetected Emotion: {emotion}")
#             print(f"Target Mood    : {target_mood}")
#             print("Recommended Songs:")

#             if songs.empty:
#                 print("  No songs found.")
#             else:
#                 for _, row in songs.iterrows():
#                     print(f"  - {row['song_name']} by {row['artist']}")

#                 # OPEN ONLY ON FIRST DETECTION
#                 if not youtube_opened:
#                     play_queue_on_youtube(songs)
#                     youtube_opened = True

#             print("-" * 40)

#             last_emotion = emotion

#         # DISPLAY ON FRAME
#         display_song = "No song"
#         if not songs.empty:
#             display_song = f"{songs.iloc[0]['song_name']} - {songs.iloc[0]['artist']}"

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, f"Emotion: {emotion}", (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
#         cv2.putText(frame, f"Song: {display_song}", (x, y+h+30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

#         break

#     cv2.imshow("Face Emotion Based Music Recommendation", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import os
import requests
import webbrowser
import random

YOUTUBE_API_KEY = "AIzaSyB1bEvIoFJnWPUlaDE3zVh7NmnMtzMGbls"

EMOTION_QUERIES = {
    "sad": [
        "motivational hindi songs",
        "uplifting english songs",
        "bollywood comeback songs",
        "english inspirational pop songs"
    ],
    "angry": [
        "calm instrumental hindi music",
        "lofi english chill music",
        "peaceful piano music",
        "relaxing guitar instrumental"
    ],
    "happy": [
        "latest bollywood party songs",
        "english pop dance hits",
        "trending hindi upbeat songs",
        "english feel good playlist"
    ],
    "neutral": [
        "trending hindi songs 2026",
        "english top hits playlist",
        "bollywood romantic hits",
        "english acoustic playlist"
    ],
    "surprise": [
        "viral hindi songs",
        "english top trending music",
        "bollywood blockbuster songs",
        "global pop hits"
    ]
}

def fetch_songs(emotion, max_results=8):

    query_list = EMOTION_QUERIES.get(emotion, EMOTION_QUERIES["neutral"])
    query = random.choice(query_list)

    url = "https://www.googleapis.com/youtube/v3/search"

    params = {
        "part": "snippet",
        "q": query,
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results,
        "type": "video",
        "videoCategoryId": "10",  # Music category
        "regionCode": "IN"        # Focus India
    }

    response = requests.get(url, params=params)
    data = response.json()

    videos = []

    for item in data.get("items", []):
        title = item["snippet"]["title"]
        video_id = item["id"]["videoId"]

        # Basic Filtering (avoid non-music)
        if any(x in title.lower() for x in ["trailer", "interview", "reaction"]):
            continue

        videos.append({
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}"
        })

    return videos


def play_smart_playlist(emotion):

    songs = fetch_songs(emotion)

    if not songs:
        print("No songs found.")
        return

    print("\n🎵 AI Recommended Songs:\n")

    for song in songs:
        print("•", song["title"])

    # Play first song
    webbrowser.open(songs[0]["url"])
  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "face_emotion_mobilenetv2_70.h5")

print("Loading model from:", model_path)

model = load_model(model_path, compile=False)


emotion_labels = [
    "surprise",
    "happy",
    "sad",
    "angry",
    "neutral"
]


cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

emotion_queue = deque(maxlen=10)

cap = cv2.VideoCapture(0)
print("\n🎥 Webcam started. Press 'q' to quit.\n")

last_emotion = None
youtube_opened = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (96, 96))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)
        emotion_queue.append(prediction[0])

        avg_prediction = np.mean(emotion_queue, axis=0)
        emotion = emotion_labels[np.argmax(avg_prediction)]

        if emotion != last_emotion:

            print(f"\nDetected Emotion: {emotion}")

            if not youtube_opened:
                play_smart_playlist(emotion)
                youtube_opened = True

            last_emotion = emotion

       
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        break

    cv2.imshow("AI Emotion Music Recommender", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
