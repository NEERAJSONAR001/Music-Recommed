# import torch
# import pandas as pd
# from transformers import pipeline

# songs_df = pd.read_csv("data/songs.csv")


# emotion_classifier = pipeline(
#     "text-classification",
#     model="SamLowe/roberta-base-go_emotions",
#     return_all_scores=True
# )

# def predict_text_emotion(text):
#     emotions = emotion_classifier(text)[0]
#     return max(emotions, key=lambda x: x["score"])["label"]

# def map_emotion(model_emotion):
#     mapping = {
#         "sadness": "sad",
#         "joy": "happy",
#         "anger": "angry",
#         "fear": "sad",
#         "neutral": "neutral"
#     }
#     return mapping.get(model_emotion, "neutral")

# def regulate_mood(emotion):
#     """
#     Converts detected emotion to a healthier target mood
#     """
#     regulation_map = {
#         "sad": "calm",
#         "fear": "calm",
#         "angry": "calm",
#         "neutral": "happy",
#         "happy": "happy"
#     }
#     return regulation_map.get(emotion, "calm")


# def recommend_songs(detected_emotion, n=3):
#     target_mood = regulate_mood(detected_emotion)

#     filtered = songs_df[songs_df["emotion"] == target_mood]

#     if filtered.empty:
#         return None, target_mood

#     return filtered.sample(min(n, len(filtered))), target_mood


# if __name__ == "__main__":
#     print("\n🎵 Music Recommendation Based on Text Emotion 🎵")
#     print("Type a sentence (or type 'exit' to quit)\n")

#     while True:
#         user_text = input("You: ")

#         if user_text.lower() ==i am "exit":
#             print("Exiting...")
#             break


#         model_emotion = predict_text_emotion(user_text)
#         detected_emotion = map_emotion(model_emotion)

   
#         songs, target_mood = recommend_songs(detected_emotion)

#         print(f"\nDetected Emotion : {detected_emotion}")
#         print(f"Target Mood     : {target_mood}")

#         if songs is not None:
#             print("\n🎶 Recommended Songs (Mood-Lifting):")
#             for _, row in songs.iterrows():
#                 print(f"- {row['song_name']} by {row['artist']}")
#         else:
#             print("No songs found for this mood.")

#         print("-" * 45)


#27/02/2025 

# import torch
# import pandas as pd
# from transformers import pipeline
# from ytapi import play_smart_playlist
# import os
# # from yt_player import play_queue_on_youtube
# import webbrowser

# def play_queue_on_youtube(songs):
#     if songs is None or songs.empty:
#         return

#     query = ""

#     for _, row in songs.iterrows():
#         query += f"{row['song_name']} {row['artist']} "

#     youtube_url = f"https://www.youtube.com/results?search_query={query}"
#     print("\n🎵 Opening YouTube Queue...\n")
#     webbrowser.open(youtube_url)



# songs_df = pd.read_csv("data/songs.csv")


# emotion_classifier = pipeline(
#     "text-classification",
#     model="SamLowe/roberta-base-go_emotions",
#     return_all_scores=True
# )


# def predict_text_emotion(text):
#     emotions = emotion_classifier(text)[0]
#     best = max(emotions, key=lambda x: x["score"])
#     return best["label"], best["score"]

# def map_emotion(model_emotion):
#     mapping = {
#         "sadness": "sad",
#         "joy": "happy",
#         "anger": "angry",
#         "fear": "sad",
#         "neutral": "neutral"
#     }
#     return mapping.get(model_emotion, "neutral")

# def regulate_mood(emotion):
#     """
#     Converts detected emotion to a healthier target mood
#     """
#     regulation_map = {
#         "sad": "calm",
#         "fear": "calm",
#         "angry": "calm",
#         "neutral": "happy",
#         "happy": "happy"
#     }
#     return regulation_map.get(emotion, "calm")

# def recommend_songs(detected_emotion, n=3):

#     target_mood = regulate_mood(detected_emotion)

#     # Energy mapping based on mood
#     energy_map = {
#         "calm": "low",
#         "happy": "high"
#     }

#     target_energy = energy_map.get(target_mood, "medium")

#     filtered = songs_df[
#         (songs_df["emotion"] == target_mood) &
#         (songs_df["energy"] == target_energy)
#     ]

#     if filtered.empty:
#         return None, target_mood, target_energy

#     return filtered.sample(min(n, len(filtered))), target_mood, target_energy

# if __name__ == "__main__":

#     print("\n🎵 Emotion-Based Mood Lifting Music System 🎵")
#     print("Type a sentence (or type 'exit' to quit)\n")

#     while True:
#         user_text = input("You: ")

#         if user_text.lower() == "exit":
#             print("Exiting...")
#             break

#         # Step 1: Detect emotion
#         model_emotion, confidence = predict_text_emotion(user_text)

#         # Step 2: Map emotion
#         detected_emotion = map_emotion(model_emotion)

#         # Step 3: Recommend songs
#         songs, target_mood, target_energy = recommend_songs(detected_emotion)

#         print("\nDetected Emotion :", detected_emotion)
#         print("Confidence       :", round(confidence, 3))
#         print("Target Mood      :", target_mood)
#         print("Target Energy    :", target_energy)
        
#         if songs is not None:
#           print("\n🎶 Recommended Songs (Mood Lifting):")
#           for _, row in songs.iterrows():
#            print(f"- {row['song_name']} by {row['artist']} ({row['language']})")
#            play_queue_on_youtube(songs)
# else:
#     print("No songs found.")


#         # if songs is not None:
#         #     print("\n🎶 Recommended Songs (Mood Lifting):")
#         #     for _, row in songs.iterrows():
#         #         print(f"- {row['song_name']} by {row['artist']} ({row['language']})")
#         # else:
#         #     print("No songs found for this mood.")

#     print("-" * 50)

import torch
from transformers import pipeline
from ytapi import play_smart_playlist


emotion_classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    return_all_scores=True
)


def predict_text_emotion(text):
    emotions = emotion_classifier(text)[0]
    best = max(emotions, key=lambda x: x["score"])
    return best["label"], best["score"]


def map_emotion(model_emotion):
    mapping = {
        "sadness": "sad",
        "joy": "happy",
        "anger": "angry",
        "fear": "sad",
        "neutral": "neutral"
    }
    return mapping.get(model_emotion, "neutral")


if __name__ == "__main__":

    print("\n🎵 AI Emotion-Based Music System 🎵")
    print("Type a sentence (or type 'exit' to quit)\n")

    last_emotion = None

    while True:
        user_text = input("You: ")

        if user_text.lower() == "exit":
            print("Exiting...")
            break


        model_emotion, confidence = predict_text_emotion(user_text)

        detected_emotion = map_emotion(model_emotion)

        print("\nDetected Emotion :", detected_emotion)
        print("Confidence       :", round(confidence, 3))

        # Step 3: Play AI Smart Playlist (Only if changed)
        if detected_emotion != last_emotion:
            play_smart_playlist(detected_emotion)
            last_emotion = detected_emotion

        print("-" * 50)