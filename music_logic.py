import pandas as pd
import random

songs = pd.read_csv("music/songs.csv")

def recommend_songs(emotion, n=3):
    mood_map = {
        "sad": "happy",
        "angry": "calm",
        "neutral": "happy",
        "surprise": "happy",
        "happy": "happy"
    }

    target_mood = mood_map.get(emotion, "happy")
    filtered = songs[songs["mood"] == target_mood]

    if len(filtered) == 0:
        return []

    return filtered.sample(min(n, len(filtered))).to_dict("records")
