import torch
import pandas as pd
from transformers import pipeline

songs_df = pd.read_csv("data/songs.csv")


emotion_classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    return_all_scores=True
)

def predict_text_emotion(text):
    emotions = emotion_classifier(text)[0]
    return max(emotions, key=lambda x: x["score"])["label"]

def map_emotion(model_emotion):
    mapping = {
        "sadness": "sad",
        "joy": "happy",
        "anger": "angry",
        "fear": "sad",
        "neutral": "neutral"
    }
    return mapping.get(model_emotion, "neutral")

def regulate_mood(emotion):
    """
    Converts detected emotion to a healthier target mood
    """
    regulation_map = {
        "sad": "calm",
        "fear": "calm",
        "angry": "calm",
        "neutral": "happy",
        "happy": "happy"
    }
    return regulation_map.get(emotion, "calm")


def recommend_songs(detected_emotion, n=3):
    target_mood = regulate_mood(detected_emotion)

    filtered = songs_df[songs_df["emotion"] == target_mood]

    if filtered.empty:
        return None, target_mood

    return filtered.sample(min(n, len(filtered))), target_mood


if __name__ == "__main__":
    print("\n🎵 Music Recommendation Based on Text Emotion 🎵")
    print("Type a sentence (or type 'exit' to quit)\n")

    while True:
        user_text = input("You: ")

        if user_text.lower() == "exit":
            print("Exiting...")
            break


        model_emotion = predict_text_emotion(user_text)
        detected_emotion = map_emotion(model_emotion)

   
        songs, target_mood = recommend_songs(detected_emotion)

        print(f"\nDetected Emotion : {detected_emotion}")
        print(f"Target Mood     : {target_mood}")

        if songs is not None:
            print("\n🎶 Recommended Songs (Mood-Lifting):")
            for _, row in songs.iterrows():
                print(f"- {row['song_name']} by {row['artist']}")
        else:
            print("No songs found for this mood.")

        print("-" * 45)
