import torch
import pandas as pd
from transformers import pipeline

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
        "neutral": "calm"
    }
    return mapping.get(model_emotion, "neutral")


if __name__ == "__main__":
    print("Type a sentence (or type 'exit' to quit)\n")

    while True:
        user_text = input("You: ")

        if user_text.lower() == "exit":
            print("Exiting...")
            break

        predicted = predict_text_emotion(user_text)
        predicted = predict_text_emotion(user_text)
        final_mood = map_emotion(predicted)
        print("Predicted Emotion:", final_mood)
        print("-" * 30)
