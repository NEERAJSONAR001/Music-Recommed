import pandas as pd

songs_df = pd.read_csv("data/songs.csv")

def recommend_songs(emotion, n=3):
    filtered = songs_df[songs_df["emotion"] == emotion]

    if filtered.empty:
        return None

    return filtered.sample(min(n, len(filtered)))
