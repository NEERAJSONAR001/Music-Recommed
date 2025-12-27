# Isko tabhi run karna jab dataset me update ho 

import os

base_dir = "face_emotions/fer13"

for split in ["train", "test"]:
    print(f"\n{split.upper()} DATA")
    path = os.path.join(base_dir, split)

    for emotion in os.listdir(path):
        emotion_path = os.path.join(path, emotion)
        print(emotion, "->", len(os.listdir(emotion_path)), "images")
