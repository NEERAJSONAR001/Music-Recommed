# import streamlit as st
# from face_emotions.face_model import detect_face_emotion
# from text_emotions.text_model import detect_text_emotion
# from music_logic import recommend_songs


# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False

# if "users" not in st.session_state:
#     st.session_state.users = {}

# st.set_page_config(page_title="Emotion Music App")

# if not st.session_state.logged_in:

#     tab1, tab2 = st.tabs(["Login", "Register"])

#     with tab1:
#         user = st.text_input("Username")
#         pwd = st.text_input("Password", type="password")

#         if st.button("Login"):
#             if user in st.session_state.users and st.session_state.users[user] == pwd:
#                 st.session_state.logged_in = True
#                 st.rerun()
#             else:
#                 st.error("Invalid credentials")

#     with tab2:
#         new_user = st.text_input("New Username")
#         new_pwd = st.text_input("New Password", type="password")

#         if st.button("Register"):
#             st.session_state.users[new_user] = new_pwd
#             st.success("Registered! Please login.")

#     st.stop()

# st.title("🎵 Emotion-Based Music Recommendation App")

# mode = st.radio("Choose emotion input:", ["Facial Emotion", "Text Emotion"])

# emotion = None

# if mode == "Facial Emotion":
#     if st.button("Detect Emotion from Face"):
#         emotion, frame = detect_face_emotion()
#         st.success(f"Detected Emotion: **{emotion}**")

# if mode == "Text Emotion":
#     text = st.text_area("How are you feeling?")
#     if st.button("Analyze Text"):
#         emotion = detect_text_emotion(text)
#         st.success(f"Detected Emotion: **{emotion}**")

# if emotion:
#     st.subheader("🎧 Recommended Songs")

#     songs = recommend_songs(emotion)

#     if songs is not None:
#         for _, row in songs.iterrows():
#             st.markdown(f"""
#             **🎵 {row['song_name']}**  
#             Artist: {row['artist']}  
#             Language: {row['language']}
#             """)
#     else:
#         st.info("No songs found for this emotion.")
import streamlit as st
import cv2

from face_emotions.face_model import detect_face_emotion
from text_emotions.text_model import detect_text_emotion
from music_logic import recommend_songs
from text_emotions.text_model import detect_text_emotion


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "users" not in st.session_state:
    st.session_state.users = {}


# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Emotion Music App",
    layout="centered"
)

# -----------------------------
# LOGIN / REGISTER
# -----------------------------
if not st.session_state.logged_in:

    st.title("🎵 Emotion-Based Music App")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Register"):
            st.session_state.users[new_user] = new_pass
            st.success("Registered successfully! Please login.")

    st.stop()

# -----------------------------
# MAIN APP
# -----------------------------
st.title("🎧 Emotion-Based Music Recommendation")

mode = st.radio(
    "Choose Emotion Input Method:",
    ["Facial Emotion", "Text Emotion"]
)

emotion = None

# -----------------------------
# FACE EMOTION
# -----------------------------
if mode == "Facial Emotion":
    st.subheader("📷 Facial Emotion Detection")

    if st.button("Detect Emotion from Face"):
        emotion, music, frame = detect_face_emotion()

        if frame is not None:
            st.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB"
            )
            st.success(f"Detected Emotion: **{emotion}**")
            st.info(f"🎵 Suggested Music Type: **{music}**")
        else:
            st.error("Could not access webcam")

# -----------------------------
# TEXT EMOTION
# -----------------------------
if mode == "Text Emotion":
    st.subheader("✍️ Text Emotion Detection")

    text = st.text_area("Type how you are feeling:")

    if st.button("Analyze Text"):
        emotion = detect_text_emotion(text)
        st.success(f"Detected Emotion: **{emotion}**")

# -----------------------------
# SONG RECOMMENDATION (CSV)
# -----------------------------
if emotion:
    st.subheader("🎶 Recommended Songs")

    songs = recommend_songs(emotion)

    if songs is not None:
        for _, row in songs.iterrows():
            st.markdown(f"""
            **🎵 {row['song_name']}**  
            Artist: {row['artist']}  
            Language: {row['language']}
            """)
    else:
        st.info("No songs found for this emotion.")
