import streamlit as st
import numpy as np
import librosa
import joblib
import soundfile as sf
from gtts import gTTS
import os
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# --- App Header ---
st.set_page_config(page_title="AI Car Fault Detector by Adekanye Abdulzohir", page_icon="🚗", layout="centered")
st.title("🚗 AI Car Fault Detection System")
st.caption("Developed by **Adekanye Abdulzohir**, Nigerian Army Engineering Unit 💡")
st.image("military_car.jpg", use_container_width=True)
st.write("Upload your car audio or video (WAV, MP3, MP4) — the AI will detect mechanical faults and explain them clearly by voice and text.")

# --- Load trained model ---
model = joblib.load("car_fault_model.pkl")

# --- Descriptions of components ---
descriptions = {
    "Engine": """The engine converts fuel into power to move your car. 
Common faults include knocking, overheating, and oil leaks.
⚠️ Causes: low oil, bad pistons, overheating.
🧰 Fix: Check oil, coolant, spark plugs.""",

    "Brake": """The brake system stops your car safely.
Grinding or squeaking sounds often mean worn pads or low brake fluid.
⚠️ Causes: thin pads, air in lines, rust.
🧰 Fix: Replace brake pads or top up brake fluid.""",

    "Gearbox": """The gearbox transfers power from the engine to the wheels.
If shifting feels rough or noisy, the gearbox may be worn.
⚠️ Causes: low transmission fluid or damaged gears.
🧰 Fix: Check and replace gearbox oil.""",

    "Clutch": """The clutch connects and disconnects the engine from the gearbox.
A burning smell or slipping gears means clutch issues.
⚠️ Causes: worn clutch plate, hydraulic leak.
🧰 Fix: Replace clutch plate or service the system.""",

    "Exhaust": """The exhaust system removes gases from the engine.
Loud noise or smoke may mean leaks or blockages.
⚠️ Causes: damaged muffler, cracked pipe.
🧰 Fix: Check and seal leaks or replace parts.""",

    "Fan Belt": """The fan belt powers key parts like the alternator and cooling fan.
Squealing means the belt is loose or cracked.
⚠️ Causes: worn belt or pulley.
🧰 Fix: Replace or tighten belt.""",
}

# --- File upload ---
uploaded_file = st.file_uploader("🎵 Upload Car Audio or Video File", type=["wav", "mp3", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    audio_path = "temp_audio.wav"

    # --- Handle Audio or Video ---
    if file_type in ["wav", "mp3"]:
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())
    elif file_type == "mp4":
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        clip = VideoFileClip("temp_video.mp4")
        clip.audio.write_audiofile(audio_path)
        clip.close()

    # --- Read Audio ---
    data, sr = librosa.load(audio_path, sr=None)
    st.audio(audio_path, format="audio/wav")

    # --- Extract features and predict ---
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20).T, axis=0).reshape(1, -1)
    pred = model.predict(mfcc)[0]
    info = descriptions.get(pred, "No detailed information available for this component.")

    # --- Display Results ---
    st.subheader(f"🧠 Detected Fault: {pred}")
    st.write(info)

    # --- Voice Feedback ---
    tts_text = f"The detected fault is {pred}. {info}"
    tts = gTTS(tts_text)
    tts.save("voice.mp3")
    audio_file = open("voice.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")

    # --- Visualize Audio ---
    st.write("🎶 Sound Analysis (MFCC Feature Map)")
    plt.figure(figsize=(8, 4))
    plt.imshow(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20), cmap="plasma", aspect="auto")
    plt.title("MFCC Visualization")
    plt.xlabel("Time Frames")
    plt.ylabel("MFCC Coefficients")
    st.pyplot(plt)

# --- Footer ---
st.markdown("---")
st.markdown("👨‍💻 **AI Car Diagnostic Project by Adekanye Abdulzohir — Nigerian Army Engineering Unit** 🇳🇬")
