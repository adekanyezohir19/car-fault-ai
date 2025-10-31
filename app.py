import streamlit as st
import numpy as np
import librosa, soundfile as sf
import requests
import os
from gtts import gTTS

st.set_page_config(page_title="🚗 Car Fault AI – by Adekanye Abdulzohir", layout="wide")

# Dashboard banner
st.image("military_car.jpg", use_container_width=True)
st.markdown("""
### 🚗 **Car Fault AI – by Adekanye Abdulzohir**
Professional Car Fault Detection from Sound  
Upload your car sound — the AI analyzes **engine, gear, brake, clutch, battery, and wiring** faults.  
*(Supports .wav, .mp3, .mp4 formats)*
""")

uploaded_file = st.file_uploader("🎵 Upload Car Sound (WAV, MP3, MP4)", type=["wav", "mp3", "mp4"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    st.info("Analyzing sound... please wait")

    # Simulated sound feature extraction
    try:
        y, sr = librosa.load(uploaded_file, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y))
        zero_cross = np.mean(librosa.zero_crossings(y))

        # Basic heuristic analysis
        if rms < 0.02:
            result = "⚠️ Weak engine or ignition fault detected."
        elif zero_cross > 0.15:
            result = "⚙️ Possible clutch or brake imbalance."
        elif duration < 1.0:
            result = "🔋 Battery or wiring issue — sound too short for stable ignition."
        else:
            result = "✅ No critical fault detected. Vehicle sound is normal."

        st.success(result)

        # Voice feedback
        tts = gTTS(result)
        tts.save("result.mp3")
        audio_file = open("result.mp3", "rb")
        st.audio(audio_file.read(), format="audio/mp3")

    except Exception as e:
        st.error("❌ Unable to process this sound. Please use a clear car sound file.")
        st.caption(str(e))

else:
    st.warning("Upload a sound file to start analysis.")

st.markdown("""
---
👨🏽‍💻 **Developed by Adekanye Abdulzohir**  
Version 2.5 — Professional Voice-Integrated Car Fault Analysis
""")
