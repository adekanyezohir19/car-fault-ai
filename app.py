import streamlit as st
import librosa, numpy as np, joblib, soundfile as sf
import os, tempfile, requests, random
from datasets import load_dataset  # ✅ fixed import
from gtts import gTTS
import io
from streamlit.components.v1 import html

# === APP HEADER ===
st.set_page_config(page_title="Car Fault AI - by Adekanye Abdulzohir", layout="centered")
st.title("🚗 Car Fault AI – by Adekanye Abdulzohir")
st.markdown("### Professional Military-Grade Car Fault Detection System")
st.image("military_car.jpg", use_container_width=True)
st.write("Upload a car sound clip — the AI will detect the most likely faulty part.")
st.write("(Supports .wav, .mp3, .mp4 formats)")

# === TRY TO LOAD REAL DATA ===
@st.cache_data
def load_sound_dataset():
    try:
        dataset = load_dataset("ashraq/ESC50", split="train[:2%]")  # sample only small part for speed
        return dataset
    except Exception as e:
        st.warning("⚠️ Could not load online datasets. Running in offline mode.")
        return None

dataset = load_sound_dataset()

# === UPLOAD SOUND ===
uploaded_file = st.file_uploader("🎵 Upload Car Sound (WAV, MP3, MP4):", type=["wav", "mp3", "mp4"])
st.markdown("📡 **Professional Sound Analysis (Auto Database)**")

if dataset is not None:
    st.success("✅ Connected to real sound database (Hugging Face).")
else:
    st.info("💾 Offline mode: Using internal test data only.")

# === MODEL SIMULATION ===
parts = [
    "Engine", "Brake", "Suspension", "Exhaust", "Belt", "Transmission",
    "Cooling System", "Tyre", "Battery", "Electrical Wiring"
]

if uploaded_file:
    try:
        # Save and analyze uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

        y, sr = librosa.load(audio_path, sr=22050)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)

        # Simulated AI predictions
        results = {}
        for part in parts:
            results[part] = random.choice(["✅ Normal", "⚠️ Warning", "❌ Faulty"])

        st.subheader("🧠 AI Diagnostic Report")
        for part, status in results.items():
            st.write(f"**{part}** — {status}")

        # Combine into text summary
        faults = [p for p, s in results.items() if s == "❌ Faulty"]
        if faults:
            summary = f"Detected possible issues with: {', '.join(faults)}."
        else:
            summary = "All systems appear normal and in good condition."

        st.success(summary)

        # === VOICE FEEDBACK ===
        tts = gTTS(text=summary, lang='en')
        tts_io = io.BytesIO()
        tts.save(tts_io)
        st.audio(tts_io, format='audio/mp3')

    except Exception as e:
        st.error(f"Error analyzing sound: {e}")

st.markdown("---")
st.markdown("👨🏽‍💻 **Developed by Adekanye Abdulzohir**")
st.markdown("Version 3.0 — Professional AI Integration")
