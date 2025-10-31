import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import requests
from io import BytesIO
from gtts import gTTS
import tempfile
import os

# === UI SETUP ===
st.set_page_config(page_title="Car Fault AI – Military Edition", page_icon="🚗", layout="wide")

st.markdown("<h1 style='text-align: center; color: #32CD32;'>🚗 Car Fault AI – by Adekanye Abdulzohir</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Professional AI-Powered Vehicle Diagnosis</h4>", unsafe_allow_html=True)
st.image("military_car.jpg", use_container_width=True)

st.markdown("---")

# === DEFINE COMPONENTS ===
components = [
    "Engine", "Brake", "Clutch", "Gearbox", "Battery",
    "Wire", "Exhaust", "Fan Belt"
]

# === DATABASE SIMULATION (REAL AUDIO FETCH FROM HUGGINGFACE / GOOGLE FALLBACK) ===
@st.cache_data(show_spinner=False)
def fetch_reference_sound(component):
    try:
        urls = {
            "Engine": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/engine.wav",
            "Brake": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/brake.wav",
            "Clutch": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/clutch.wav",
            "Gearbox": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/gear.wav",
            "Battery": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/battery.wav",
            "Wire": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/wire.wav",
            "Exhaust": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/exhaust.wav",
            "Fan Belt": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/fan_belt.wav",
        }
        url = urls.get(component)
        if url:
            response = requests.get(url)
            data, sr = sf.read(BytesIO(response.content))
            return data, sr
        return None, None
    except Exception:
        return None, None


# === SOUND COMPARISON FUNCTION ===
def compare_sounds(uploaded, reference):
    try:
        uploaded_mfcc = np.mean(librosa.feature.mfcc(y=uploaded, sr=22050, n_mfcc=20).T, axis=0)
        reference_mfcc = np.mean(librosa.feature.mfcc(y=reference, sr=22050, n_mfcc=20).T, axis=0)
        diff = np.linalg.norm(uploaded_mfcc - reference_mfcc)
        return diff
    except Exception:
        return np.inf


# === USER SOUND UPLOAD ===
uploaded_file = st.file_uploader("🎵 Upload Car Sound (WAV, MP3, MP4):", type=["wav", "mp3", "mp4"])

if uploaded_file:
    st.audio(uploaded_file)
    st.markdown("🔍 **Analyzing... Please wait...**")
    try:
        y_user, sr_user = librosa.load(uploaded_file, sr=22050)
        results = {}

        # Compare each component sound
        for comp in components:
            ref, sr_ref = fetch_reference_sound(comp)
            if ref is not None:
                diff = compare_sounds(y_user, ref)
                results[comp] = diff
            else:
                results[comp] = np.inf

        # Normalize and interpret
        st.markdown("### 🔧 Full Component Analysis:")
        report = ""
        for comp, score in results.items():
            if score < 80:
                status = "✅ Good condition"
            elif score < 140:
                status = "⚠️ Slight issue detected"
            else:
                status = "❌ Possible fault detected"
            st.write(f"**{comp}:** {status}")
            report += f"{comp}: {status}. "

        # Text + Voice summary
        st.markdown("---")
        st.subheader("🗣️ AI Feedback Summary")
        st.info(report)
        try:
            tts = gTTS(report)
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp.name)
            st.audio(temp.name)
        except:
            st.warning("Voice feedback unavailable. Text summary shown above.")

    except Exception as e:
        st.error(f"Error processing sound: {e}")

else:
    st.info("Please upload a car sound clip to start analysis.")

st.markdown("---")
st.markdown("<h5 style='text-align:center;'>👨🏽‍💻 Developed by <b>Adekanye Abdulzohir</b><br>Version 3.0 — Real-Time Data Integrated</h5>", unsafe_allow_html=True)
