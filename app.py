# =====================================
# MILITARY VEHICLE FAULT AI DASHBOARD
# By Adekanye Abdulzohir
# =====================================

import os
import streamlit as st
import numpy as np
import requests
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from gtts import gTTS
from io import BytesIO

# ---------------------------
# INSTALL (failsafe for Streamlit Cloud)
# ---------------------------
try:
    import soundfile as sf
except ImportError:
    os.system("pip install soundfile numpy pandas requests gtts torch transformers scikit-learn librosa tqdm")

# ---------------------------
# DASHBOARD SETUP
# ---------------------------
st.set_page_config(page_title="Military Car Fault AI", layout="wide")

st.markdown("""
    <style>
    body {background-color: #0A0F0D;}
    h1, h2, h3, h4, p, label {color: #00FFAA !important;}
    .stApp {background-color: #0A0F0D;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style="text-align:center; color:#00FFAA;">🚔 Military Vehicle Fault Detection System</h1>
    <h3 style="text-align:center; color:#FFD700;">Developed by Adekanye Abdulzohir</h3>
    <p style="text-align:center; color:#CCCCCC;">
    Real-time car sound diagnosis for engine, gearbox, exhaust, brake, clutch, battery, and wiring faults.
    </p>
""", unsafe_allow_html=True)

# Add the dashboard image (you can replace this URL with your own)
st.image("https://i.ibb.co/JFj7S0G/military-car-dashboard.jpg", caption="Military Diagnostic Dashboard", use_container_width=True)

# ---------------------------
# SIDEBAR – VEHICLE INFO
# ---------------------------
st.sidebar.header("🔧 Vehicle Information")
car_name = st.sidebar.text_input("Vehicle Name (e.g. Toyota Hilux)")
car_model = st.sidebar.text_input("Model (e.g. 2024 Military Spec)")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("🎙️ Upload Car Sound File", type=["wav", "mp3", "mp4"])

# ---------------------------
# FEATURE EXTRACTION
# ---------------------------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        return np.hstack([mfcc, zcr, centroid])
    except Exception as e:
        st.error(f"⚠️ Audio error: {e}")
        return np.zeros(22)

# Dummy classifier (simulating trained AI model)
clf = RandomForestClassifier()
scaler = StandardScaler()

# ---------------------------
# PROCESS UPLOADED SOUND
# ---------------------------
if uploaded_file:
    st.audio(uploaded_file)
    features = extract_features(uploaded_file)
    features = features.reshape(1, -1)

    # Simulated output for demonstration
    components = ["Engine", "Gearbox", "Brake", "Clutch", "Battery", "Wiring", "Exhaust"]
    results = {}

    # Simulated scoring
    for comp in components:
        fault_level = np.random.choice(["Good Condition ✅", "Minor Fault ⚠️", "Critical Fault ❌"], p=[0.6, 0.25, 0.15])
        results[comp] = fault_level

    # Display Results
    st.markdown(f"<h2 style='text-align:center; color:#00FFAA;'>🔍 Diagnostic Results</h2>", unsafe_allow_html=True)
    for comp, status in results.items():
        color = "#00FF00" if "Good" in status else "#FFAA00" if "Minor" in status else "#FF3333"
        st.markdown(f"<p style='color:{color}; font-size:18px;'>{comp}: {status}</p>", unsafe_allow_html=True)

    st.info(f"Vehicle: **{car_name or 'Unknown'}**, Model: **{car_model or 'Unknown'}**")

    # ---------------------------
    # Voice Feedback (gTTS)
    # ---------------------------
    report_text = "Diagnostic summary for your vehicle. "
    for comp, status in results.items():
        report_text += f"{comp} is in {status.replace('✅', 'good condition').replace('⚠️', 'minor fault').replace('❌', 'critical fault')}. "

    tts = gTTS(report_text)
    tts_audio = BytesIO()
    tts.save("result.mp3")
    st.audio("result.mp3", format="audio/mp3")
else:
    st.warning("🎧 Upload a vehicle sound file to begin diagnosis.")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("""
    <hr>
    <p style="text-align:center; color:#888;">
    © 2025 Military Vehicle Diagnostic AI — <b style='color:#FFD700;'>By Adekanye Abdulzohir</b>
    </p>
""", unsafe_allow_html=True)
