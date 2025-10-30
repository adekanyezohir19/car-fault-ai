 # app.py
# AI Smart Vehicle Diagnostic (rule-based) - By Adekanye Abdulzohir
import streamlit as st
import numpy as np
import librosa
import joblib
import soundfile as sf
from gtts import gTTS
import os
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import tempfile
import warnings
import io
import random
warnings.filterwarnings("ignore")

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Smart Vehicle Diagnostic by Adekanye", page_icon="🚗", layout="centered")
st.image("military_car.jpg", use_container_width=True)
st.title("🚗 Smart Vehicle Diagnostic System")
st.caption("Developed by **Adekanye Abdulzohir** — Nigerian Army Engineering Unit")

st.markdown("---")
st.write("Upload a car sound clip or video for full AI-based diagnostic analysis.")

# ✅ Correct uploader
uploaded_file = st.file_uploader(
    "Upload car audio/video (wav, mp3, mp4, m4a, ogg, flac)",
    type=["wav", "mp3", "mp4", "m4a", "ogg", "flac"],
    help="You can upload car sound or short video clip for full AI analysis."
)

# --- Load model if available ---
MODEL_PATH = "car_fault_model.pkl"
trained_model = None
if os.path.exists(MODEL_PATH):
    try:
        trained_model = joblib.load(MODEL_PATH)
    except Exception:
        trained_model = None

# --- helper: feature extraction ---
def extract_audio_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Convert video → audio
    if suffix in [".mp4", ".mov", ".mkv", ".avi"]:
        clip = VideoFileClip(tmp_path)
        audio_tmp = tmp_path + ".wav"
        clip.audio.write_audiofile(audio_tmp, verbose=False, logger=None)
        clip.close()
        return audio_tmp
    return tmp_path

def compute_features(path, sr_target=22050):
    y, sr = librosa.load(path, sr=sr_target, mono=True)
    y, _ = librosa.effects.trim(y)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_bw = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    return {
        "y": y, "sr": sr, "rms": rms, "zcr": zcr,
        "spec_cent": spec_cent, "spec_bw": spec_bw, "tempo": tempo,
        "mfcc_mean": mfcc_mean, "mfcc": mfcc
    }

# --- Heuristic Diagnostic Logic ---
def assess_all(feat):
    scores = {}
    rms, zcr, sc, sbw, mf1, tempo = feat["rms"], feat["zcr"], feat["spec_cent"], feat["spec_bw"], feat["mfcc_mean"][0], feat["tempo"]

    # Engine
    if rms > 0.035 and sc < 1600:
        scores["Engine"] = ("Fault", "High low-frequency energy (possible knocking/misfire).", "Check oil, spark plugs, and compression.")
    elif rms > 0.03:
        scores["Engine"] = ("Warning", "Stronger than usual engine vibration.", "Inspect engine mounts and fuel system.")
    else:
        scores["Engine"] = ("OK", "Engine sound normal.", "Routine checks.")

    # Gearbox
    if sc < 2000 and sbw > 2200 and rms > 0.03:
        scores["Gearbox"] = ("Fault", "Grinding/gear wear likely.", "Check gearbox oil and clutch.")
    elif sbw > 2500:
        scores["Gearbox"] = ("Warning", "Rough shifting signals.", "Check transmission oil.")
    else:
        scores["Gearbox"] = ("OK", "Gearbox normal.", "Routine inspection.")

    # Brake
    if sc > 3000 and zcr > 0.06:
        scores["Brake"] = ("Fault", "Likely brake squeal or grinding.", "Inspect pads and rotors.")
    elif sc > 2500:
        scores["Brake"] = ("Warning", "Brake wear possible.", "Check brake pads and fluid.")
    else:
        scores["Brake"] = ("OK", "Brakes sound normal.", "Routine brake check.")

    # Clutch
    if 0.02 < rms < 0.04 and sc < 1800 and mf1 > 0:
        scores["Clutch"] = ("Warning", "Possible clutch slipping.", "Inspect clutch plate.")
    else:
        scores["Clutch"] = ("OK", "Clutch sound normal.", "Routine check.")

    # Battery
    if rms < 0.006:
        scores["Battery"] = ("Warning", "Low acoustic energy — weak battery.", "Check voltage and charging.")
    else:
        scores["Battery"] = ("OK", "Battery normal.", "Routine test.")

    # Wiring
    if zcr > 0.09:
        scores["Wiring & Connections"] = ("Warning", "Electrical noise detected.", "Inspect wiring harness.")
    else:
        scores["Wiring & Connections"] = ("OK", "No wiring noise.", "Routine inspection.")

    # Suspension
    if sbw > 3000 and rms > 0.03:
        scores["Suspension & Bearings"] = ("Warning", "Possible suspension or wheel issue.", "Inspect shocks and bearings.")
    else:
        scores["Suspension & Bearings"] = ("OK", "Suspension normal.", "Routine inspection.")

    return scores

# --- MAIN SECTION ---
if uploaded_file:
    with st.spinner("🔍 Extracting and analyzing audio..."):
        audio_path = extract_audio_file(uploaded_file)
        feat = compute_features(audio_path)

    st.subheader("🔊 Audio Summary")
    st.write(f"RMS: {feat['rms']:.5f} | ZCR: {feat['zcr']:.5f} | Spectral Centroid: {feat['spec_cent']:.2f} Hz")
    st.audio(audio_path, format="audio/wav")

    # AI analysis
    result = assess_all(feat)
    st.markdown("## 🧠 Component Report")
    for comp, (status, reason, sol) in result.items():
        color = st.success if status == "OK" else st.warning if status == "Warning" else st.error
        color(f"{comp}: {status}")
        st.write(f"**Reason:** {reason}")
        st.write(f"**Solution:** {sol}")
        st.markdown("---")

    # Voice summary
    try:
        voice_text = "Smart Vehicle Diagnostic Report. " + " ".join([f"{c}: {s}, {r}" for c, (s, r, _) in result.items()])
        tts = gTTS(voice_text)
        tts.save("report.mp3")
        st.audio("report.mp3", format="audio/mp3")
    except Exception as e:
        st.error(f"Voice generation failed: {e}")

    # Visualization
    st.subheader("📈 Waveform")
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(feat["y"])
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

else:
    st.info("Upload a car sound (.wav, .mp3, .mp4, etc.) to start full system analysis.")    
