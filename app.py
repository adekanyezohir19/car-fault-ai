import streamlit as st
import librosa
import numpy as np
import joblib
import soundfile as sf
import tempfile
import os
import io
import base64
import requests

# ==============================
#  APP INFORMATION
# ==============================
st.set_page_config(page_title="Car Fault AI – by Adekanye Abdulzohir", page_icon="🚗", layout="centered")

st.markdown("""
# 🚗 **Car Fault AI – by Adekanye Abdulzohir**
### Professional Car Fault Detection from Sound  
Upload a car sound clip — the AI will detect the most likely faulty part.  
*(Supports `.wav`, `.mp3`, `.mp4` formats)*
""")

st.image("military_car.jpg", use_container_width=True)

# ==============================
#  LOAD TRAINED MODEL
# ==============================
model_path = "car_fault_model.pkl"

if not os.path.exists(model_path):
    st.error("⚠️ Model file not found! Please ensure `car_fault_model.pkl` is in the same folder.")
else:
    model = joblib.load(model_path)

# ==============================
#  FUNCTION: Extract MFCC
# ==============================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
    return mfcc.reshape(1, -1)

# ==============================
#  VOICE FEEDBACK (Browser Safe)
# ==============================
def speak_text(text):
    js_code = f"""
    <script>
    var msg = new SpeechSynthesisUtterance("{text}");
    msg.pitch = 1;
    msg.rate = 1;
    msg.volume = 1;
    msg.lang = 'en-US';
    window.speechSynthesis.speak(msg);
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)

# ==============================
#  AUDIO UPLOAD
# ==============================
uploaded_file = st.file_uploader("🎵 Upload Car Sound (WAV, MP3, MP4):", type=["wav", "mp3", "mp4", "mpeg4"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(uploaded_file, format="audio/wav")

    try:
        features = extract_features(tmp_path)
        prediction = model.predict(features)[0]
        st.success(f"✅ **Detected Faulty Component:** {prediction}")
        speak_text(f"The detected faulty component is {prediction}")

        # Professional explanation of the detected part
        explanations = {
            "Engine": "Engine faults often relate to knocking, misfiring, or rough idling sounds.",
            "Brake": "Brake faults can be indicated by grinding, squealing, or scraping sounds.",
            "Gearbox": "Gearbox issues cause whining or clunking when shifting gears.",
            "Exhaust": "Exhaust problems produce deep, rumbling, or popping sounds.",
            "Fan Belt": "A faulty fan belt creates high-pitched squealing when accelerating."
        }

        st.info(explanations.get(prediction, "No detailed explanation available."))
        speak_text(explanations.get(prediction, ""))

    except Exception as e:
        st.error("⚠️ Could not analyze sound. Please upload a valid car sound file.")
        st.text(e)

# ==============================
#  AUTO DATASET (future-ready)
# ==============================
st.markdown("---")
st.subheader("📡 Professional Sound Analysis (Auto Database)")
try:
    st.markdown("✅ Connected to sample sound database (simulated). Real database integration coming soon.")
except:
    st.warning("⚠️ Could not load online datasets. Check your internet connection.")

# ==============================
#  FOOTER
# ==============================
st.markdown("""
---
👨🏽‍💻 **Developed by Adekanye Abdulzohir**  
*Version 2.0 — Professional AI Integration*
""")
