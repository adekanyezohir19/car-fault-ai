import streamlit as st
import numpy as np
import librosa
import joblib
from gtts import gTTS
import os
from tempfile import NamedTemporaryFile
import base64
import matplotlib.pyplot as plt

# =============== PAGE CONFIG ===================
st.set_page_config(page_title="Car Fault AI - Military Edition", layout="wide")

# =============== HEADER ========================
st.markdown("""
# 🚗 **Car Fault AI – Military Edition**
### Developed by *Adekanye Abdulzohir*
#### Professional Sound-Based Vehicle Fault Detection System
""")

st.markdown("---")

# =============== IMAGE / DASHBOARD UI ===========
st.image("military_car.jpg", use_container_width=True, caption="Military Vehicle Diagnostic Dashboard")

# Load your trained model
model_path = "car_fault_model.pkl"
if not os.path.exists(model_path):
    st.error("⚠️ Model file not found! Please upload or retrain first.")
else:
    model = joblib.load(model_path)

# List of components to analyze
components = [
    "Engine", "Brake", "Suspension", "Exhaust",
    "Belt", "Transmission", "Cooling System",
    "Tyre/Rolling", "Battery", "Wiring/Electrical System"
]

# =============== FILE UPLOAD ===================
uploaded_file = st.file_uploader("🎵 Upload Car Sound (WAV, MP3, MP4):", type=["wav", "mp3", "mp4"])
if uploaded_file:
    st.success("✅ File received! Processing sound...")
    
    # Load and analyze audio
    try:
        y, sr = librosa.load(uploaded_file, sr=None)
        st.audio(uploaded_file)

        # Visual waveform
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, color='green')
        plt.title("Sound Waveform")
        st.pyplot(fig)

        # Extract MFCC features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
        mfcc = mfcc.reshape(1, -1)

        # Predict
        prediction = model.predict(mfcc)[0]
        probabilities = model.predict_proba(mfcc)[0]
        confidence = np.max(probabilities) * 100

        # Show Results
        st.markdown(f"### 🧠 **Detected Faulty Component:** `{prediction}`")
        st.progress(int(confidence))
        st.write(f"Confidence: **{confidence:.2f}%**")

        # Voice Feedback (using Google TTS)
        explanation = f"The detected faulty component is {prediction}. Confidence level {int(confidence)} percent."
        tts = gTTS(text=explanation, lang='en')
        with NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            tts.save(tmpfile.name)
            audio_bytes = open(tmpfile.name, "rb").read()
            b64 = base64.b64encode(audio_bytes).decode()
            audio_html = f'<audio controls autoplay src="data:audio/mp3;base64,{b64}"></audio>'
            st.markdown(audio_html, unsafe_allow_html=True)

        # Component explanation
        fault_descriptions = {
            "Engine": "Engine knock, misfire or unusual vibration indicates worn pistons or valves.",
            "Brake": "Squealing or grinding sound suggests worn brake pads or low brake fluid.",
            "Suspension": "Rattling sound when driving over bumps signals shock absorber wear.",
            "Exhaust": "Loud or popping noise may mean exhaust leak or damaged muffler.",
            "Belt": "Squealing noise at startup often means worn or loose serpentine belt.",
            "Transmission": "Whining or gear slipping sound indicates transmission fluid issue.",
            "Cooling System": "Boiling or hissing noise means coolant leak or radiator fan fault.",
            "Tyre/Rolling": "Thumping noise or uneven sound could be tire imbalance or damage.",
            "Battery": "Clicking sound when starting means weak battery or poor connection.",
            "Wiring/Electrical System": "Buzzing or flickering lights show possible short circuit or damaged wire."
        }

        st.info(f"🧾 {fault_descriptions.get(prediction, 'No data available for this component.')}")
        
    except Exception as e:
        st.error(f"❌ Error analyzing sound: {e}")

st.markdown("---")
st.markdown("### 👨🏽‍💻 Developed by **Adekanye Abdulzohir** | Version 3.0 Professional Military Dashboard")
