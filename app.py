import streamlit as st
import librosa
import numpy as np
import joblib
import soundfile as sf
import matplotlib.pyplot as plt

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="AI Car Fault Detection", page_icon="🚗", layout="centered")

# -------------------------------
# HEADER WITH IMAGE
# -------------------------------
st.image("military_car.jpg", use_container_width=True)
st.title("🚘 AI Car Fault Detection & Maintenance Assistant")
st.write("Developed by **Adekanye Abdulzohir** 🇳🇬")
st.markdown("""
This AI system uses **sound analysis** to detect vehicle faults and suggest **maintenance actions**.  
Upload a short car sound clip, and the AI will analyze, visualize, and recommend solutions.  
""")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("car_fault_model.pkl")

model = load_model()

# -------------------------------
# UPLOAD AUDIO SECTION
# -------------------------------
uploaded_file = st.file_uploader("🎧 Upload a car sound (.wav format)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    
    try:
        # Load and process audio
        y, sr = sf.read(uploaded_file)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
        mfcc = mfcc.reshape(1, -1)
        
        # Predict and get confidence
        pred_proba = model.predict_proba(mfcc)
        pred_label = model.classes_[np.argmax(pred_proba)]
        confidence = np.max(pred_proba) * 100

        # Display results
        st.success(f"✅ Detected possible fault in: **{pred_label}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")
        from gtts import gTTS
import os
import streamlit as st

# Suppose your prediction result is stored in a variable called `fault_label`
# For example:
# fault_label = model.predict([features])[0]

# Convert the diagnosis to speech
tts = gTTS(f"The detected fault is {fault_label}")
tts.save("diagnosis.mp3")

# Play the audio in Streamlit
audio_file = open("diagnosis.mp3", "rb")
audio_bytes = audio_file.read()
st.audio(audio_bytes, format="audio/mp3")

        # -------------------------------
        # SYSTEM STATUS DASHBOARD
        # -------------------------------
        st.markdown("### 📈 System Status Dashboard")

        # Waveform
        st.subheader("Sound Waveform")
        fig, ax = plt.subplots()
        ax.plot(y)
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        # MFCC visualization
        st.subheader("MFCC Frequency Heatmap")
        mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        fig2, ax2 = plt.subplots()
        img = librosa.display.specshow(mfcc_data, x_axis='time', ax=ax2)
        fig2.colorbar(img, ax=ax2)
        st.pyplot(fig2)

        # -------------------------------
        # MAINTENANCE ASSISTANT PANEL
        # -------------------------------
        st.markdown("### 🧠 Maintenance Assistant AI")

        maintenance_tips = {
            "engine": [
                "Check oil and coolant levels immediately.",
                "Inspect spark plugs for carbon buildup.",
                "Avoid long idling sessions; it causes overheating."
            ],
            "gearbox": [
                "Inspect transmission fluid for color and level.",
                "Avoid aggressive shifting; it wears out gear teeth.",
                "Schedule a mechanical inspection if vibration persists."
            ],
            "brakes": [
                "Check brake pads and fluid levels.",
                "Listen for grinding or squealing sounds.",
                "Do not drive if the brake pedal feels soft or spongy."
            ],
            "exhaust": [
                "Inspect exhaust pipe for leaks or rust.",
                "Clean catalytic converter if blocked.",
                "Avoid engine revving when stationary."
            ]
        }

        # Match fault label to tips
        tips = maintenance_tips.get(pred_label.lower(), [
            "Perform general inspection and diagnostics.",
            "Ensure regular oil changes and filter replacements."
        ])

        st.write("🛠️ Recommended Actions:")
        for tip in tips:
            st.markdown(f"- {tip}")

        # Health indicator
        st.markdown("### ⚙️ System Health Status")
        if confidence > 85:
            st.success("🟢 Status: Operational — Minimal risk detected.")
        elif confidence > 60:
            st.warning("🟡 Status: Warning — Possible moderate issue.")
        else:
            st.error("🔴 Status: Critical — Immediate inspection advised!")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Made with ❤️ by Adekanye Abdulzohir — Nigeria Army Engineering Unit AI Project")
