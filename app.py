import streamlit as st
import librosa
import numpy as np
import joblib
import soundfile as sf

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="AI Car Fault Detection", page_icon="🚗", layout="centered")

# -------------------------------
# HEADER WITH IMAGE
# -------------------------------
st.image("military_car.jpg", use_container_width=True)  # military car image
st.title("🚘 AI Car Fault Detection System")
st.write("Developed by **Adekanye Abdulzohir** 🇳🇬")
st.markdown("""
This AI system uses **sound analysis** to detect mechanical faults in vehicles.  
Upload a short car sound clip, and the model will predict the likely faulty part.  
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
        y, sr = sf.read(uploaded_file)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
        mfcc = mfcc.reshape(1, -1)
        
        prediction = model.predict(mfcc)[0]
        st.success(f"✅ Detected possible fault in: **{prediction}**")
    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Made with ❤️ by Adekanye Abdulzohir — Nigeria Army Engineering Unit AI Project")
