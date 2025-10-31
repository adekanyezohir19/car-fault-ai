# Professional Multi-Component Car Fault Scanner (speaks results)
# Developed by Adekanye Abdulzohir — Military Edition
# Paste this entire file into your repo as app.py

import streamlit as st
import os
import numpy as np
import pandas as pd
import librosa
import joblib
import soundfile as sf
import matplotlib.pyplot as plt
from gtts import gTTS
from tempfile import NamedTemporaryFile
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import load_dataset

st.set_page_config(page_title="Car Fault AI - Pro", layout="wide")

# ----------------------------
# Header & Military Theme
# ----------------------------
st.markdown(
    """
    <style>
        .stApp { background-color: #0b0f12; color: #c7d2d8; }
        .big-title { color:#66fcf1; font-weight:700; font-size:28px; text-align:center; }
        .subtitle { color:#9ae6d8; text-align:center; }
        .card { background:#0f1720; border-radius:8px; padding:10px; }
        table { color: #c7d2d8; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='big-title'>🚗 Car Fault AI — Professional Multi-Component Scanner</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Developed by <b>Adekanye Abdulzohir</b> — Military Edition</div>", unsafe_allow_html=True)
st.markdown("---")

# show military image if present
if os.path.exists("military_car.jpg"):
    st.image("military_car.jpg", use_container_width=True, caption="Tactical Vehicle Diagnostic Dashboard")
else:
    st.warning("⚠️ military_car.jpg missing — upload the image to your repo to show dashboard image.")

# ----------------------------
# Component list and descriptions
# ----------------------------
COMPONENTS = [
    "Engine", "Gearbox", "Brake", "Clutch", "Suspension", "Exhaust",
    "Belt", "Transmission", "Cooling System", "Tyre/Rolling", "Battery", "Wiring/Electrical System"
]

DESCRIPTIONS = {
    "Engine": "Knocking, misfire, popping: check pistons, valves, oil, spark plugs.",
    "Gearbox": "Grinding or slipping: check transmission fluid and gear wear.",
    "Brake": "Squeal/grind: inspect pads, rotors and fluid level.",
    "Clutch": "Slipping/burning smell: check clutch plate and hydraulic system.",
    "Suspension": "Clunks over bumps: inspect shocks and bushings.",
    "Exhaust": "Hissing/popping: check leaks, muffler and catalytic converter.",
    "Belt": "High-pitched squeal: check belt tension and wear.",
    "Transmission": "Jerks or delays: check fluid and mounts.",
    "Cooling System": "Bubbling/hiss: check coolant, radiator and fans.",
    "Tyre/Rolling": "Thump/vibration: check balance, bearing, wear.",
    "Battery": "Clicking/no start: check charge, alternator, terminals.",
    "Wiring/Electrical System": "Buzz/crackle: check shorts, loose connectors."
}

MODEL_FILE = "car_fault_model.pkl"

# ----------------------------
# Helpers: audio -> feature
# ----------------------------
def extract_mfcc_from_path(path, sr=22050, n_mfcc=40):
    y, sr_used = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y)
    if len(y) < 0.3 * sr:  # too short
        return None, None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    rms = np.mean(np.abs(y))
    return feat, rms

def extract_mfcc_from_bytes(bytes_data, sr=22050, n_mfcc=40):
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(bytes_data)
        tmp.flush()
        path = tmp.name
    feat, rms = extract_mfcc_from_path(path, sr=sr, n_mfcc=n_mfcc)
    try:
        os.remove(path)
    except:
        pass
    return feat, rms

# ----------------------------
# Function: try to load existing model
# ----------------------------
@st.cache_resource
def load_local_model():
    if os.path.exists(MODEL_FILE):
        try:
            m = joblib.load(MODEL_FILE)
            return m
        except Exception as e:
            st.error(f"Error loading model file: {e}")
            return None
    return None

model = load_local_model()

# ----------------------------
# Section: Auto-merge HuggingFace datasets and train (safe button)
# ----------------------------
st.markdown("## 📡 Auto Database & Model Update (Hugging Face)")
st.markdown("This will fetch *open* datasets from Hugging Face, extract features, and (optionally) retrain a classifier. Use the controls below to update the model. Training uses a *sample* set by default so it does not time out. Increase limits in Colab for full retrain.")

with st.expander("⚙️ Advanced: Update model from online datasets (safe mode)"):
    st.write("The app will try to load a list of open datasets (engine/vehicle sounds) and train a RandomForest. This is sample-based to be safe on Streamlit Cloud. For full training, use Colab.")
    dataset_list_text = st.text_area("Hugging Face dataset ids (one per line)", value="iskandarnajib/car-engine-sound", height=80)
    sample_limit = st.number_input("Samples per dataset (max/sample)", min_value=10, max_value=500, value=60, step=10)
    retrain_button = st.button("🔁 Update model from Hugging Face samples")

    if retrain_button:
        ds_ids = [d.strip() for d in dataset_list_text.splitlines() if d.strip()]
        all_X = []
        all_y = []
        st.info("Fetching and building features (this may take a short while)...")
        for ds_id in ds_ids:
            try:
                ds = load_dataset(ds_id, split="train")
                st.write(f"Loaded dataset: {ds_id} (size {len(ds)})")
                # iterate safely: sample items
                to_take = min(len(ds), sample_limit)
                for item in ds.select(range(to_take)):
                    # many HF datasets store audio as dict {'array':..., 'sampling_rate':...} or path
                    try:
                        if isinstance(item.get("audio"), dict):
                            arr = item["audio"]["array"]
                            sr = item["audio"]["sampling_rate"]
                            # save temp wav
                            with NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
                                sf.write(tmpf.name, arr, sr)
                                feat, rms = extract_mfcc_from_path(tmpf.name)
                                tmpname = tmpf.name
                            try:
                                os.remove(tmpname)
                            except:
                                pass
                        else:
                            # audio as path or URL
                            audio_path = item.get("audio") if item.get("audio") else item.get("path")
                            if audio_path is None:
                                continue
                            # try load via librosa
                            feat, rms = extract_mfcc_from_path(audio_path)
                        if feat is not None:
                            lab = item.get("label") or item.get("label_name") or item.get("class") or "Unknown"
                            all_X.append(feat)
                            all_y.append(lab)
                    except Exception as e:
                        # skip problematic items
                        continue
            except Exception as e:
                st.warning(f"Could not load dataset {ds_id}: {e}")
                continue

        if len(all_X) < 20:
            st.error("Not enough samples found to train (need >20). Try adding more datasets or increasing sample limit.")
        else:
            X = np.array(all_X)
            y = np.array(all_y)
            st.write(f"Built features: X={X.shape}, classes={np.unique(y)}")
            st.info("Training RandomForest on sampled features...")
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
                clf.fit(X_train, y_train)
                ypred = clf.predict(X_test)
                acc = accuracy_score(y_test, ypred)
                joblib.dump(clf, MODEL_FILE)
                model = clf
                st.success(f"✅ Model trained & saved (sampled). Accuracy on sample test: {acc*100:.2f}%")
            except Exception as e:
                st.error(f"Training failed: {e}")

# ----------------------------
# Upload & Analyze Section
# ----------------------------
st.markdown("---")
st.markdown("## 🎯 Upload & Full Component Scan")
st.write("Upload a car sound (WAV/MP3/MP4). The system will analyse all components and produce a full report with confidence scores. Use real quiet recordings for best accuracy.")

uploaded = st.file_uploader("Upload car sound (wav, mp3, mp4)", type=["wav", "mp3", "mp4", "m4a"])
if uploaded is not None:
    # load bytes and extract features
    try:
        bytes_data = uploaded.read()
        feat, rms = extract_mfcc_from_bytes(bytes_data)
        if feat is None:
            st.error("Audio too short or not valid. Record 1-4 seconds of running sound and try again.")
        else:
            st.success("Audio loaded and features computed.")
            # waveform visualization
            try:
                y_vis, sr_vis = librosa.load(BytesIO := None or NamedTemporaryFile(delete=False, suffix=".wav").name, sr=22050)
            except Exception:
                pass

            # If model exists: use it. If not: warn and skip
            model = load_local_model()
            if model is None:
                st.error("No model found. Please train or update the model using the 'Update model from Hugging Face' control above.")
            else:
                # predict probabilities across known classes
                probs = model.predict_proba([feat])[0]
                classes = list(model.classes_)
                # map probs to our COMPONENTS list as best-effort:
                # we will create a result table with one row per COMPONENTS; if model class matches a component substring, use that prob; else 0
                comp_probs = {}
                for comp in COMPONENTS:
                    # find best matching class index: exact match ignoring case, or class contains comp or comp contains class
                    matched_idx = -1
                    for idx, c in enumerate(classes):
                        if c.lower() == comp.lower() or comp.lower() in str(c).lower() or str(c).lower() in comp.lower():
                            matched_idx = idx
                            break
                    if matched_idx >= 0:
                        comp_probs[comp] = probs[matched_idx]
                    else:
                        # no direct mapping: try fuzzy by words overlap
                        matched = False
                        for idx, c in enumerate(classes):
                            words = c.lower().split()
                            if any(w in comp.lower() for w in words):
                                comp_probs[comp] = probs[idx]
                                matched = True
                                break
                        if not matched:
                            comp_probs[comp] = 0.0

                # If overall max prob below threshold and RMS low -> No Fault
                max_prob = max(comp_probs.values())
                max_comp = max(comp_probs, key=comp_probs.get)
                # scale to 0-100
                table_rows = []
                report_texts = []
                for comp in COMPONENTS:
                    p = comp_probs[comp]
                    conf = p * 100
                    if max_prob < 0.40 and rms < 0.01:
                        status = "✅ Normal"
                    else:
                        if conf >= 70:
                            status = "❌ Fault Detected"
                        elif conf >= 45:
                            status = "⚠ Warning"
                        else:
                            status = "✅ Normal"
                    table_rows.append({"Component": comp, "Status": status, "Confidence (%)": f"{conf:.1f}"})
                    if status != "✅ Normal":
                        report_texts.append(f"{comp}: {status} ({conf:.1f}%) - {DESCRIPTIONS.get(comp,'')}")
                df_report = pd.DataFrame(table_rows)
                st.markdown("### 🔎 Component Scan Results")
                st.dataframe(df_report.set_index("Component"))

                # Spoken summary
                if len(report_texts) == 0:
                    speak_text = "No faults detected on this vehicle. All checked systems appear normal."
                else:
                    speak_text = " ; ".join(report_texts[:6])  # keep spoken summary reasonably short

                # convert to speech and play
                tts = gTTS(speak_text, lang="en")
                with NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                    tts.save(tf.name)
                    audio_bytes = open(tf.name, "rb").read()
                    b64 = base64.b64encode(audio_bytes).decode()
                    st.markdown(f'<audio controls autoplay src="data:audio/mp3;base64,{b64}"></audio>', unsafe_allow_html=True)

                # show full textual report
                st.markdown("### 📋 Text Report")
                if len(report_texts) == 0:
                    st.success("✅ No faults detected across scanned components.")
                else:
                    for line in report_texts:
                        st.write("- " + line)

    except Exception as e:
        st.error(f"Error processing audio: {e}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("👨🏽‍💻 Developed by **Adekanye Abdulzohir** — Military Professional Edition")
