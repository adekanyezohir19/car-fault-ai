import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from utils.feature_extraction import extract_features

# Define your dataset path
DATA_DIR = "sounds"

X, y = [], []
for label in os.listdir(DATA_DIR):
    folder = os.path.join(DATA_DIR, label)
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            if file.endswith((".wav", ".mp3")):
                file_path = os.path.join(folder, file)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(label)

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print(f"✅ Training complete with accuracy: {acc:.2f}")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/car_fault_model.pkl")
