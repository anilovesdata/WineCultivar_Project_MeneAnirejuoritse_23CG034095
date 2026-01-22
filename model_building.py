# model_building.py
"""
Wine Cultivar Prediction Model - Single Script Version
- Loads sklearn wine dataset
- Selects 6 features
- Scales data
- Trains Random Forest Classifier
- Evaluates performance
- Saves model + scaler using joblib
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ────────────────────────────────────────────────
#  CONFIGURATION
# ────────────────────────────────────────────────
SELECTED_FEATURES = [
    'alcohol',
    'malic_acid',
    'ash',
    'magnesium',
    'flavanoids',
    'proline'
]

MODEL_SAVE_PATH = "wine_cultivar_model.pkl"
SCALER_SAVE_PATH = "scaler.pkl"

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)
MODEL_SAVE_PATH = os.path.join("model", MODEL_SAVE_PATH)
SCALER_SAVE_PATH = os.path.join("model", SCALER_SAVE_PATH)

# ────────────────────────────────────────────────
#  1. Load Dataset
# ────────────────────────────────────────────────
print("Loading Wine dataset...")
wine = load_wine()
df = pd.DataFrame(
    data=np.c_[wine.data, wine.target],
    columns=wine.feature_names + ['cultivar']
)

print("\nDataset shape:", df.shape)
print("Class distribution:\n", df['cultivar'].value_counts())

# ────────────────────────────────────────────────
#  2. Feature Selection & Target
# ────────────────────────────────────────────────
print("\nSelected features:", SELECTED_FEATURES)
X = df[SELECTED_FEATURES].copy()
y = df['cultivar'].astype(int)

# ────────────────────────────────────────────────
#  3. Train-Test Split
# ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ────────────────────────────────────────────────
#  4. Feature Scaling (very important for many algorithms)
# ────────────────────────────────────────────────
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ────────────────────────────────────────────────
#  5. Train Model
# ────────────────────────────────────────────────
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# ────────────────────────────────────────────────
#  6. Evaluation
# ────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["class 0", "class 1", "class 2"],
    digits=4
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance (optional but informative)
print("\nFeature Importances:")
for name, imp in zip(SELECTED_FEATURES, model.feature_importances_):
    print(f"{name:18} : {imp:.5f}")

# ────────────────────────────────────────────────
#  7. Save Model & Scaler
# ────────────────────────────────────────────────
import joblib

print(f"\nSaving model to:   {MODEL_SAVE_PATH}")
joblib.dump(model, MODEL_SAVE_PATH)

print(f"Saving scaler to:  {SCALER_SAVE_PATH}")
joblib.dump(scaler, SCALER_SAVE_PATH)

print("\nModel building completed successfully.")
print("Files ready for deployment:")
print(f"  • {MODEL_SAVE_PATH}")
print(f"  • {SCALER_SAVE_PATH}")