# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Paths (assuming model folder is at the same level as app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "wine_cultivar_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# Load model and scaler once when the app starts
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Get form values (all required)
            alcohol     = float(request.form["alcohol"])
            malic_acid  = float(request.form["malic_acid"])
            ash         = float(request.form["ash"])
            magnesium   = float(request.form["magnesium"])
            flavanoids  = float(request.form["flavanoids"])
            proline     = float(request.form["proline"])

            # Prepare input array (must match the 6 features order)
            input_data = np.array([[alcohol, malic_acid, ash, magnesium, flavanoids, proline]])

            # Scale
            input_scaled = scaler.transform(input_data)

            # Predict
            pred_class = model.predict(input_scaled)[0]

            # Map class to friendly name
            cultivar_names = {
                0: "Cultivar 1",
                1: "Cultivar 2",
                2: "Cultivar 3"
            }
            prediction = cultivar_names.get(pred_class, "Unknown")

        except ValueError:
            error = "Please enter valid numeric values for all fields."
        except Exception as e:
            error = f"Prediction error: {str(e)}"

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(debug=True)