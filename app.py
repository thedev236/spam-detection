import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Configure Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load Trained Model Artifacts
MODEL_PATH = "model/classifier.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logging.info("Model and Vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model artifacts: {e}")
    raise RuntimeError("Model files not found. Run train.py first.")

def analyze_text(text):
    """Core classification engine."""
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    spam_prob = round(probabilities[1] * 100, 2)
    ham_prob = round(probabilities[0] * 100, 2)
    label = "SPAM" if prediction == 1 else "HAM"
    confidence = spam_prob if label == "SPAM" else ham_prob

    # Log prediction
    logging.info(f"Input: '{text[:30]}...' | Result: {label} ({confidence}%)")

    return {
        "label": label,
        "confidence": confidence,
        "spam_probability": spam_prob,
        "ham_probability": ham_prob
    }

# ---------------------------------------------------------
# Web Routes
# ---------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("email_text", "").strip()
        if user_text:
            result = analyze_text(user_text)

    return render_template("index.html", result=result, text=user_text)

# ---------------------------------------------------------
# REST API Endpoint
# ---------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Field 'text' cannot be empty"}), 400

    analysis = analyze_text(text)
    return jsonify({
        "status": "success",
        "timestamp": datetime.utcnow().isoformat(),
        "data": analysis
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)