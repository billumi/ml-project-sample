
from flask import Flask, request, jsonify
import pickle, os, numpy as np

app = Flask(__name__)
MODEL_PATH = os.environ.get("MODEL_PATH", "models/saved/model.pkl")

@app.route("/ping")
def ping():
    return jsonify(status="ok")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "features" not in data:
        return jsonify(error="missing features"), 400
    if not os.path.exists(MODEL_PATH):
        return jsonify(error="model not found"), 500
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    arr = np.array(data["features"]).reshape(1, -1)
    pred = model.predict(arr).tolist()
    return jsonify(prediction=pred)
