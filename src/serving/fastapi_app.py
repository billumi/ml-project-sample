
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import os

app = FastAPI(title="ML Toolkit - FastAPI Serving")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/saved/model.pkl")

class PredictRequest(BaseModel):
    features: list

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/predict")
async def predict(req: PredictRequest):
    if not os.path.exists(MODEL_PATH):
        return {"error": f"model not found at {MODEL_PATH}"}
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    arr = np.array(req.features).reshape(1, -1)
    pred = model.predict(arr).tolist()
    return {"prediction": pred}
