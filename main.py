from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import os

# --- Load model and scaler ---
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler file not found!")

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

# --- Crop label mapping ---
# Make sure indices match exactly the label encoding used during training
label_encoder = {
    0: "Rice",
    1: "Maize",
    2: "Chickpea",
    3: "Kidneybeans",
    4: "Pigeonpeas",
    5: "Mothbeans",
    6: "Mungbean",
    7: "Blackgram",
    8: "Lentil",
    9: "Pomegranate",
    10: "Banana",
    11: "Mango",
    12: "Grapes",
    13: "Watermelon",
    14: "Muskmelon",
    15: "Apple",
    16: "Orange",
    17: "Papaya",
    18: "Coconut",
    19: "Cotton",
    20: "Jute",
    21: "Coffee"
}

app = FastAPI(title="Crop Recommendation API")

# --- Allow frontend connection ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request model matching your new columns ---
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    pH: float
    EC: float

# --- Root endpoint ---
@app.get("/")
def read_root():
    return {"message": "✅ FastAPI Crop Recommendation API is running!"}

# --- Prediction endpoint ---
@app.post("/predict")
def predict(data: CropInput):
    # Feature order must match training: N, P, K, temperature, humidity, pH, EC
    features = [[
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.pH,
        data.EC
    ]]

    # Scale features
    scaled = scaler.transform(features)

    # Predict probabilities
    probs = model.predict_proba(scaled)[0]

    # Get top 4 predictions
    top_indices = np.argsort(probs)[::-1][:4]
    recommendations = []

    for idx in top_indices:
        crop_name = label_encoder.get(idx, f"Crop_{idx}")
        prob = probs[idx]
        recommendations.append({
            "crop": crop_name,
            "probability": float(round(prob, 4))
        })

    return {"সুপারিশকৃত ফসল": recommendations}
