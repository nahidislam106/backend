from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Crop label mapping (update if needed)
label_encoder = {
    0: "Balsam_Apple",
    1: "Cauliflower",
    2: "Chili",
    3: "Cucumber",
    4: "Maize",
    5: "Sweet_pumpkin"
}

# Reverse the dictionary for decoding
label_decoder = {v: k for k, v in label_encoder.items()}

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    pH: float
    EC: float
    temperature: float
    humidity: float

@app.get("/")
def read_root():
    return {"message": "✅ ফাস্টএপিআই চালু আছে!"}

@app.post("/predict")
def predict(data: CropInput):
    features = [[
        data.temperature,
        data.EC,
        data.pH,
        data.humidity,
        data.N,
        data.P,
        data.K
    ]]

    scaled = scaler.transform(features)
    probs = model.predict_proba(scaled)[0]

    # Get top 3-4 crop predictions
    top_indices = np.argsort(probs)[::-1][:4]
    recommendations = []

    for idx in top_indices:
        crop_name = label_encoder.get(idx)
        prob = probs[idx]
        recommendations.append({
            "crop": crop_name,
            "probability": float(round(prob, 4))
        })

    return {"সুপারিশকৃত ফসল": recommendations}
