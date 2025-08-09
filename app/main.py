# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load models
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")
kmeans = joblib.load("models/kmeans.pkl")

# App
app = FastAPI(title="Anomaly Detection API")

class Metrics(BaseModel):
    cpu_util_percent: float
    memory_util_percent: float
    disk_io: float
    network_io: float

@app.post("/predict")
def predict_anomaly(metrics: Metrics):
    data = np.array([[metrics.cpu_util_percent,
                      metrics.memory_util_percent,
                      metrics.disk_io,
                      metrics.network_io]])
    
    data_scaled = scaler.transform(data)
    data_pca = pca.transform(data_scaled)
    cluster = kmeans.predict(data_pca)[0]
    
    return {
        "cluster": int(cluster),
        "is_anomaly": bool(cluster == 1)  # assuming cluster 1 is anomalies
    }
