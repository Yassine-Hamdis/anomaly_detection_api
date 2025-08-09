# train_model.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path

# Paths
Path("models").mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv("data/server_usage.csv")

# Features
X = df.values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# KMeans
kmeans = KMeans(n_clusters=2, random_state=42)  # 2 = normal vs anomaly
kmeans.fit(X_pca)

# Save models
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(pca, "models/pca.pkl")
joblib.dump(kmeans, "models/kmeans.pkl")

print("✅ Models trained and saved in /models")
