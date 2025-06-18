import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import zscore
import matplotlib.pyplot as plt

# Buat folder untuk menyimpan model
os.makedirs('model', exist_ok=True)

# Fitur dan target
FEATURES = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
TARGET = 'AQI Value'

# Fungsi preprocessing
def preprocess_data(df):
    df = df[FEATURES + [TARGET]].copy()
    df.dropna(inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    df = df[(np.abs(zscore(df)) < 3).all(axis=1)]  # hapus outlier
    return df

# Load dan preprocess data
print("📥 Membaca dan memproses dataset...")
df = pd.read_csv('data/global_air_quality_processed.csv')
df = preprocess_data(df)
X, y = df[FEATURES], df[TARGET]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("🧠 Melatih model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('model/random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model disimpan di 'model/random_forest_model.pkl'")

# Evaluasi model
y_pred = model.predict(X_test)
print("\n📊 Evaluasi Model:")
print(f"MAE       : {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE      : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R² Train  : {model.score(X_train, y_train):.2f}")
print(f"R² Test   : {r2_score(y_test, y_pred):.2f}")

# Visualisasi prediksi vs aktual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Actual vs Predicted AQI')
plt.grid(True)
plt.tight_layout()
plt.show()
