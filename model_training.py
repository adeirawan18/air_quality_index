import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from scipy.stats import zscore

if not os.path.exists('model'):
    os.makedirs('model')

# Preprocessing
def preprocess_data(df, features, target):
    # Ambil hanya kolom yang diperlukan
    df = df[features + [target]].copy()

    # Hapus nilai kosong
    df = df.dropna()

    # Konversi semua kolom ke numerik
    for col in features + [target]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop jika ada NaN hasil konversi
    df = df.dropna()

    # Hapus outlier
    z_scores = np.abs(zscore(df))
    df = df[(z_scores < 3).all(axis=1)]

    return df

# Baca dan Preprocess Dataset
print("Membaca dataset...")
data = pd.read_csv('data/global_air_quality_processed.csv')

# Tentukan fitur dan target
features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
target = 'AQI Value'

print("Melakukan preprocessing...")
data = preprocess_data(data, features, target)

# Pisahkan fitur dan target
X = data[features]
y = data[target]

# Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat dan latih model
print("Melatih model Random Forest...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Simpan model ke file
with open('model/random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model sudah disimpan di: model/random_forest_model.pkl")

# Evaluasi model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Evaluasi Model:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")
