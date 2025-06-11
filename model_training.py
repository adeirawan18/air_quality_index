import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Buat folder model jika belum ada
if not os.path.exists('model'):
    os.makedirs('model')

# Baca dataset
data = pd.read_csv('data/global_air_quality_processed.csv')

# Tentukan fitur dan target
features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
target = 'AQI Value'

# Bersihkan data dari nilai kosong
data = data.dropna(subset=features + [target])

# Pisahkan fitur dan target
X = data[features]
y = data[target]

# Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat dan latih model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Simpan model ke file
with open('model/random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model sudah disimpan di model/random_forest_model.pkl")


y_pred = model.predict(X_test)

# Hitung metrik evaluasi
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Tampilkan hasil evaluasi
print("\nEvaluasi Model:")
print(f"MAE  (Mean Absolute Error): {mae:.2f}")
print(f"MSE  (Mean Squared Error) : {mse:.2f}")
print(f"RMSE (Root MSE)           : {rmse:.2f}")
print(f"R² Score                  : {r2:.4f}")
