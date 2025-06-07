import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

if not os.path.exists('model'):
    os.makedirs('model')

data = pd.read_csv('data/global_air_quality_processed.csv')

features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
target = 'AQI Value'

# Drop rows yang ada nilai kosong pada fitur atau target
data = data.dropna(subset=features + [target])

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open('model/random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model sudah disimpan di model/random_forest_model.pkl")
