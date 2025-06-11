from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

# Load model
with open('model/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

def classify_aqi(aqi_value):
    if aqi_value <= 50:
        return "Baik", "Kualitas udara sangat baik dan tidak membahayakan kesehatan."
    elif aqi_value <= 100:
        return "Sedang", "Kualitas udara masih dapat diterima, namun beberapa polutan mungkin berisiko bagi individu yang sangat sensitif."
    elif aqi_value <= 150:
        return "Tidak Sehat untuk Kelompok Sensitif", "Kelompok sensitif seperti anak-anak dan lansia mungkin mulai merasakan dampak kesehatan."
    elif aqi_value <= 200:
        return "Tidak Sehat", "Setiap orang mungkin mulai mengalami efek kesehatan; kelompok sensitif bisa mengalami dampak yang lebih serius."
    elif aqi_value <= 300:
        return "Sangat Tidak Sehat", "Peringatan kesehatan: semua orang bisa mengalami efek serius terhadap kesehatan."
    elif aqi_value <= 400:
        return "Berbahaya", "Kondisi darurat kesehatan; seluruh populasi dapat terkena dampak serius."
    else:
        return "Ekstrem Berbahaya", "Kualitas udara sangat berbahaya dan bisa menyebabkan dampak kesehatan parah bahkan untuk orang sehat."

def evaluate_model(model):
    # Muat ulang data dan evaluasi model setiap kali
    data = pd.read_csv('data/global_air_quality_processed.csv')
    features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
    target = 'AQI Value'
    data = data.dropna(subset=features + [target])
    X = data[features]
    y = data[target]
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    return {
        'MAE': round(mae, 2),
        'MSE': round(mse, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4)
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    evaluation = evaluate_model(model)  # evaluasi dinamis

    if request.method == 'POST':
        try:
            co = float(request.form['CO'])
            o3 = float(request.form['O3'])
            no2 = float(request.form['NO2'])
            pm25 = float(request.form['PM25'])

            if any(val < 0 for val in [co, o3, no2, pm25]):
                raise ValueError("Nilai tidak boleh negatif.")

            input_df = pd.DataFrame([[co, o3, no2, pm25]], 
                                    columns=['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value'])

            pred_aqi = model.predict(input_df)[0]
            category, message = classify_aqi(pred_aqi)

            return render_template('index.html',
                                   prediction=round(pred_aqi, 2),
                                   category=category,
                                   message=message,
                                   input_values={
                                       'CO': co, 'O3': o3, 'NO2': no2, 'PM2.5': pm25
                                   },
                                   evaluation=evaluation)
        except Exception as e:
            return f"Terjadi kesalahan: {e}"

    return render_template('index.html', evaluation=evaluation)

if __name__ == '__main__':
    app.run(debug=True)