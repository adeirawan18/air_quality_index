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
        return "Sehat", "Kualitas udara sangat baik dan tidak membahayakan kesehatan."
    elif aqi_value <= 100:
        return "Cukup Sehat", "Masih dapat diterima, namun bisa berisiko bagi individu sensitif."
    elif aqi_value <= 150:
        return "Tidak Sehat untuk Kelompok Sensitif", "Kelompok seperti anak-anak dan lansia mungkin terdampak."
    elif aqi_value <= 200:
        return "Tidak Sehat", "Semua orang mungkin mulai terdampak."
    elif aqi_value <= 300:
        return "Sangat Tidak Sehat", "Efek serius bagi semua kelompok populasi."
    elif aqi_value <= 400:
        return "Berbahaya", "Kondisi darurat kesehatan untuk seluruh populasi."
    else:
        return "Ekstrem Berbahaya", "Sangat berbahaya bahkan bagi orang sehat."

def evaluate_model(model):
    data = pd.read_csv('data/global_air_quality_processed.csv')
    features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
    target = 'AQI Value'
    data.dropna(subset=features + [target], inplace=True)
    X = data[features]
    y = data[target]
    y_pred = model.predict(X)

    return {
        'MAE': round(mean_absolute_error(y, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y, y_pred)), 2),
        'R2': round(r2_score(y, y_pred), 4)
    }

@app.route('/')
def landing():
    return render_template('index.html')  

@app.route('/prediksi', methods=['GET', 'POST'])
def index():
    evaluation = evaluate_model(model)
    if request.method == 'POST':
        try:
            co = float(request.form['CO'])
            o3 = float(request.form['O3'])
            no2 = float(request.form['NO2'])
            pm25 = float(request.form['PM25'])

            if any(v < 0 for v in [co, o3, no2, pm25]):
                raise ValueError("Nilai tidak boleh negatif.")

            input_df = pd.DataFrame([[co, o3, no2, pm25]], columns=[
                'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value'
            ])

            pred_aqi = model.predict(input_df)[0]
            category, message = classify_aqi(pred_aqi)

            return render_template('predict.html',
                                   prediction=round(pred_aqi, 2),
                                   category=category,
                                   message=message,
                                   input_values={
                                       'CO': co, 'O3': o3, 'NO2': no2, 'PM2.5': pm25
                                   },
                                   evaluation=evaluation)
        except Exception as e:
            return f"❌ Terjadi kesalahan: {e}"

    return render_template('predict.html', evaluation=evaluation)


if __name__ == '__main__':
    app.run(debug=True)
