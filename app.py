import os
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request

app = Flask(__name__)

# Load models and dataset
MODEL_DIR = '.'
clf_model = joblib.load(os.path.join(MODEL_DIR, 'drought_classifier.pkl'))
reg_model = joblib.load(os.path.join(MODEL_DIR, 'irrigation_regressor.pkl'))
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))

df_dataset = pd.read_excel('Kancheepuram_Drought_Complete_2015_2025_SYNTHETIC.xlsx', sheet_name='Daily Dataset')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    error_msg = None
    weather_summary = None

    if request.method == 'POST':
        try:
            date_str = request.form.get('prediction_date')
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            month = date_obj.month
            day = date_obj.day

            # Find historical records for this month/day in Kancheepuram dataset
            matched = df_dataset[(df_dataset['Month'] == month) & (df_dataset['Day'] == day)]
            
            if matched.empty:
                matched = df_dataset[df_dataset['Month'] == month]

            avg_rainfall = float(matched['Rainfall (mm)'].mean())
            avg_temp = float(matched['Temperature (°C)'].mean())
            avg_humidity = float(matched['Relative Humidity (%)'].mean())
            avg_soil = float(matched['Soil Moisture (%)'].mean())
            avg_wind = float(matched['Wind Speed (m/s)'].mean())
            avg_et0 = float(matched['Evapotranspiration ET0 (mm)'].mean())
            avg_drought_idx = float(matched['Drought Index (SPI-like)'].mean())

            # Build feature row with explicit column names
            feature_names = [
                'Month', 'Day', 'Rainfall (mm)', 'Temperature (°C)', 
                'Relative Humidity (%)', 'Soil Moisture (%)', 
                'Wind Speed (m/s)', 'Evapotranspiration ET0 (mm)', 'Drought Index (SPI-like)'
            ]
            
            input_data = pd.DataFrame([[
                month, day, avg_rainfall, avg_temp, 
                avg_humidity, avg_soil, avg_wind, avg_et0, avg_drought_idx
            ]], columns=feature_names)

            # Predict Classification & Regression
            pred_encoded = clf_model.predict(input_data)[0]
            drought_class = label_encoder.inverse_transform([pred_encoded])[0]

            irrigation_rec = reg_model.predict(input_data)[0]

            prediction_result = {
                'drought_class': drought_class,
                'irrigation_recommendation': round(float(irrigation_rec), 2),
                'date': date_str
            }

            weather_summary = {
                'rainfall': round(avg_rainfall, 2),
                'temperature': round(avg_temp, 2),
                'humidity': round(avg_humidity, 2),
                'soil_moisture': round(avg_soil, 2),
                'wind_speed': round(avg_wind, 2),
                'et0': round(avg_et0, 2),
                'drought_index': round(avg_drought_idx, 2)
            }

        except Exception as e:
            error_msg = f"Prediction Error: {str(e)}"

    return render_template('index.html', result=prediction_result, weather=weather_summary, error_msg=error_msg)

if __name__ == '__main__':
    app.run(debug=True)
