import os
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained ML models and encoders
MODEL_DIR = '.'
clf_model = joblib.load(os.path.join(MODEL_DIR, 'drought_classifier.pkl'))
reg_model = joblib.load(os.path.join(MODEL_DIR, 'irrigation_regressor.pkl'))
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    error_msg = None

    if request.method == 'POST':
        try:
            # Capture Form Inputs
            future_date_str = request.form.get('future_date')
            rainfall = float(request.form.get('rainfall'))
            temperature = float(request.form.get('temperature'))
            humidity = float(request.form.get('humidity'))
            soil_moisture = float(request.form.get('soil_moisture'))
            wind_speed = float(request.form.get('wind_speed'))
            et0 = float(request.form.get('et0'))
            drought_index = float(request.form.get('drought_index'))

            # Extract Month and Day from future date
            date_obj = datetime.strptime(future_date_str, '%Y-%m-%d')
            month = date_obj.month
            day = date_obj.day

            # Build Feature DataFrame for ML Model
            input_data = pd.DataFrame([[
                month, day, rainfall, temperature, 
                humidity, soil_moisture, wind_speed, et0, drought_index
            ]], columns=[
                'Month', 'Day', 'Rainfall (mm)', 'Temperature (°C)', 
                'Relative Humidity (%)', 'Soil Moisture (%)', 
                'Wind Speed (m/s)', 'Evapotranspiration ET0 (mm)', 'Drought Index (SPI-like)'
            ])

            # Predict Drought Class & Irrigation Requirement
            pred_class_encoded = clf_model.predict(input_data)[0]
            drought_class = label_encoder.inverse_transform([pred_class_encoded])[0]

            irrigation_rec = reg_model.predict(input_data)[0]

            prediction_result = {
                'drought_class': drought_class,
                'irrigation_recommendation': round(float(irrigation_rec), 2),
                'date': future_date_str
            }

        except ValueError as e:
            error_msg = f"Invalid input format: {str(e)}"

    return render_template('index.html', result=prediction_result, error_msg=error_msg)

if __name__ == '__main__':
    app.run(debug=True)