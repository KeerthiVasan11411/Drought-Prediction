import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_excel('Kancheepuram_Drought_Complete_2015_2025_SYNTHETIC.xlsx', sheet_name='Daily Dataset')

# 2. Select Features & Targets
features = [
    'Month', 'Day', 'Rainfall (mm)', 'Temperature (°C)', 
    'Relative Humidity (%)', 'Soil Moisture (%)', 
    'Wind Speed (m/s)', 'Evapotranspiration ET0 (mm)', 'Drought Index (SPI-like)'
]

X = df[features]
y_class = df['Drought Class']
y_reg = df['Irrigation Recommendation (mm)']

# 3. Encode Target Labels for Classification
label_encoder = LabelEncoder()
y_class_encoded = label_encoder.fit_transform(y_class)

# 4. Train Models with class balancing
print("Training Models...")
clf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf_model.fit(X, y_class_encoded)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X, y_reg)

# 5. Save Artifacts
joblib.dump(clf_model, 'drought_classifier.pkl')
joblib.dump(reg_model, 'irrigation_regressor.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Models successfully trained and saved!")