"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                      Sougnabe's changes start                              ║
║                                                                            ║
║  WHAT: Complete End-to-End Prediction Pipeline (MySQL Version)            ║
║  WHY:  Task 4 requirement - integrate all tasks for predictions           ║
║  INCLUDES:                                                                 ║
║    - Fetch data from MySQL API                                            ║
║    - Preprocess data (same pipeline as Task 1)                            ║
║    - Load trained model                                                   ║
║    - Make predictions                                                     ║
║    - Output forecasts                                                     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import requests
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import json
import sys

print("="*80)
print("POWER CONSUMPTION PREDICTION PIPELINE - MYSQL VERSION")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = "http://localhost:5000"  # MySQL API
MODEL_PATH = "../../mongoDB-Team/task-1_EDA/outputs/models/random_forest_model.pkl"
SCALER_PATH = "../../mongoDB-Team/task-1_EDA/outputs/models/scaler.pkl"

print("\n[Configuration]")
print(f"  • API URL: {API_BASE_URL}")
print(f"  • Model: {MODEL_PATH}")
print(f"  • Scaler: {SCALER_PATH}")

# ============================================================================
# STEP 1: FETCH DATA FROM API
# ============================================================================

print("\n" + "="*80)
print("STEP 1: FETCHING DATA FROM MYSQL API")
print("="*80)

try:
    # Fetch latest 200 readings for context
    print("\n[Fetching] Latest readings from API...")
    response = requests.get(f"{API_BASE_URL}/sql/readings?page=1&limit=200")
    
    if response.status_code != 200:
        print(f"❌ API Error: {response.status_code}")
        print(f"   Response: {response.text}")
        print("\n   Please ensure MySQL API is running:")
        print("   cd mysql_team/task-3_api")
        print("   uvicorn api:app --port 5000")
        exit(1)
    
    data = response.json()
    readings = data['results']
    
    print(f"✓ Fetched {len(readings)} readings successfully")
    
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to API. Please ensure:")
    print("   1. MySQL API is running: uvicorn mysql_team.task-3_api.api:app --port 5000")
    print("   2. MySQL database has data loaded")
    exit(1)
except Exception as e:
    print(f"❌ Error fetching data: {e}")
    exit(1)

# ============================================================================
# STEP 2: PREPROCESS DATA (SAME AS TASK 1)
# ============================================================================

print("\n" + "="*80)
print("STEP 2: PREPROCESSING DATA")
print("="*80)

print("\n[Processing] Converting API response to DataFrame...")

# Convert API response to DataFrame
df = pd.DataFrame(readings)
df['datetime'] = pd.to_datetime(df['recorded_at'])
df = df.sort_values('datetime').reset_index(drop=True)

# Rename columns to match training data format
df = df.rename(columns={
    'global_active_power': 'Global_active_power',
    'global_reactive_power': 'Global_reactive_power',
    'voltage': 'Voltage',
    'global_intensity': 'Global_intensity',
    'kitchen': 'Sub_metering_1',
    'laundry': 'Sub_metering_2',
    'water_heater_ac': 'Sub_metering_3'
})

print(f"✓ Created DataFrame: {len(df)} rows")

# ============================================================================
# STEP 3: FEATURE ENGINEERING (LAG & ROLLING FEATURES)
# ============================================================================

print("\n[Engineering] Creating lag and rolling features...")

# Create lagged features (same as training)
df['power_lag_1h'] = df['Global_active_power'].shift(1)
df['power_lag_24h'] = df['Global_active_power'].shift(24)
df['power_lag_168h'] = df['Global_active_power'].shift(168)

# Create rolling statistics
df['power_ma_24h'] = df['Global_active_power'].rolling(window=24, min_periods=1).mean()
df['power_ma_168h'] = df['Global_active_power'].rolling(window=168, min_periods=1).mean()

# Handle NaN from lagging
df['power_lag_1h'].fillna(df['Global_active_power'].mean(), inplace=True)
df['power_lag_24h'].fillna(df['Global_active_power'].mean(), inplace=True)
df['power_lag_168h'].fillna(df['Global_active_power'].mean(), inplace=True)

print("✓ Features engineered")

# ============================================================================
# STEP 4: LOAD TRAINED MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 4: LOADING TRAINED MODEL")
print("="*80)

try:
    print(f"\n[Loading] Model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print(f"✓ Model loaded: {type(model).__name__}")
    
    print(f"\n[Loading] Scaler from {SCALER_PATH}...")
    scaler = joblib.load(SCALER_PATH)
    print("✓ Scaler loaded")
    
except FileNotFoundError as e:
    print(f"❌ Model file not found: {e}")
    print("   Please run mongoDB-Team/task-1_EDA/model_training.py first")
    exit(1)

# ============================================================================
# STEP 5: PREPARE FEATURES FOR PREDICTION
# ============================================================================

print("\n" + "="*80)
print("STEP 5: PREPARING FEATURES")
print("="*80)

feature_cols = [
    'Global_reactive_power', 'Voltage', 'Global_intensity',
    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
    'hour', 'day_of_week', 'month', 'is_weekend',
    'power_lag_1h', 'power_lag_24h', 'power_lag_168h',
    'power_ma_24h', 'power_ma_168h'
]

X = df[feature_cols]

# Random Forest doesn't need scaling
if hasattr(model, 'n_estimators'):
    X_prepared = X
    print("✓ Random Forest model - no scaling needed")
else:
    X_prepared = scaler.transform(X)
    print("✓ Features scaled")

print(f"✓ Prepared {len(X_prepared)} samples for prediction")

# ============================================================================
# STEP 6: MAKE PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 6: MAKING PREDICTIONS")
print("="*80)

print("\n[Predicting] Running model inference...")
predictions = model.predict(X_prepared)

print(f"✓ Generated {len(predictions)} predictions")

# Add predictions to DataFrame
df['predicted_power'] = predictions
df['actual_power'] = df['Global_active_power']
df['prediction_error'] = df['actual_power'] - df['predicted_power']
df['absolute_error'] = np.abs(df['prediction_error'])

# ============================================================================
# STEP 7: OUTPUT RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 7: PREDICTION RESULTS")
print("="*80)

# Overall statistics
mae = df['absolute_error'].mean()
rmse = np.sqrt((df['prediction_error']**2).mean())
mape = (df['absolute_error'] / df['actual_power'] * 100).mean()

print("\n📊 PREDICTION ACCURACY:")
print(f"  • Mean Absolute Error (MAE): {mae:.3f} kW")
print(f"  • Root Mean Square Error (RMSE): {rmse:.3f} kW")
print(f"  • Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Show sample predictions
print("\n📋 SAMPLE PREDICTIONS (Latest 10):")
print("-" * 90)
print(f"{'Timestamp':<22} {'Actual (kW)':<14} {'Predicted (kW)':<16} {'Error (kW)':<12}")
print("-" * 90)

for idx, row in df.tail(10).iterrows():
    print(f"{row['datetime'].strftime('%Y-%m-%d %H:%M:%S'):<22} "
          f"{row['actual_power']:<14.3f} "
          f"{row['predicted_power']:<16.3f} "
          f"{row['prediction_error']:<12.3f}")

# ============================================================================
# STEP 8: FORECAST NEXT TIME PERIOD
# ============================================================================

print("\n" + "="*80)
print("STEP 8: FORECASTING NEXT HOUR")
print("="*80)

# Use latest data point to forecast next hour
latest = df.iloc[-1]
next_timestamp = latest['datetime'] + timedelta(hours=1)

print(f"\n[Forecasting] Next timestamp: {next_timestamp}")

# Create feature vector for next hour
next_features = {
    'Global_reactive_power': latest['Global_reactive_power'],
    'Voltage': latest['Voltage'],
    'Global_intensity': latest['Global_intensity'],
    'Sub_metering_1': latest['Sub_metering_1'],
    'Sub_metering_2': latest['Sub_metering_2'],
    'Sub_metering_3': latest['Sub_metering_3'],
    'hour': next_timestamp.hour,
    'day_of_week': next_timestamp.dayofweek,
    'month': next_timestamp.month,
    'is_weekend': int(next_timestamp.dayofweek >= 5),
    'power_lag_1h': latest['predicted_power'],
    'power_lag_24h': latest['power_lag_24h'],
    'power_lag_168h': latest['power_lag_168h'],
    'power_ma_24h': latest['power_ma_24h'],
    'power_ma_168h': latest['power_ma_168h']
}

X_next = pd.DataFrame([next_features])[feature_cols]

if hasattr(model, 'n_estimators'):
    X_next_prepared = X_next
else:
    X_next_prepared = scaler.transform(X_next)

forecast = model.predict(X_next_prepared)[0]

print(f"\n🔮 FORECAST:")
print(f"  • Timestamp: {next_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  • Predicted Power: {forecast:.3f} kW")
print(f"  • Based on latest reading: {latest['predicted_power']:.3f} kW")
print(f"  • Change: {(forecast - latest['predicted_power']):.3f} kW")

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 9: SAVING RESULTS")
print("="*80)

os.makedirs('outputs', exist_ok=True)

# Save predictions to CSV
output_file = 'outputs/predictions_mysql.csv'
df[['datetime', 'actual_power', 'predicted_power', 'prediction_error']].to_csv(
    output_file, index=False
)
print(f"✓ Predictions saved: {output_file}")

# Save forecast result
forecast_result = {
    'forecast_timestamp': next_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
    'predicted_power_kw': round(forecast, 3),
    'model_used': type(model).__name__,
    'api_source': 'MySQL',
    'mae': round(mae, 3),
    'rmse': round(rmse, 3),
    'mape_percent': round(mape, 2)
}

forecast_file = 'outputs/forecast_mysql.json'
with open(forecast_file, 'w') as f:
    json.dump(forecast_result, f, indent=2)
print(f"✓ Forecast saved: {forecast_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n\n" + "="*80)
print("PREDICTION PIPELINE COMPLETE ✓")
print("="*80)
print(f"""
✅ COMPLETED STEPS:
  1. Fetched {len(readings)} readings from MySQL API
  2. Preprocessed data (lag features, moving averages)
  3. Loaded trained {type(model).__name__} model
  4. Generated predictions with {mae:.3f} kW MAE
  5. Forecasted next hour: {forecast:.3f} kW

📁 OUTPUT FILES:
  • {output_file}
  • {forecast_file}

🎯 PERFORMANCE:
  • MAE: {mae:.3f} kW
  • RMSE: {rmse:.3f} kW
  • MAPE: {mape:.2f}%
  
🔮 NEXT FORECAST:
  • Time: {next_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
  • Power: {forecast:.3f} kW
""")

print("="*80)

"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                     Sougnabe's Change ending                               ║
║                                                                            ║
║  WHAT WAS CHANGED:                                                         ║
║  ✓ Created complete end-to-end prediction pipeline for MySQL             ║
║  ✓ Step 1: Fetch data from MySQL API                                      ║
║  ✓ Step 2-3: Preprocess with same pipeline as Task 1                      ║
║  ✓ Step 4: Load trained model and scaler                                  ║
║  ✓ Step 5-6: Prepare features and make predictions                        ║
║  ✓ Step 7-8: Evaluate and forecast next period                            ║
║  ✓ Step 9: Save results to CSV and JSON                                   ║
║                                                                            ║
║  IMPACT: Task 4 MySQL version now complete (was empty)                    ║
║          Demonstrates integration with MySQL API                          ║
║          Provides alternative implementation for comparison               ║
║                                                                            ║
║  TO RUN:                                                                   ║
║  1. Start MySQL API: uvicorn mysql_team.task-3_api.api:app --port 5000    ║
║  2. Run: python mysql_team/task-4_prediction/prediction.py                ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
