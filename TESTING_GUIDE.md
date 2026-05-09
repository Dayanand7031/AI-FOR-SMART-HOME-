# Testing Guide for the Improved Smart Home Energy Forecasting Model

## 🚀 How to Test the Model

### Option 1: API Testing (Recommended)
1. Start the FastAPI server:
   ```bash
   python api/main.py
   ```
2. Open your browser and go to: `http://localhost:8000/docs`
3. Use the interactive Swagger UI to test the `/predict` endpoint
4. Example request:
   ```json
   {
     "latitude": 40.7128,
     "longitude": -74.0060,
     "temperature": 25.0,
     "humidity": 60.0,
     "wind_speed": 4.1,
     "pressure": 1012.5,
     "cloud_amount": 15.0,
     "precipitation": 0.0,
     "lag_1": 14.2,
     "lag_7": 12.5,
     "rolling_mean_7": 13.1,
     "day_of_week": 3,
     "month": 6,
     "season": 2
   }
   ```

### Option 2: Streamlit Web Interface
1. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
2. Open your browser and go to: `http://localhost:8501`
3. Fill in the form with your parameters and click "Predict"

### Option 3: Direct Python Testing
```python
import joblib
import numpy as np
from src.data_ingestion import fetch_nasa_data
from src.preprocessing import preprocess_data

# Load model and scaler
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Get and preprocess data (same as training)
df_raw = fetch_nasa_data()
X_train, X_test, y_train, y_test, scaler, processed_df = preprocess_data(df_raw, return_processed=True)

# For prediction, you would use the most recent row's features
# (This is simplified - actual implementation would handle sequencing for DL models)
```

## 📊 Expected Performance
Based on our final evaluation:
- **RMSE**: ~4.93 kWh/m²/day
- **MAE**: ~3.93 kWh/m²/day
- **R² Score**: ~0.58
- **MAPE**: ~38.5% (estimated)

## 🔧 Model Information
- **Best Model**: Stacking Regressor (Random Forest + XGBoost with Ridge meta-learner)
- **Features**: 47 engineered features including:
  - Lag features (1,2,3,7,14,30 days)
  - Lag differences
  - Rolling statistics (mean, std, min, max for various windows)
  - Exponentially weighted moving averages
  - Time-based features (day of week, month, day of year, week of year)
  - Cyclical encoding (sin/cos for day of week and month)
  - Weather interaction features (temp×humidity, wind×solar, pressure×temp, wind×cloud)
  - Feature derivatives (temperature, humidity, pressure, wind speed changes)
  - Season one-hot encoding

## 📁 Files
- `models/best_model.pkl` - Trained stacking regressor
- `models/scaler.pkl` - StandardScaler for feature preprocessing
- `training_log.csv` - Complete training metrics
- `data/processed/processed_energy_data.csv` - Processed features for inspection

## ⚠️ Notes
1. The model predicts solar radiation as a proxy for home energy consumption
2. For actual home energy consumption, you would need to replace the target variable
3. The model works best for patterns similar to the training data (New York City area)
4. For different locations, you may need to retrain with location-specific data