import numpy as np
import pandas as pd
import joblib
from config import BEST_MODEL_PATH, SCALER_PATH

def predict_energy(input_data):
    """
    Takes a dictionary of feature values, scales them, and predicts
    next-day energy consumption.

    Args:
        input_data (dict): Dictionary containing basic features from API request

    Returns:
        dict: Prediction result including value and model info
    """
    # 1. Load model and scaler
    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2. Create a complete feature set by estimating missing features
    # Start with the provided input data
    features = {}

    # Map API parameter names to technical feature names used in the model
    features['T2M'] = input_data.get('temperature', 25.0)      # temperature -> T2M
    features['RH2M'] = input_data.get('humidity', 60.0)        # humidity -> RH2M
    features['WS10M'] = input_data.get('wind_speed', 5.0)      # wind_speed -> WS10M
    features['PS'] = input_data.get('pressure', 1013.0)        # pressure -> PS
    features['CLOUD_AMT'] = input_data.get('cloud_amount', 20.0)  # cloud_amount -> CLOUD_AMT
    features['PRECTOTCORR'] = input_data.get('precipitation', 0.0)  # precipitation -> PRECTOTCORR
    features['lag_1'] = input_data.get('lag_1', 15.0)          # lag_1 (same name)
    features['lag_7'] = input_data.get('lag_7', 12.0)          # lag_7 (same name)
    features['rolling_mean_7'] = input_data.get('rolling_mean_7', 13.0)  # rolling_mean_7 (same name)
    features['day_of_week'] = input_data.get('day_of_week', 3)   # day_of_week (same name)
    features['month'] = input_data.get('month', 6)             # month (same name)
    features['season'] = input_data.get('season', 2)           # season (same name, will convert to one-hot later)

    # Extract provided values for convenience (model uses technical feature names)
    T2M = features['T2M']  # temperature -> T2M
    RH2M = features['RH2M']  # humidity -> RH2M
    WS10M = features['WS10M']  # wind_speed -> WS10M
    PS = features['PS']  # pressure -> PS
    CLOUD_AMT = features['CLOUD_AMT']  # cloud_amount -> CLOUD_AMT
    PRECTOTCORR = features['PRECTOTCORR']  # precipitation -> PRECTOTCORR
    lag_1 = features['lag_1']  # yesterday's solar radiation
    lag_7 = features['lag_7']  # solar radiation 7 days ago
    rolling_mean_7 = features['rolling_mean_7']  # 7-day avg solar radiation
    day_of_week = features['day_of_week']  # 0-6
    month = features['month']  # 1-12
    season = features['season']  # 0-3

    # 3. Estimate missing lag features
    # We don't have lag_2, lag_3, lag_14, lag_30, so we'll estimate them
    # Simple approach: interpolate/extrapolate from available lags
    features['lag_2'] = lag_1 * 0.9 + lag_7 * 0.1  # Weight recent more heavily
    features['lag_3'] = lag_1 * 0.8 + lag_7 * 0.2
    features['lag_14'] = lag_1 * 0.3 + lag_7 * 0.7  # Further back, weight 7-day more
    features['lag_30'] = lag_7  # Best estimate we have for 30 days ago

    # 4. Compute lag differences (need today's value for true diff, estimate)
    # We don't have today's ALLSKY_SFC_SW_DWN, so we'll estimate it
    # Simple estimate: assume today is similar to recent average
    estimated_today = rolling_mean_7
    features['lag_diff_1'] = estimated_today - lag_1  # today - yesterday
    features['lag_diff_7'] = estimated_today - lag_7  # today - 7 days ago

    # 5. Compute momentum features
    features['momentum_3'] = estimated_today - features['lag_3']  # today - 3 days ago
    features['momentum_7'] = estimated_today - lag_7  # today - 7 days ago

    # 6. Estimate rolling statistics
    # We don't have enough history for true rolling stats, so we'll estimate
    # Based on the values we have: lag_1, lag_7, and rolling_mean_7
    features['rolling_mean_3'] = (lag_1 + features['lag_2'] + features['lag_3']) / 3
    features['rolling_mean_14'] = rolling_mean_7  # Reasonable approximation

    # For std, min, max - we need to estimate variability
    # Use the spread between our lag values as a proxy for volatility
    lag_values = [lag_1, features['lag_2'], features['lag_3'], lag_7]
    lag_std = np.std(lag_values) if len(lag_values) > 1 else 1.0
    lag_min = np.min(lag_values)
    lag_max = np.max(lag_values)

    features['rolling_std_3'] = lag_std * 0.8  # Slightly less for shorter window
    features['rolling_std_7'] = lag_std
    features['rolling_std_14'] = lag_std * 1.2  # Slightly more for longer window

    features['rolling_min_7'] = lag_min
    features['rolling_max_7'] = lag_max

    # 7. Estimate exponentially weighted moving averages
    # These give more weight to recent values
    features['ewm_alpha_0_3'] = estimated_today * 0.7 + lag_1 * 0.3  # More weight to recent
    features['ewm_alpha_0_7'] = estimated_today * 0.5 + lag_1 * 0.5  # Equal weight

    # 8. Compute cyclical encoding (we have the inputs!)
    features['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    features['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)

    # 9. Compute weather interaction features (we have the inputs!)
    features['temp_humidity'] = T2M * RH2M / 100
    features['wind_solar'] = WS10M * estimated_today  # Need solar radiation estimate
    features['pressure_temp'] = PS * T2M
    features['wind_cloud'] = WS10M * CLOUD_AMT

    # 10. Estimate change features (we don't have yesterday's weather)
    # We'll use simplified estimates or assume minimal change
    features['temp_change'] = 0.0  # Assume no big change from yesterday
    features['humidity_change'] = 0.0
    features['pressure_change'] = 0.0
    features['wind_speed_change'] = 0.0

    # 11. Compute season one-hot encoding
    features['season_0'] = 1.0 if season == 0 else 0.0
    features['season_1'] = 1.0 if season == 1 else 0.0
    features['season_2'] = 1.0 if season == 2 else 0.0
    features['season_3'] = 1.0 if season == 3 else 0.0

    # 12. Estimate day_of_year and week_of_year
    # Without the actual date, we'll estimate based on month
    # Rough approximation: day_of_year = (month-1)*30 + 15 (middle of month)
    features['day_of_year'] = (month - 1) * 30 + 15
    features['week_of_year'] = int(features['day_of_year'] / 7)

    # 13. Compute is_weekend
    features['is_weekend'] = 1.0 if day_of_week >= 5 else 0.0

    # 14. Add the ALLSKY_SFC_SW_DWN column (needed for feature alignment but will be ignored)
    # We'll use our estimated today value
    features['ALLSKY_SFC_SW_DWN'] = estimated_today

    # 15. Convert to DataFrame and ensure correct column order
    df_input = pd.DataFrame([features])

    # Get the expected feature order from processed data
    processed_df = pd.read_csv("data/processed/processed_energy_data.csv")
    feature_cols = [col for col in processed_df.columns if col not in ['date', 'target']]

    # IMPORTANT: Remove ALLSKY_SFC_SW_DWN from features (it's the target, not a feature)
    # This matches what was done in preprocessing.py during training
    if 'ALLSKY_SFC_SW_DWN' in feature_cols:
        feature_cols.remove('ALLSKY_SFC_SW_DWN')

    # Ensure we have all required columns in the correct order
    # Add any missing columns with default values
    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = 0.0  # Default value for missing features

    # Reorder columns to match training data
    df_input = df_input[feature_cols]

    # 16. Scale the input
    X_scaled = scaler.transform(df_input)

    # 17. Handle DL vs ML predictions
    is_keras_model = hasattr(model, 'predict') and hasattr(model, 'layers')

    if is_keras_model:
        # For DL models, we need to create a sequence
        # Since we don't have historical sequences in API, we'll approximate
        # by repeating the current features (not ideal but functional)
        X_scaled_seq = np.repeat(X_scaled, 7, axis=0).reshape(1, 7, -1)
        prediction = model.predict(X_scaled_seq).flatten()[0]
    else:
        prediction = model.predict(X_scaled).flatten()[0]

    # 18. Determine confidence
    confidence = "medium"
    if not is_keras_model:
        confidence = "high"  # Tree models often more stable for tabular time-series

    # Debug: print the features we're about to use
    # print(f"DEBUG: Features shape: {df_input.shape}, columns: {list(df_input.columns)}")

    return {
        "predicted_energy_consumption": float(prediction),
        "unit": "kWh/m²/day",
        "confidence": confidence,
        "model_used": type(model).__name__
    }

if __name__ == "__main__":
    # Sample prediction test
    # These values are mock-ups; in a real app, these come from the API request
    sample_input = {
        "T2M": 25.0,
        "RH2M": 60.0,
        "WS10M": 5.0,
        "PRECTOTCORR": 0.1,
        "PS": 1013.0,
        "CLOUD_AMT": 20.0,
        "ALLSKY_SFC_SW_DWN": 15.0,
        "lag_1": 14.0,
        "lag_7": 12.0,
        "lag_30": 10.0,
        "rolling_mean_7": 13.0,
        "rolling_std_7": 1.5,
        "day_of_week": 3,
        "month": 6,
        "is_weekend": 0,
        "season": 2
    }

    try:
        res = predict_energy(sample_input)
        print(f"Prediction Result: {res}")
    except Exception as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print("Error: Model or Scaler files not found. Please run train.py first.")
