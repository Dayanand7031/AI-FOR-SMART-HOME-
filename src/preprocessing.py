import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from config import PROCESSED_DATA_DIR, SCALER_PATH, TRAIN_TEST_SPLIT, RANDOM_STATE

def preprocess_data(df, remove_outliers=False):
    """
    Cleans and engineers features for energy forecasting.

    Args:
        df (pd.DataFrame): Raw dataframe from data_ingestion
        remove_outliers (bool): Whether to remove outliers using IQR method

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Ensure date is datetime and sorted
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Replace NASA fill values (-999.0) with NaN
    df = df.replace(-999.0, np.nan)

    # 1. Handle Missing Values
    # Use interpolation for better gap filling, then forward/backward fill for any remaining
    df = df.interpolate(method='linear', limit_direction='both')
    # If any missing values remain (e.g., at the edges), fill with forward/backward
    df = df.ffill().bfill()

    # 2. Target Variable
    # Target: Solar radiation shifted by -1 (predict next day)
    # We use ALLSKY_SFC_SW_DWN as the energy consumption proxy
    df['target'] = df['ALLSKY_SFC_SW_DWN'].shift(-1)

    # 3. Feature Engineering
    # Lag features: Previous days' solar radiation
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'lag_{lag}'] = df['ALLSKY_SFC_SW_DWN'].shift(lag)

    # Advanced lag features: differences
    df['lag_diff_1'] = df['ALLSKY_SFC_SW_DWN'].diff(1)
    df['lag_diff_7'] = df['ALLSKY_SFC_SW_DWN'].diff(7)
    # Momentum features
    df['momentum_3'] = df['ALLSKY_SFC_SW_DWN'] - df['ALLSKY_SFC_SW_DWN'].shift(3)
    df['momentum_7'] = df['ALLSKY_SFC_SW_DWN'] - df['ALLSKY_SFC_SW_DWN'].shift(7)

    # Rolling statistics
    df['rolling_mean_3'] = df['ALLSKY_SFC_SW_DWN'].rolling(window=3).mean()
    df['rolling_mean_7'] = df['ALLSKY_SFC_SW_DWN'].rolling(window=7).mean()
    df['rolling_mean_14'] = df['ALLSKY_SFC_SW_DWN'].rolling(window=14).mean()
    df['rolling_std_3'] = df['ALLSKY_SFC_SW_DWN'].rolling(window=3).std()
    df['rolling_std_7'] = df['ALLSKY_SFC_SW_DWN'].rolling(window=7).std()
    df['rolling_std_14'] = df['ALLSKY_SFC_SW_DWN'].rolling(window=14).std()
    df['rolling_min_7'] = df['ALLSKY_SFC_SW_DWN'].rolling(window=7).min()
    df['rolling_max_7'] = df['ALLSKY_SFC_SW_DWN'].rolling(window=7).max()

    # Exponentially weighted moving averages
    df['ewm_alpha_0_3'] = df['ALLSKY_SFC_SW_DWN'].ewm(alpha=0.3).mean()
    df['ewm_alpha_0_7'] = df['ALLSKY_SFC_SW_DWN'].ewm(alpha=0.7).mean()

    # Time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week

    # Cyclical encoding for temporal features (helps models learn periodic patterns)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Weather interaction features
    df['temp_humidity'] = df['T2M'] * df['RH2M'] / 100
    df['wind_solar'] = df['WS10M'] * df['ALLSKY_SFC_SW_DWN']
    df['pressure_temp'] = df['PS'] * df['T2M']
    df['wind_cloud'] = df['WS10M'] * df['CLOUD_AMT']

    # Feature derivatives (changes) - helps capture sudden spikes and drops
    df['temp_change'] = df['T2M'].diff()
    df['humidity_change'] = df['RH2M'].diff()
    df['pressure_change'] = df['PS'].diff()
    df['wind_speed_change'] = df['WS10M'].diff()

    # Season mapping
    def get_season(month):
        if month in [12, 1, 2]: return 0 # Winter
        if month in [3, 4, 5]: return 1  # Spring
        if month in [6, 7, 8]: return 2  # Summer
        return 3                        # Autumn

    df['season'] = df['month'].apply(get_season)
    # One-hot encode season
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)
    # Drop the original season column
    df = df.drop('season', axis=1)

    # 4. Outlier Detection and Treatment (optional)
    if remove_outliers:
        # Remove extreme outliers using IQR on the target variable (before shifting)
        # We'll use the original solar radiation for outlier detection
        Q1 = df['ALLSKY_SFC_SW_DWN'].quantile(0.25)
        Q3 = df['ALLSKY_SFC_SW_DWN'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_mask = (df['ALLSKY_SFC_SW_DWN'] >= lower_bound) & (df['ALLSKY_SFC_SW_DWN'] <= upper_bound)
        df = df[outlier_mask]
        # Reset index after filtering
        df = df.reset_index(drop=True)

    # 5. Drop rows with NaN values created by shifting/rolling
    df = df.dropna()

    # 6. Prepare features and target
    # Remove the date and the original target (which is now the shift) from features
    feature_cols = [col for col in df.columns if col not in ['date', 'target']]
    # We also remove ALLSKY_SFC_SW_DWN because it's the target for the next day
    # (though the lag features already contain its info)
    feature_cols = [col for col in feature_cols if col != 'ALLSKY_SFC_SW_DWN']
    X = df[feature_cols]
    y = df['target']

    # 7. Chronological Train-Test Split (No Shuffle)
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # 8. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)

    # Save processed data for reference
    processed_df = df.copy()
    processed_df.to_csv(PROCESSED_DATA_DIR / "processed_energy_data.csv", index=False)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    # Integration test with a sample dataframe if raw data is not present
    try:
        from src.data_ingestion import fetch_nasa_data
        df_raw = fetch_nasa_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df_raw, remove_outliers=False)
        print("Preprocessing complete. Scaler saved.")
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print(f"Number of features: {X_train.shape[1]}")
    except Exception as e:
        print(f"Error during preprocessing test: {e}")