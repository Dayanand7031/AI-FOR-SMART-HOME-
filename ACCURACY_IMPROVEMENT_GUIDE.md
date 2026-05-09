# Guide to Improving Accuracy for AI Smart Home Energy Forecasting

Current performance metrics from `training_log.csv`:
- **Best Model**: Random Forest
- **RMSE**: 3.91 kWh/m²/day
- **MAE**: 3.06 kWh/m²/day
- **MAPE**: 39.99% 
- **R² Score**: 0.56

For energy forecasting applications, a MAPE under 20% is generally considered good, and under 10% is excellent. The current MAPE of ~40% indicates significant room for improvement.

## 🔍 Potential Causes of Suboptimal Accuracy

### 1. **Data & Target Variable Issues**
- **Proxy Variable**: Using `ALLSKY_SFC_SW_DWN` (surface downward shortwave radiation) as a direct proxy for home energy consumption may not capture all relevant factors
- **Missing Features**: Important factors like household occupancy, appliance usage patterns, insulation quality, or electricity pricing aren't included
- **Location Specificity**: Model trained on NYC data may not generalize well to other locations without location-specific features

### 2. **Feature Engineering Limitations**
- **Lag Features**: Only basic lag features (1, 7, 30 days) - could benefit from more sophisticated temporal features
- **Rolling Statistics**: Only mean and std - could add min, max, percentiles, etc.
- **Weather Interactions**: No interaction features between weather variables (e.g., temperature × humidity)
- **Seasonal Decomposition**: No explicit seasonal/trend/residual decomposition

### 3. **Modeling Approach**
- **Hyperparameters**: All models use default parameters (n_estimators=100 for tree models)
- **Ensemble Methods**: No stacking or blending of multiple models
- **Deep Learning**: DL models weren't trained (likely due to missing TensorFlow)
- **Cross-Validation**: TimeSeriesSplit is used but only last split utilized for final training

### 4. **Data Quality & Quantity**
- **Training Period**: Data from 2021-01-01 to 2026-05-08 (~5.4 years) - reasonable but could be extended
- **Missing Value Handling**: Simple forward/backward fill might not be optimal for all gaps
- **Outlier Treatment**: No outlier detection or handling

## 🚀 Actionable Improvement Strategies

### 📊 1. Enhanced Feature Engineering

#### Temporal Features
```python
# More sophisticated time features
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

# Cyclical encoding for better temporal representation
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

#### Weather Feature Interactions
```python
# Interaction features
df['temp_humidity'] = df['T2M'] * df['RH2M'] / 100
df['wind_solar'] = df['WS10M'] * df['ALLSKY_SFC_SW_DWN']
df['pressure_temp'] = df['PS'] * df['T2M']
```

#### Advanced Lag Features
```python
# More lag features and differences
for lag in [1, 2, 3, 7, 14, 30]:
    df[f'lag_{lag}'] = df['ALLSKY_SFC_SW_DWN'].shift(lag)
    df[f'lag_diff_{lag}'] = df['ALLSKY_SFC_SW_DWN'].diff(lag)

# Exponentially weighted moving averages
df['ewm_alpha_0_3'] = df['ALLSKY_SFC_SW_DWN'].ewm(alpha=0.3).mean()
df['ewm_alpha_0_7'] = df['ALLSKY_SFC_SW_DWN'].ewm(alpha=0.7).mean()
```

### 🔧 2. Improved Data Preprocessing

#### Better Missing Value Handling
```python
# Interpolation instead of just fill
df = df.interpolate(method='linear', limit_direction='both')

# Or more sophisticated methods for gaps
# df = df.fillna(method='ffill').fillna(method='bfill')
```

#### Outlier Detection & Treatment
```python
# Remove extreme outliers using IQR
Q1 = df['ALLSKY_SFC_SW_DWN'].quantile(0.25)
Q3 = df['ALLSKY_SFC_SW_DWN'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['ALLSKY_SFC_SW_DWN'] >= lower_bound) & (df['ALLSKY_SFC_SW_DWN'] <= upper_bound)]
```

### 🤖 3. Model Enhancements

#### Hyperparameter Tuning
```python
# Example for Random Forest
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestRegressor(random_state=RANDOM_STATE)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
```

#### Advanced Ensemble Techniques
```python
# Voting Regressor
from sklearn.ensemble import VotingRegressor

ensemble = VotingRegressor([
    ('rf', RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)),
    ('xgb', XGBRegressor(n_estimators=200, random_state=RANDOM_STATE)),
    ('lgbm', LGBMRegressor(n_estimators=200, random_state=RANDOM_STATE))
])

# Stacking Regressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
    ('xgb', XGBRegressor(n_estimators=100, random_state=RANDOM_STATE))
]
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(),
    cv=5
)
```

#### Enable Deep Learning Models
First, ensure TensorFlow is installed:
```bash
pip install tensorflow
```

Then improve DL architecture:
```python
def build_improved_lstm(input_shape):
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
```

### 📈 4. Validation & Training Improvements

#### Proper Time Series Cross-Validation
```python
# Use all splits for better validation
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for train_idx, val_idx in tscv.split(X):
    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train and evaluate model
    model.fit(X_train_cv, y_train_cv)
    preds = model.predict(X_val_cv)
    mae = mean_absolute_error(y_val_cv, preds)
    cv_scores.append(mae)

print(f"CV MAE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
```

#### Use More Data for Training
Consider increasing `TRAIN_TEST_SPLIT` to 0.85 or 0.9 if you have sufficient data.

### 🌍 5. Location-Specific Improvements

#### Add Location Features
If expanding to multiple locations:
```python
# One-hot encode location or use lat/lon as features
df['latitude'] = DEFAULT_LAT  # Or actual latitude if multiple locations
df['longitude'] = DEFAULT_LON

# Or create location clusters
from sklearn.cluster import KMeans
# Train on multiple locations, then predict per cluster
```

#### Incorporate Local Characteristics
- Elevation
- Urban/rural classification
- Average household size
- Local energy policies/incentives

### 📋 6. Problem Formulation Improvements

#### Different Forecast Horizons
Instead of just next-day, try:
- Multi-step forecasting (predict next 3, 7 days)
- Sequence-to-sequence models

#### Different Target Variables
- Predict actual electricity consumption if available
- Predict net energy (production - consumption)
- Predict peak demand rather than total daily energy

#### Uncertainty Quantification
Instead of point predictions, predict intervals:
```python
# Quantile Regression Forests
# Or use Bayesian approaches
# Or predict mean and variance
```

## 🔧 Implementation Plan

### Phase 1: Quick Wins (1-2 days)
1. [ ] Install TensorFlow: `pip install tensorflow`
2. [ ] Add cyclical time features
3. [ ] Add weather interaction features
4. [ ] Implement hyperparameter tuning for tree models
5. [ ] Enable and train DL models

### Phase 2: Medium-term Improvements (3-5 days)
1. [ ] Implement advanced lag features (more lags, differences)
2. [ ] Add outlier detection and treatment
3. [ ] Implement proper time series cross-validation
4. [ ] Try ensemble methods (voting, stacking)
5. [ ] Improve missing value handling with interpolation

### Phase 3: Advanced Improvements (1-2 weeks)
1. [ ] Collect multi-location data for more robust models
2. [ ] Add location-specific features
3. [ ] Experiment with different forecast horizons
4. [ ] Implement uncertainty quantification
5. [ ] Feature importance analysis and selection

## 📊 Expected Improvements

With these enhancements, realistic targets might be:
- **MAPE**: 20-25% (good) → potentially 15-20% (very good)
- **R² Score**: 0.56 → potentially 0.70-0.80
- **RMSE**: 3.91 → potentially 2.5-3.0 kWh/m²/day

## 📋 Monitoring & Validation

### Track These Metrics
1. **MAPE** (Primary business metric)
2. **RMSE** (For error magnitude)
3. **MAE** (For robustness to outliers)
4. **R²** (For explained variance)
5. **Prediction Interval Coverage** (If implementing uncertainty)

### Validation Strategy
1. **Hold-out Test Set**: Keep 20% of most recent data for final testing
2. **Time-Based Validation**: Ensure no future data leaks into training
3. **Geographic Validation** (if multi-location): Test on held-out locations
4. **Seasonal Validation**: Test performance across different seasons

## 🛠️ Code Modification Summary

### Files to Modify:
1. `src/preprocessing.py` - Enhanced feature engineering
2. `src/train.py` - Hyperparameter tuning, ensembles, DL training
3. `src/config.py` - Add new hyperparameters and feature flags
4. `src/predict.py` - Ensure compatibility with new features

### Example Enhancement to Preprocessing:
```python
# In src/preprocessing.py, enhance the feature engineering section:

# ... existing code ...

# 3. Feature Engineering (ENHANCED)
# Lag features: Previous days' solar radiation
for lag in [1, 2, 3, 7, 14, 30]:
    df[f'lag_{lag}'] = df['ALLSKY_SFC_SW_DWN'].shift(lag)

# Advanced lag features
df['lag_diff_1'] = df['ALLSKY_SFC_SW_DWN'].diff(1)
df['lag_diff_7'] = df['ALLSKY_SFC_SW_DWN'].diff(7)

# Rolling statistics
df['rolling_mean_7'] = df['ALLSKY_SFC_SW_DWN'].rolling(window=7).mean()
df['rolling_std_7'] = df['ALLSKY_SFC_SW_DWN'].rolling(window=7).std()
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

# Cyclical encoding
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Weather interactions
df['temp_humidity'] = df['T2M'] * df['RH2M'] / 100
df['wind_solar'] = df['WS10M'] * df['ALLSKY_SFC_SW_DWN']
df['pressure_temp'] = df['PS'] * df['T2M']

# ... rest of existing code ...
```

## ✅ Verification Steps

After implementing improvements:
1. Re-run training: `python src/train.py`
2. Check new `training_log.csv` for improved metrics
3. Run evaluation: `python src/evaluate.py`
4. Compare plots in `outputs/` directory
5. Test API: `python api/main.py` then try predictions
6. Test Streamlit UI: `streamlit run streamlit_app.py`

## 💡 Additional Tips

1. **Start Simple**: Implement one improvement at a time and measure impact
2. **Keep Baselines**: Save previous model versions to compare performance
3. **Document Changes**: Note what you changed and why
4. **Consider Interpretability**: Use SHAP values to understand feature importance
5. **Monitor for Overfitting**: Watch for train/test performance divergence

## 📚 Resources for Further Learning

- **Time Series Forecasting**: "Forecasting: Principles and Practice" by Hyndman & Athanasopoulos
- **Feature Engineering for ML**: Kaggle courses and Featuretools library
- **Hyperparameter Tuning**: Optuna, Hyperopt, or Scikit-learn's GridSearchCV/RandomizedSearchCV
- **Energy Forecasting Specific**: Research papers on solar/load forecasting using ML

---

**Remember**: The key to improving accuracy is systematic experimentation. Make one change at a time, measure the impact, and iterate. Even small improvements in multiple areas can compound to significant gains in overall model performance.