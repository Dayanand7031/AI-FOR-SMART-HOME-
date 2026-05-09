# AI for Smart Home — Energy Forecasting System
## Interview Explanation Guide

### 🎯 Project Overview
This is an end-to-end machine learning system designed to forecast daily energy consumption (specifically solar radiation) using NASA POWER API climate data. The system predicts next-day energy production to help smart homes optimize energy usage, storage, and grid interaction.

### 🏗️ System Architecture
The project follows a modular architecture with four main layers:

1. **Data Layer** - NASA POWER API ingestion and storage
2. **Processing Layer** - Data cleaning, feature engineering, and preparation
3. **Model Layer** - ML/DL model training, evaluation, and selection
4. **Deployment Layer** - REST API (FastAPI) and user interface (Streamlit)

### 🔧 Technical Implementation

#### 1. Data Ingestion (`src/data_ingestion.py`)
- Fetches climate data from NASA POWER API
- Parameters collected: 
  - `ALLSKY_SFC_SW_DWN` (Solar Radiation - target variable)
  - `T2M` (Temperature at 2m)
  - `RH2M` (Relative Humidity)
  - `WS10M` (Wind Speed at 10m)
  - `PRECTOTCORR` (Precipitation)
  - `PS` (Surface Pressure)
  - `CLOUD_AMT` (Cloud Amount)
- Stores raw data as CSV in `data/raw/`

#### 2. Preprocessing (`src/preprocessing.py`)
- Handles missing values using interpolation
- Feature engineering:
  - Lag features: `lag_1` (1 day), `lag_7` (7 days), `lag_30` (30 days)
  - Rolling statistics: `rolling_mean_7`, `rolling_std_7` (7-day window)
  - Temporal features: `day_of_week`, `month`, `season`, `is_weekend`
- Saves processed data to `data/processed/`

#### 3. Model Training (`src/train.py`)
- Compares 6 algorithms:
  - **ML Models**: XGBoost, Random Forest, LightGBM
  - **Deep Learning Models**: LSTM, GRU, CNN-LSTM
- Automatic model selection based on lowest RMSE
- Features scaling using StandardScaler
- Saves best model to `models/best_model.pkl`
- Logs all metrics to `training_log.csv`

#### 4. Model Prediction (`src/predict.py`)
- Loads trained model and scaler
- Handles both ML (2D input) and DL (3D sequence input) models
- For DL models: creates sequence from single input by repeating (simulation)
- Returns prediction with confidence and model metadata

#### 5. API Layer (`api/`)
- **FastAPI** framework for high-performance async API
- **Endpoints**:
  - `POST /predict` - Main prediction endpoint
  - `GET /health` - API health check
  - `GET /model-info` - Deployed model metadata
- **Pydantic models** for request/response validation
- **CORS middleware** for frontend integration
- Automatic Swagger/OpenAPI docs at `/docs`

#### 6. Frontend (`streamlit_app.py`)
- **Streamlit** based interactive UI
- Features:
  - Form-based input for all required features
  - Real-time API health checking
  - Prediction visualization with metrics
  - Model info sidebar showing training metrics
  - Expandable sections for request/response details
  - Responsive design

### 💾 Key Files & Structure
```
AI FOR SMART HOME/
├── src/                 # Core ML logic
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── api/                 # FastAPI backend
│   ├── main.py
│   ├── schemas.py
│   └── router.py
├── data/                # Data storage
│   ├── raw/
│   └── processed/
├── models/              # Saved models
│   └── best_model.pkl
├── streamlit_app.py     # Frontend UI
├── config.py            # Configuration constants
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

### ⚙️ Technologies Used
- **Programming Language**: Python 3.x
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Deep Learning**: TensorFlow/Keras
- **API Framework**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Utilities**: Joblib (model serialization), Requests (API calls)
- **Visualization**: Matplotlib, Seaborn (in evaluation)

### 🔑 Key Features & Innovations
1. **End-to-End Pipeline**: From raw API data to deployable API with zero manual intervention
2. **Model Agnostic Design**: Easy to add/remove algorithms without changing core logic
3. **Hybrid ML/DL Support**: Handles both traditional ML and sequence-based DL models
4. **Automatic Feature Engineering**: Lag features, rolling statistics, temporal encoding
5. **Production-Ready API**: Async FastAPI with proper error handling, validation, and docs
6. **User-Friendly Interface**: Streamlit frontend for non-technical stakeholders
7. **Configuration Management**: Centralized config.py for easy parameter tuning
8. **Comprehensive Logging**: Training metrics tracked for model comparison and reproducibility

### 📊 Model Performance & Evaluation
- **Metrics Tracked**: RMSE, MAE, MAPE, R² Score
- **Best Model Selection**: Automatic based on lowest validation RMSE
- **Visualization**: Actual vs predicted plots, feature importance (in evaluate.py)
- **Experiment Tracking**: training_log.csv records all model runs

### 🚀 Deployment & Usage
1. **Data Pipeline**:
   ```bash
   python src/data_ingestion.py   # Fetch data
   python src/preprocessing.py    # Process data
   python src/train.py            # Train & select best model
   ```
2. **API Deployment**:
   ```bash
   python api/main.py             # Starts API on http://localhost:8000
   ```
3. **Frontend**:
   ```bash
   streamlit run streamlit_app.py # Launches UI on http://localhost:8501
   ```
4. **Direct API Usage**:
   ```bash
   curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{...}'
   ```

### 💡 Smart Home Applications
- **Energy Optimization**: Schedule high-consumption appliances during predicted high-production periods
- **Battery Management**: Optimize charging/discharging cycles based on forecast
- **Grid Interaction**: Better predict energy export/import to utility grid
- **Cost Savings**: Reduce reliance on grid during peak pricing hours
- **Sustainability**: Maximize self-consumption of solar energy

### 🔧 Challenges & Solutions
1. **Data Variability**: NASA API sometimes returns missing values → Used interpolation and robust feature engineering
2. **Model Selection**: Needed objective comparison → Automated training and evaluation of multiple algorithms
3. **API-Model Interface**: Different input formats for ML vs DL models → Created unified predict.py handling both
4. **Deployment Complexity**: Wanted simple deployment → Used FastAPI + Streamlit for minimal setup
5. **Real-world Usability**: Needed non-technical interface → Built Streamlit frontend with intuitive controls

### 📈 Future Enhancements
1. **Real-time Data Integration**: WebSocket connections for live data updates
2. **Uncertainty Quantification**: Prediction intervals using conformal prediction or Bayesian methods
3. **Multi-location Support**: Geographic scaling for regional energy forecasting
4. **Explainability**: SHAP/LIME integration for model interpretability
5. **Containerization**: Docker deployment for easier scaling
6. **Monitoring**: API metrics logging and drift detection
7. **Extended Forecast**: Multi-day ahead predictions (3-day, 7-day forecasts)

### 🎓 Why This Project Matters
This system demonstrates full-stack ML engineering skills:
- **Data Engineering**: API integration, ETL pipelines, feature engineering
- **Machine Learning**: Algorithm selection, hyperparameter tuning, model evaluation
- **Software Engineering**: Modular design, API development, frontend creation
- **Deployment**: Model serving, API documentation, production considerations
- **Problem Solving**: End-to-end thinking from data collection to user value

The project bridges the gap between theoretical ML knowledge and practical deployment skills that are valuable in industry roles.





Made Improvement 
Changes Made

  src/preprocessing.py Enhancements:

  1. Advanced Feature Engineering:
    - More lag features (1,2,3,7,14,30 days)
    - Lag differences (1-day, 7-day)
    - Enhanced rolling statistics (mean, std, min, max)
    - Exponentially weighted moving averages (α=0.3, 0.7)
    - Cyclical time encoding (sin/cos for day-of-week, month)
    - Weather interaction features (temp×humidity, wind×solar, pressure×temp)
    - Day-of-year and week-of-year features
  2. Improved Data Handling:
    - Interpolation for missing values (better than simple fill)
    - Optional outlier removal using IQR method
    - Chronological train/test split maintained

  src/train.py Enhancements:

  1. Hyperparameter Tuning:
    - GridSearchCV for Random Forest (n_estimators, max_depth, etc.)
    - TimeSeriesSplit cross-validation for tuning
    - Example focused on RF (extendable to XGBoost/LightGBM)
  2. Model Expansion:
    - Enabled Deep Learning models (LSTM, GRU, CNN-LSTM) with improved architecture
    - Added Dropout layers and increased units for better capacity
    - Ensemble methods: VotingRegressor and StackingRegressor
  3. Improved Training Workflow:
    - Proper use of all TimeSeriesSplit crosses for validation
    - Clear separation: tuning → final training → test evaluation
    - Enhanced logging showing all model types

  🚀 Phased Implementation Plan

  Phase 1: Quick Wins (1-2 Days)

  Focus: Enable DL models and basic feature enhancements
  1. Install TensorFlow:
  pip install tensorflow
  2. Enable basic feature enhancements:
    - Start with cyclical time features and additional lags
    - In preprocessing.py, uncomment/add:
  # Cyclical encoding
  df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
  df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
  # Additional lags
  for lag in [2, 3, 14]:
      df[f'lag_{lag}'] = df['ALLSKY_SFC_SW_DWN'].shift(lag)
  3. Run training to verify DL models work:
  python src/train.py
  4. Expected improvement: 5-10% MAPE reduction from DL models capturing temporal patterns

  Phase 2: Medium-term Improvements (3-5 Days)

  Focus: Hyperparameter tuning and weather interactions
  1. Implement hyperparameter tuning for tree models:
    - The updated train.py already includes RF tuning
    - Extend to XGBoost/LightGBM by adding similar GridSearchCV blocks
    - Example XGBoost parameters to tune:
  param_grid = {
      'n_estimators': [100, 200],
      'max_depth': [3, 6, 9],
      'learning_rate': [0.01, 0.1, 0.2],
      'subsample': [0.8, 0.9, 1.0]
  }
  2. Add weather interaction features:
    - Uncomment in preprocessing.py:
  df['temp_humidity'] = df['T2M'] * df['RH2M'] / 100
  df['wind_solar'] = df['WS10M'] * df['ALLSKY_SFC_SW_DWN']
  df['pressure_temp'] = df['PS'] * df['T2M']
  3. Run full training with tuning:
  python src/train.py
  4. Expected improvement: Additional 10-15% MAPE reduction from better-optimized models

  Phase 3: Advanced Improvements (1-2 Weeks)

  Focus: Ensemble methods, outlier handling, and validation
  1. Implement outlier detection:
    - In preprocess_data() call, set remove_outliers=True
    - Monitor impact - may help or hurt depending on data distribution
  2. Validate ensemble methods:
    - The updated train.py already includes Voting and Stacking ensembles
    - Compare performance of individual models vs. ensembles
  3. Experiment with feature selection:
    - Use SelectFromModel or RFE to identify most important features
    - Remove redundant features to reduce overfitting
  4. Run final comprehensive training:
  python src/train.py
  5. Expected improvement: Final 5-10% MAPE reduction from robustness and ensemble effects

  📊 Expected Results

  After completing all phases, realistic targets:
  - MAPE: 39.99% → 18-25% (good to very good range)
  - R² Score: 0.56 → 0.65-0.75
  - RMSE: 3.91 → 2.5-3.2 kWh/m²/day

  🔧 Verification Steps

  After each phase:
  1. Check updated training_log.csv for improved metrics
  2. Run evaluation: python src/evaluate.py
  3. Compare plots in outputs/ directory
  4. Test API: python api/main.py then try predictions
  5. Test Streamlit UI: streamlit run streamlit_app.py

  💡 Key Implementation Notes

  1. Start small: Implement one change at a time and measure impact
  2. Keep baselines: Save previous model versions to compare performance
  3. Watch for overfitting: Monitor train vs. validation performance
  4. Leverage logs: The training log now shows all model types for easy comparison
  5. Feature count: New preprocessing creates ~25 features (vs. ~8 originally)

  The modifications are ready to use - you can run python src/train.py immediately to start with all enhancements enabled, or follow the phased approach above to incrementally
  improve accuracy while understanding each change's impact.
