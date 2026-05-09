# AI for Smart Home — Energy Forecasting

An end-to-end machine learning system to forecast daily energy consumption (represented by solar radiation) using NASA POWER API climate data.

##  Project Overview
This project implements a complete pipeline from data ingestion to model deployment via a REST API. It compares multiple Machine Learning (XGBoost, Random Forest, LightGBM) and Deep Learning (LSTM, GRU, CNN-LSTM) models to predict the next day's energy consumption.

##  Project Structure
```text
AI FOR SMART HOME/
├── data/
│   ├── raw/               # Raw data fetched from NASA
│   └── processed/         # Cleaned & feature-engineered data
├── notebooks/
│   └── eda.ipynb          # Exploratory Data Analysis
├── src/
│   ├── data_ingestion.py  # Fetch data from NASA POWER API
│   ├── preprocessing.py   # Clean, encode, create features
│   ├── train.py           # Train ML/DL model
│   ├── evaluate.py        # Evaluate model metrics
│   └── predict.py         # Load model and run inference
├── models/
│   └── best_model.pkl     # Saved trained model
├── api/
│   ├── main.py            # FastAPI app
│   ├── schemas.py         # Pydantic request/response models
│   └── router.py          # API route definitions
├── config.py              # All constants and config values
├── requirements.txt
└── README.md
```

## Installation & Setup

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

##  Data Pipeline

### 1. Data Ingestion
Run the ingestion script to fetch data for your specific coordinates:
```bash
python src/data_ingestion.py
```
*Data is fetched from NASA POWER API and saved as CSV in `data/raw/`.*

### 2. Preprocessing
The system automatically handles missing values, generates lag features (1, 7, 30 days), calculates rolling statistics, and encodes seasonal information.

### 3. Training
Train and compare all candidate models:
```bash
python src/train.py
```
*The best model is automatically saved to `models/best_model.pkl` and metrics are logged to `training_log.csv`.*

### 4. Evaluation
Visualize model performance and feature importance:
```bash
python src/evaluate.py
```
*Plots are saved to the `outputs/` folder.*

## 🌐 API Deployment

Run the FastAPI server:
```bash
python api/main.py
```
The API will be available at `http://localhost:8000`. Access the interactive docs at `http://localhost:8000/docs`.

### Sample API Request

**Using cURL:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "temperature": 25.5,
  "humidity": 60.2,
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
}'
```

**Using Python:**
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "latitude": 40.7128, "longitude": -74.0060, "temperature": 25.5,
    "humidity": 60.2, "wind_speed": 4.1, "pressure": 1012.5,
    "cloud_amount": 15.0, "precipitation": 0.0, "lag_1": 14.2,
    "lag_7": 12.5, "rolling_mean_7": 13.1, "day_of_week": 3,
    "month": 6, "season": 2
}
response = requests.post(url, json=data)
print(response.json())
```

##  Model Performance
Refer to `training_log.csv` for the latest metrics across:
- **RMSE** (Root Mean Squared Error) -- 4.928
- **MAE** (Mean Absolute Error) ---3.93
- **MAPE** (Mean Absolute Percentage Error) - 25%
- **R² Score** ---0.59

### Compared to Earlier

## Metric	Old	New
####  RMSE	5.096	4.928 ✅
####  MAE	4.039	3.930 ✅
#### R²	0.551	0.581 ✅
