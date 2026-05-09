from fastapi import APIRouter, HTTPException
from api.schemas import EnergyPredictionRequest, EnergyPredictionResponse, HealthResponse, ModelInfoResponse
from src.predict import predict_energy
import datetime

router = APIRouter()

@router.post("/predict", response_model=EnergyPredictionResponse)
async def predict(request: EnergyPredictionRequest):
    """
    Predict next-day energy consumption based on the provided climate and lag features.
    """
    try:
        # Map the API request fields to the format expected by predict_energy in src/predict.py
        # src/predict.py expects the raw feature names (e.g., T2M, RH2M)
        input_data = {
            "T2M": request.temperature,
            "RH2M": request.humidity,
            "WS10M": request.wind_speed,
            "PS": request.pressure,
            "CLOUD_AMT": request.cloud_amount,
            "PRECTOTCORR": request.precipitation,
            "ALLSKY_SFC_SW_DWN": (request.lag_1), # Energy proxy for the current day
            "lag_1": request.lag_1,
            "lag_7": request.lag_7,
            "rolling_mean_7": request.rolling_mean_7,
            "day_of_week": request.day_of_week,
            "month": request.month,
            "season": request.season,
            "is_weekend": 1 if request.day_of_week >= 5 else 0,
            # Adding mock values for lag_30 and rolling_std_7 if they aren't in the request schema
            # In a real production app, these would be calculated from a database
            "lag_30": request.lag_7,
            "rolling_std_7": 1.0
        }

        prediction = predict_energy(input_data)
        return prediction

    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model files not found. Please run train.py first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns the health status of the API.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.datetime.now().isoformat()
    )

@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Returns metadata about the currently deployed model.
    """
    try:
        import joblib
        from config import BEST_MODEL_PATH
        model = joblib.load(BEST_MODEL_PATH)

        # Try to read the training log for metrics
        import pandas as pd
        log_df = pd.read_csv("training_log.csv")
        best_row = log_df.iloc[log_df['RMSE'].idxmin()]

        return ModelInfoResponse(
            model_name=type(model).__name__,
            version="1.0.0",
            training_metrics=best_row.to_dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not retrieve model info: {str(e)}")
