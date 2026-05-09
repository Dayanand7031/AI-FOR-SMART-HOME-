from pydantic import BaseModel, Field
from typing import Optional

class EnergyPredictionRequest(BaseModel):
    """
    Pydantic model for energy prediction request.
    Matches the features generated in preprocessing.py.
    """
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")
    temperature: float = Field(..., description="T2M: Temperature at 2 meters")
    humidity: float = Field(..., description="RH2M: Relative Humidity")
    wind_speed: float = Field(..., description="WS10M: Wind Speed at 10 meters")
    pressure: float = Field(..., description="PS: Surface Pressure")
    cloud_amount: float = Field(..., description="CLOUD_AMT: Cloud Amount")
    precipitation: float = Field(..., description="PRECTOTCORR: Precipitation")
    lag_1: float = Field(..., description="Previous day solar radiation")
    lag_7: float = Field(..., description="Solar radiation from 7 days ago")
    rolling_mean_7: float = Field(..., description="7-day rolling average of solar radiation")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0-6)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    season: int = Field(..., ge=0, le=3, description="Season index (0-3)")

class EnergyPredictionResponse(BaseModel):
    """
    Pydantic model for energy prediction response.
    """
    predicted_energy_consumption: float
    unit: str = "kWh/m²/day"
    confidence: str
    model_used: str

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    training_metrics: dict
