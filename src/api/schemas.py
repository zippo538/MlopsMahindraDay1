from pydantic import BaseModel, Field
from typing import Dict, Any

# --- Metrik untuk Masalah Regresi ---
class ModelMetrics(BaseModel):
    """Metrics specific to the deployed Regression Model."""
    MAE: float = Field(..., description="Mean Absolute Error.")
    RMSE: float = Field(..., description="Root Mean Squared Error.")
    R2_SCORE: float = Field(..., description="R2 Score (Coefficient of Determination).")

# --- Informasi Model yang Dikerahkan ---
class ModelInfo(BaseModel):
    """Information about the currently loaded MLflow model."""
    run_id: str = Field(..., description="MLflow Run ID where the model was logged.")
    model_name: str = Field(..., description="Registered Model Name.")
    version: int = Field(..., description="Model Version in Production Stage.")
    metrics: ModelMetrics = Field(..., description="Key metrics from the best run.")
    load_timestamp: str = Field(..., description="Timestamp when the model was loaded.")

# --- Skema Input Prediksi ---
class HousePredictionRequest(BaseModel):
    """Input data required for making a house price prediction."""
    BEDS: int = Field(..., description="Number of bedrooms.")
    BATH: int = Field(..., description="Number of bathrooms.")
    PROPERTYSQFT: int = Field(..., description="Property square footage.")
    LATITUDE : float = Field(..., description="Latitude Number.")
    LONGITUDE : float = Field(..., description="Longitude Number.")
    LOCALITY: str = Field(..., description="Locality or neighborhood name.")
    
    class Config:
        # Contoh data untuk dokumentasi Swagger/Redoc
        schema_extra = {
            "example": {
                "BEDS": 3,
                "BATH": 2,
                "PROPERTYSQFT": 1500,
                'LATITUDE' : 40.761255 ,
                'LONGITUDE': -73.9744834,
                "LOCALITY": "Brooklyn"
            }
        }

# --- Skema Output Prediksi ---
class HousePredictionResponse(BaseModel):
    """Prediction output containing the estimated house price."""
    estimated_price: float = Field(..., description="Estimated house price in USD.")
    model_info: ModelInfo = Field(..., description="Metadata of the model used for prediction.")