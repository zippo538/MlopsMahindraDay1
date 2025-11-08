from fastapi import FastAPI, HTTPException
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
import pandas as pd
from datetime import datetime
import os
import pickle
from typing import Tuple, Any, Dict, List

# Mengimpor modul lokal
from src.utils.logger import default_logger as logger
from src.utils.config import config
from src.data.data_processor import DataProcessor
# Mengimpor skema baru untuk Regresi
from src.api.schemas import (
    HousePredictionRequest,
    HousePredictionResponse,
    ModelInfo,
    ModelMetrics
)

app = FastAPI(
    title="New York House Price Prediction API",
    description="API for predicting New York House Prices using MLflow Model Registry.",
    version="1.0.0"
)

# Global variables untuk model dan preprocessor
model: Any = None
preprocessor: DataProcessor = None
model_info: ModelInfo = None

# Tentukan nama model yang diharapkan (sesuai dengan yang Anda log di trainer.py)
# Asumsi: Anda memilih model terbaik dan mendaftarkannya, misal New_York_XGB_Regressor
PRODUCTION_MODEL_NAME = "XGB_Regressor_best_model" 
# Note: Pastikan kunci PRODUCTION_MODEL_NAME ada di config.yaml atau gunakan default di atas.


def load_production_model() -> Tuple[Any, ModelInfo]:
    """
    Loads the model currently marked as 'Production' from the MLflow Model Registry 
    and extracts its metadata.
    
    Returns:
        Tuple containing the loaded model object and ModelInfo metadata.
    """
    client = MlflowClient()
    
    try:
        # 1. Dapatkan versi Production dari Model Registry
        # Ini mengembalikan list, meskipun hanya boleh ada satu di Production
        production_versions: List[ModelVersion] = client.get_latest_versions(
            name=PRODUCTION_MODEL_NAME, 
            stages=["Production"]
        )
        
        if not production_versions:
            raise ValueError(f"No model found in 'Production' stage for name: {PRODUCTION_MODEL_NAME}")
            
        version_object = production_versions[0]
        run_id = version_object.run_id
        experiment = client.get_experiment_by_name(config.get('EXPERIMENT_NAME'))
        
        
        logger.info(f"Loading Production Model {version_object.name} V{version_object.version} from Run ID: {run_id}")
        
        # Try to load model by run ID
        try:
            # Load directly from run ID and model name
            model_path = f"models:/{PRODUCTION_MODEL_NAME}/Production"
            logger.info(f"Trying to load model from: {model_path}")
            loaded_model = mlflow.pyfunc.load_model(model_path)
            logger.info(f"Successfully loaded model from: {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Try alternative path with 'model' artifact name
            try:
                model_path = f"runs:/{run_id}/model"
                logger.info(f"Trying alternative path: {model_path}")
                loaded_model = mlflow.pyfunc.load_model(model_path)
                logger.info(f"Successfully loaded model from alternative path")
            except Exception as e2:
                logger.error(f"Error loading from alternative path: {str(e2)}")
                # Try local filesystem path
                try:
                    local_path = os.path.join("mlruns", experiment.experiment_id, 
                                            "models",run_id,"artifacts")
                    logger.info(f"Trying local filesystem path: {local_path}")
                    if not os.path.exists(local_path):
                        raise ValueError(f"Local path does not exist: {local_path}")
                    loaded_model = mlflow.pyfunc.load_model(local_path)
                    model_path = local_path
                    logger.info(f"Successfully loaded model from local path")
                except Exception as e3:
                    logger.error(f"Error loading from local path: {str(e3)}")
                    raise ValueError(f"Could not load model from any path. Errors:\n"
                                f"Primary: {str(e)}\n"
                                f"Alternative: {str(e2)}\n"
                                f"Local: {str(e3)}")
        
        # 4. Ambil Metrik dari Run
        run = client.get_run(run_id)
        metrics_data = run.data.metrics
        
        # Mapping metrik Regresi yang diharapkan
        metrics = ModelMetrics(
            MAE=metrics_data.get('MAE', 0.0),
            RMSE=metrics_data.get('RMSE', 0.0),
            R2_SCORE=metrics_data.get('R2_SCORE', 0.0)
        )
        
        # 5. Buat Model Info
        info = ModelInfo(
            run_id=run_id,
            model_name=PRODUCTION_MODEL_NAME,
            version=version_object.version,
            metrics=metrics,
            load_timestamp=datetime.now().isoformat()
        )
        
        return loaded_model, info

    except Exception as e:
        logger.error(f"Error loading Production model: {str(e)}")
        raise RuntimeError(f"Failed to load model from registry: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup"""
    global model, preprocessor, model_info
    
    # 1. Set MLflow tracking URI
    # Pastikan ini menunjuk ke lokasi mlflow.db Anda, defaultnya adalah 'sqlite:///mlflow.db'
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    
    try:
        logger.info("Loading production model from MLflow Registry.")
        
        # 2. Cari dan muat model Production
        model, model_info = load_production_model()
        
        # 3. Inisialisasi dan muat preprocessor
        # Ini penting agar API dapat menggunakan preprocessor yang sudah di-fit
        preprocessor = DataProcessor()
        preprocessor.load_preprocessor()
        
        logger.info(f"Model ({model_info.model_name} V{model_info.version}) and preprocessor loaded successfully")
        
    except Exception as e:
        logger.error(f"FATAL ERROR during startup: {str(e)}")
        # Biarkan pengecualian naik sehingga FastAPI gagal startup jika model tidak dapat dimuat
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "New York House Price Prediction API",
        "model_status": "Loaded",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=HousePredictionResponse)
async def predict(request: HousePredictionRequest):
    """
    Predict the house price using the loaded Production model.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or Preprocessor not loaded yet.")

    try:
        # 1. Konversi request Pydantic menjadi DataFrame 1 baris
        # Gunakan list di sekitar request.dict() agar menjadi DataFrame 1 baris (2D container)
        data = pd.DataFrame([request.dict()])[config.get('FEATURE_COLUMN')] 
        
        
        estimated_price = model.predict(data)[0]
        
        response = HousePredictionResponse(
            estimated_price=float(estimated_price),
            model_info=model_info
        )
        
        logger.info(f"Prediction complete. Estimated Price: ${estimated_price:.2f}")
        return response
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health", response_model=ModelInfo)
async def health_check():
    """Health check endpoint, returns details about the currently loaded model."""
    if model is None or model_info is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")
        
    return model_info