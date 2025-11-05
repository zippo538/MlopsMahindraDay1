import os
import sys
from pathlib import Path
import mlflow
from sklearn.model_selection import train_test_split

# Add src to path for imports
src_path = str(Path(__file__).parent.parent)
sys.path.append(src_path)

from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.models.trainer import ModelTrainer
from src.utils.logger import default_logger as logger
from src.utils.config import config

def setup_mlflow():
    """Setup MLflow configuration"""
    try:
        # Set MLflow tracking URI explicitly
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        
        # Set experiment
        experiment_name = "telco_churn_experiment"
        try:
            mlflow.create_experiment(experiment_name)
        except:
            pass
        
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow setup completed. Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"Experiment name: {experiment_name}")
        
        return experiment_name
        
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        raise

def run_pipeline():
    """
    Run the complete training pipeline
    """
    try:
        logger.info("Starting pipeline execution")
        
        # Setup MLflow first
        experiment_name = setup_mlflow()
        
        # Start MLflow run for the entire pipeline
        with mlflow.start_run(run_name="full_pipeline") as parent_run:
            # Log pipeline run ID
            logger.info(f"Started pipeline run with ID: {parent_run.info.run_id}")
            
            # 1. Load Data
            logger.info("Step 1: Loading data")
            data_loader = DataLoader()
            df = data_loader.load_data()
            
            if not data_loader.validate_data(df):
                raise ValueError("Data validation failed")
            
            # Log data info
            mlflow.log_param("data_shape", str(df.shape))
            mlflow.log_param("data_columns", str(list(df.columns)))
                
            # 2. Preprocessing
            logger.info("Step 2: Preprocessing data")
            preprocessor = DataProcessor()
            X, y = preprocessor.fit_transform(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Log preprocessing info
            mlflow.log_param("train_size", X_train.shape[0])
            mlflow.log_param("test_size", X_test.shape[0])
            
            # Save preprocessors for later use
            preprocessor.save_preprocessors()
            
            # 3. Model Training and Evaluation
            logger.info("Step 3: Training and evaluating models")
            trainer = ModelTrainer(experiment_name)
            
            # Train all models (will create nested runs)
            results = trainer.train_all_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test
            )
            
            # 4. Log best model info
            best_model = trainer.get_best_model()
            mlflow.log_params({
                "best_model_type": best_model['model'].__class__.__name__,
                "best_model_params": str(best_model['model'].get_params())
            })
            mlflow.log_metrics({
                f"best_model_{k}": v for k, v in best_model['metrics'].items()
            })
            
            logger.info("Pipeline execution completed successfully")
            return True
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Delete existing MLflow database if exists
    mlflow_db = Path("mlflow.db")
    if mlflow_db.exists():
        mlflow_db.unlink()
        logger.info("Deleted existing MLflow database")
    
    # Run pipeline
    run_pipeline()