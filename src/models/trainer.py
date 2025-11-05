import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from src.utils.logger import default_logger as logger
from src.models.model import ModelFactory
from src.utils.config import config

class ModelTrainer:
    
    def __init__(self,experiment_name :str ="New York House Prediction"):
        self.experiment_name = experiment_name
        self.models_info = {}
        self.best_model = None
        self.setup_mlflow()
        logger.info(f"Initialized Model Trainer with Experiment : {experiment_name}")
    
    def _calculate_metrics(self,y_true: np.ndarray, y_pred : np.ndarray) -> Dict[str, float]:
        
        try : 
            metrics= {
            'MAE' : mean_absolute_error(y_true,y_pred),
            'RMSE' : mean_squared_error(y_true,y_pred),
            'R2_SCORE' : r2_score(y_true,y_pred)
            }
            return metrics
        except Exception as e:
            logger.error(f"Error Calculating Metrics : {str(e)}")
            raise
    def setup_mlflow(self) -> None :
        try : 
            tracking_uri = config.get('mlflow.tracking_uri', 'sqlite:///mlflow.db')
            
            mlflow.set_tracking_uri(tracking_uri)

            mlflow.set_experiment(tracking_uri)
            
            # crete or et experiment
            try :
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            except:
                self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
            mlflow.set_experiment(self.experiment_name)
            logger.info("MLfow setup completed successfully")
        
        except Exception as e:
            logger.error(f"Error settingg up MLflow : {str(e)}")
            raise
        
    def train_model(self,model_type:str, X_train: pd.DataFrame, y_train : pd.Series,X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str,Any] : 
        try : 
            logger.info(f"Training {model_type} model")
            
            model = ModelFactory.create_model(model_type)
            model.fit(X_train,y_train)
                        
            # make predictions
            y_pred = model.predict(X_test)
            
            # calucate metrics
            metrics = self._calculate_metrics(y_test,y_pred)
            
            # log with Mlflow using nested runs
            with mlflow.start_run(run_name=model_type,nested=True) as run :
                
                #log paramaters and metrics 
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                
                #log model
                mlflow.sklearn.log_model(
                    model,
                    model_type,
                    registered_model_name=f"House_Predicted_{model_type}"
                )
            
            # store model info 
            model_info = {
                'model' : model,
                'metrics' : metrics,
                'run_id' : run
            }
            
            self.models_info[model_type] = model_info
            
            return model_info
        except Exception as e:
            logger.error(f"Error Training {model_type} model : {str(e)}")
            raise
    
    def train_all_models(self,X_train: pd.DataFrame, y_train : pd.Series, X_test : pd.DataFrame, y_test: pd.Series) -> Dict[str,Dict[str,Any]]:
        
        try : 
            logger.info("Starting training of all models")
            
            for model_type in ModelFactory.get_model_config().keys():
                self.train_model(model_type,X_train,y_train,X_test, y_test)
            
            self._select_best_model()
            
            logger.info("Completed Training all Models")
            return self.models_info
        except Exception as e:
            logger.error(f"Error Training Models : {str(e)}")
            raise
    
    def _select_best_model(self)->None : 
        try :
            logger.info("Selecting Best Model")
            best_model_type = None
            
            if  best_model_type:
                self.best_model = self.models_info[best_model_type]
                
                # Transition best model to production MLflow
                client = mlflow.tracking.MlflowClient()
                model_name = f"New_York_{best_model_type}"
                
                latest_versions = client.get_latest_versions(model_name)
                if latest_versions:
                    latest_versions = latest_versions[0]
                    client.transition_model_version_stage(
                        name=model_name,
                        version=latest_versions,
                        stage="Production"
                    )
                logger.info(f"Selected {best_model_type} as best model")
                logger.info(f"Best Model Metrics :  {self.best_model['metrics']}")
        except Exception as e:
            logger.error(f"Error Selecting Best Model : {str(e)}")
            raise 
    def get_best_model(self) -> Dict[str,Any]:
        if self.best_model is None : 
            raise ValueError("No Best Model Selected. Train models first")
        return self.best_model
    
    def get_all_metrics(self) -> Dict[str,Dict[str,float]]:
        return {model_type : info['metrics']
                for model_type, info in self.models_info.items()}
        
    
                
                
            

        