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
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

class ModelTrainer:
    
    def __init__(self,experiment_name :str =config.get('EXPERIMENT_NAME')):
        self.experiment_name = experiment_name
        self.models_info = {}
        self.best_model = None
        self.model_pipeline : Pipeline
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
    
        
        
    def train_gridsearch(self,preprocessor,X_train: pd.DataFrame,X_test : pd.DataFrame, y_train : pd.Series, y_test : pd.Series) -> None:
        
        try : 
            logger.info("Starting training of all models")
            
            
            #running Grid Search CV
            for model_type,model_config in ModelFactory.get_gridsearch_config().items():
                with mlflow.start_run(run_name=model_type,nested=True) as run :
                    model_instance = model_config['model']
                    params_instance = model_config['params']
                    
                    pipeline_gs = self.build_pipeline(preprocessor,model_instance)
                    
                    run_gridsearch = self.model_gridsearch(pipeline_gs,X_train,y_train,params_instance)
                    
                    # get best params
                    best_model = run_gridsearch.best_estimator_
                    
                    #train pipeline 
                    pipeline_gs.fit(X_train,y_train)
                    
                    #make predictions pipeline
                    y_pred = pipeline_gs.predict(X_test)
                    
                    # save log paramns gridsearch
                    mlflow.log_params(best_model.get_params())
                    
                    #make prediction gridsearch
                    y_pred = pipeline_gs.predict(X_test)
                    
                    metrics = self._calculate_metrics(y_pred,y_test)
                    
                    #log metrik
                    mlflow.log_metrics(metrics)
                    
                    self.models_info[model_type] = {
                        'model' : best_model,
                        'metrics' : metrics,
                        'run_id' : run.info.run_id
                    }
                    
                    
                    #save log model
                    mlflow.sklearn.log_model(best_model,
                                             model_type,
                                             registered_model_name=f"{model_type}_best_model"
                                             )
            
                
            logger.info("Completed Train Grid Search CV")    
                
            self._select_best_model()
            
            
            logger.info("Completed Training all Models")
            return self.models_info
        except Exception as e:
            logger.error(f"Error Training Models : {str(e)}")
            raise
    
    def model_gridsearch(
        self,
        pipeline : Pipeline, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        param_grid: Dict[str, List[Any]], 
        scoring_metric: str = 'neg_mean_squared_error',
        cv: int = 3
    ) :
        """
        Menerapkan Pipeline penuh (Preprocessing + Model) dan melakukan Grid Search.
        
        Args:
            X_train: Data fitur yang sudah di-preprocess (dari fit_transform).
            y_train: Data target.
            model: Estimator model regresi (e.g., SVR(), Ridge()).
            param_grid: Kamus parameter yang akan dicari oleh GridSearchCV.
            scoring_metric: Metrik untuk dioptimalkan (neg_root_mean_squared_error, r2, dll).
            cv: Jumlah fold untuk cross-validation.
            
        Returns:
            GridSearchCV: Objek GridSearchCV yang sudah fit.
        """
        try : 
            logger.info("Starting Model GridSearchCV....")
            
            
            
            
            # Ubah kunci param_grid: 'C' -> 'regressor__C'
            grid_search_params = {f'regressor__{key}': value for key, value in param_grid.items()}

            
            # 3. Inisialisasi dan Jalankan Grid Search
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=grid_search_params,
                scoring=scoring_metric,
                cv=cv,
                verbose=1,
                n_jobs=-1 
            )
            
            # Grid Search Fit: Ini akan menjalankan semua langkah di full_pipeline 
            # (termasuk transform oleh preprocessor) untuk setiap kombinasi parameter
            grid_search.fit(X_train, y_train)
            
                        
            
            logger.info(f"GridSearchCV completed. Best score: {grid_search.best_score_:.4f}")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            return grid_search
        except Exception as e:
            logger.error(f"Error Train GridSearchCV : {str(e)}")
            raise 
    
    @staticmethod
    def build_pipeline(preprocessor : Any,model: BaseEstimator) -> Pipeline: 
        full_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
        full_pipeline
        return full_pipeline
        
    
    
    def _select_best_model(self)->None : 
        try :
            logger.info("Selecting Best Model")
            best_model_type = None
            best_r2_score = -float('inf')
            
            for model_type, model_info in self.models_info.items():
                metrics = model_info.get('metrics', {})
                current_r2 = metrics.get('R2_SCORE', -float('inf'))

                if current_r2 > best_r2_score:
                    best_r2_score = current_r2
                    best_model_type = model_type
                
            
            if  best_model_type:
                self.best_model = self.models_info[best_model_type] 

                # Ambil objek model untuk pickling:
                model_instance_to_pickle = self.best_model['model']
                # Transition best model to production MLflow
                client = mlflow.tracking.MlflowClient()
                model_name = f"{best_model_type}_best_model"
                
                latest_versions = client.get_latest_versions(model_name)
                if latest_versions:
                    model_version_object = latest_versions[0]
                    version_id = model_version_object.version
                    client.transition_model_version_stage(
                        name=model_name,
                        version=version_id,
                        stage="Production"
                    )
                #save model pickle
                with open("best_model.pkl",'wb') as f:
                    pickle.dump(model_instance_to_pickle,f)
                
                logger.info(self.models_info)
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
        
    
                
                
            

        