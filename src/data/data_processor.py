import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List, Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple, Dict, Optional
from pathlib import Path
from src.utils.logger import default_logger as logger
from src.utils.config import config
from sklearn.compose import ColumnTransformer


class DataProcessor :
    
    def __init__(self, preprocessing_path : Optional[str] = None):
        
        self.preprocessing_path = preprocessing_path or config.get('preprocessing_path', 'models/preprocessing')
        self.preprocessor_pipeline : Optional[ColumnTransformer] = None
        self.feature_cols : List[str] = []
        self.target_col : Optional[str] = None
        self.trained = False
        
        logger.info("Initialized DataProcessor")
        
    def _prepare_preprocessing_path(self)-> None :
        
        Path(self.preprocessing_path).mkdir(parents=True,exist_ok=True)
    
    def data_split(self, df:pd.DataFrame, target_col:str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        try : 
            self.target_col = target_col
            X = df.drop(columns=[target_col], axis=1)
            y = df[target_col] 
            self.feature_cols = X.columns.tolist()
            
                        
            logger.info(f"Split Data. New features count: {X.shape[1]}")
            return X, y
        except Exception as e:
            logger.error(f"Split Data : {str(e)}")
            raise
            

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try : 
            if not self.trained or self.preprocessor_pipeline is None:
                raise ValueError("Processor not fitted yet. Call fit_transform first.")
            
            # Pastikan kolom sesuai dengan yang digunakan saat fit
            X = df[self.feature_cols]
            X_transformed_array = self.preprocessor_pipeline.transform(X)

            # Mendapatkan nama fitur setelah OHE (sama seperti di fit_transform)
            numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            try:
                feature_names = numerical_cols + list(self.preprocessor_pipeline.named_transformers_['cat'].get_feature_names_out(categorical_cols))
            except:
                feature_names = [f"feature_{i}" for i in range(X_transformed_array.shape[1])]
                
            return pd.DataFrame(X_transformed_array, columns=feature_names, index=df.index)
        except Exception as e:
            logger.error(f"Error Transform : {str(e)}")
            raise

    ## preprocess 
    def preprocessor_columns_transformer(self,X_train: pd.DataFrame, y_train : pd.Series) -> ColumnTransformer : 
            feature_columns_transformer = config.get('columns_tranformer')
            scale = feature_columns_transformer.get('SCALE')
            ohe = feature_columns_transformer.get('OHE')
            
            # 1. Definisikan ColumnTransformer (Preprocessing Pipeline)
            preprocessor = ColumnTransformer(
                transformers=[
                    ('scale', StandardScaler(), scale),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'), ohe)
                ],
            )
            
            # Fit dan Transform
            preprocessor.fit(X_train,y_train)
            self.preprocessor_pipeline = preprocessor 
            self.trained = True
            
            self.save_preprocessor()
            
            
            logger.info(f"Data Has Been Format Coloumns Transformer")
            return preprocessor
    
            
    
    def save_preprocessor(self)-> None : 
        try : 
            logger.info(f"Saving preprocessor to {self.preprocessing_path}")
            self._prepare_preprocessing_path()
            
            # save pipeline 
            joblib.dump(
                self.preprocessor_pipeline,
                Path(self.preprocessing_path) / 'pipeline.joblib'
            )
        except Exception as e:
            logger.error(f"Error Save Preprocessor : {str(e)}")
            raise
            
    
    def load_preprocessor(self)-> None:
        try :
            logger.info(f"Loading Preprocessor from {self.preprocessing_path}")
            
            #load pipeline
            pipeline_path = Path(self.preprocessing_path) / 'pipeline.joblib'
            self.preprocessor_pipeline = joblib.load(pipeline_path)
            
            self.trained = True 
            logger.info("Preprocessor Loaded Successfully")
        except Exception as e:
            logger.error(f"Error loading preprocessor : {str(e)}")
            raise
            
    

        