import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List, Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple, Dict, Optional
from pathlib import Path
from src.utils.logger import default_logger as logger
from src.utils.config import config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator

class DataProcessor :
    
    def __init__(self, preprocessing_path : Optional[str] = None):
        
        self.preprocessin_path = preprocessing_path or config.get('preprocessig_path')
        self.scaler = StandardScaler()
        self.preprocessor_pipeline : Optional[ColumnTransformer] = None
        self.feature_cols : List[str] = []
        self.target_col : Optional[str] = None
        self.trained = False
        logger.info("Initialized DataProcessor")
        
    def _prepare_preprocessing_path(self)-> None :
        
        Path(self.preprocessin_path).mkdir(parents=True,exist_ok=True)
    
    def fit_transform(self, df:pd.DataFrame, target_col:str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        try : 
            self.target_col = target_col
            y = df[target_col] if target_col in df.columns else None
            X = df.drop(columns=[target_col], errors='ignore')
            self.feature_cols = X.columns.tolist()
            
            # Pisahkan kolom numerik dan kategorikal
            numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

            # 1. Definisikan ColumnTransformer (Preprocessing Pipeline)
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ],
                remainder='passthrough' # Biarkan kolom lain apa adanya (jika ada)
            )
            
            # Fit dan Transform
            X_transformed_array = preprocessor.fit_transform(X)
            self.preprocessor_pipeline = preprocessor # Simpan preprocessor yang sudah fit
            self.trained = True

            # Mengembalikan ke DataFrame (opsional, tergantung kebutuhan)
            # Mendapatkan nama fitur setelah OHE
            try:
                feature_names = numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
            except:
                # Fallback jika get_feature_names_out tidak tersedia
                feature_names = [f"feature_{i}" for i in range(X_transformed_array.shape[1])]

            X_transformed = pd.DataFrame(X_transformed_array, columns=feature_names, index=X.index)
            
            logger.info(f"Data transformed. New features count: {X_transformed.shape[1]}")
            return X_transformed, y
        except Exception as e:
            logger.error(f"Error FIT Transform : {str(e)}")
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

    # --- Bagian Baru: Penerapan Pipeline dan Grid Search ---
    
    def train_with_gridsearch(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        model: BaseEstimator, 
        param_grid: Dict[str, List[Any]], 
        scoring_metric: str = 'neg_root_mean_squared_error',
        cv: int = 5
    ) -> GridSearchCV:
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
        if self.preprocessor_pipeline is None:
             raise ValueError("Preprocessing pipeline not fitted. Run fit_transform first.")
        
        # 2. Buat Pipeline Penuh: Preprocessing (ColumnTransformer) + Model
        # Catatan: Kita menggunakan preprocessor_pipeline yang sudah di fit.
        full_pipeline = Pipeline(steps=[
            # Tahap 1: Preprocessing (Sudah dilakukan di fit_transform, 
            # tapi di sini kita hanya menggunakan ColumnTransformer sebagai step)
            # Karena ColumnTransformer sudah disimpan, kita bisa menggunakannya di sini
            ('preprocessor', self.preprocessor_pipeline),
            
            # Tahap 2: Model Regresi
            ('regressor', model)
        ])
        
        # Penamaan parameter untuk GridSearch: 'nama_step__nama_parameter'
        # Contoh: Jika model Anda adalah 'regressor', parameter 'C' akan menjadi 'regressor__C'
        # Kita perlu memodifikasi param_grid agar sesuai dengan nama step 'regressor'
        
        # Ubah kunci param_grid: 'C' -> 'regressor__C'
        grid_search_params = {f'regressor__{key}': value for key, value in param_grid.items()}

        logger.info(f"Starting GridSearchCV with scoring: {scoring_metric}")
        
        # 3. Inisialisasi dan Jalankan Grid Search
        grid_search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=grid_search_params,
            scoring=scoring_metric,
            cv=cv,
            verbose=1,
            n_jobs=-1 # Gunakan semua core CPU
        )
        
        # Grid Search Fit: Ini akan menjalankan semua langkah di full_pipeline 
        # (termasuk transform oleh preprocessor) untuk setiap kombinasi parameter
        grid_search.fit(X_train, y_train)
        
        logger.info(f"GridSearchCV completed. Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search
    
    

        