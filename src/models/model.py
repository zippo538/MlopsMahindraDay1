from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from typing import Dict,Any, Type
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.logger import default_logger as logger
from src.utils.config import config


#diseusiakan dengan grid searchcv
class ModelFactory:
    @staticmethod
    def get_model_config() -> Dict[str,Dict[str,Any]]:
        return {
        'XGB_Regressor': {
            'model' : XGBRegressor(),
            'params' : {
                'n_estinamtor' : 200,
                'max_dept'  : 4,
                'learning_rate' : 0.1, 
                'random_state' : 42
                }
            },
        'Random_Forest_Regressor': {
        'model' : RandomForestRegressor(),
        'params' : {
            'n_estimators' : 300,
            'max_depth' : 12,
            'random_state' : 42,
            'max_features' : 1.0,
            'min_samples_leaf' : 1,
            'min_samples_split' : 2,
                        }
        },    
        'Gradient_Boosting_Regressor': {
        'model' : GradientBoostingRegressor(),
        'params' : {
            'n_estimators' : 300,
            'learning_rate' : 0.05,
            'max_depth' : 6,
            'min_impurity_decrease' : 0.0,
            'min_samples_leaf' : 1,
            'min_samples_split' : 2,
            'subsample' : 1.0,
            'tolerance' : 0.0001,
            'validation_fraction' : 0.1,
            'random_state' : 42
            }
        },
        'Decision_Tree_Regressor': {
        'model' : DecisionTreeRegressor(),
        'params' : {
            'max_depth' : 15,
            'min_samples_leaf' : 1,
            'min_samples_split' : 2,
            'random_state' : 42
            }
        }
    }
    @staticmethod
    def get_gridsearch_config() -> Dict[str,Dict[str,Any]]:
        return {
        'XGB_Regressor': {
        'model' : XGBRegressor(random_state=42,enable_categorical= True, use_label_encoder = False),
        'params' : {
            'n_estinamtors' : [200,300,400],
            'max_depth'  : [4,6,8],
            'learning_rate' : [0.05 ,0.1], 
            }
        },
        'Random_Forest_Regressor': {
        'model' : RandomForestRegressor(random_state=42),
        'params' : {
            'n_estimators' : [100,200,300],
            'max_depth' : [8,10,12],
            }
        },    
        'Gradient_Boosting_Regressor': {
        'model' : GradientBoostingRegressor(random_state=42),
        'params' : {
            'n_estimators' : [100,200,300],
            'learning_rate' : [0.05,0.1],
            'max_depth' : [4,6,8],
            }
        },
        'Decision_Tree_Regressor': {
        'model' : DecisionTreeRegressor(random_state=42),
        'params' : {
            'max_depth' : [8,10,12,15],
            }
        }
        }
        
    @classmethod
    def create_model(cls,model_type:str)-> Any :
        try : 
            logger.info(f"Create model of type : {model_type}")
            
            # get model config
            model_configs=  cls.get_gridsearch_config()
            
            if model_type not in model_configs:
                raise ValueError(f"Unknown model type :{model_type}")
            
            # get model class and parameters
            model_info = model_configs[model_type]
            model_name = model_info['model']
            model_params = model_info['params']
            
            # override parameters from config if provided
            config_params = config.get(f"model_params.{model_type}",{})
            model_params.update(config_params)
            
            # create model instance
            model = model_name(**model_params)
            
            logger.info(f"Successfully Creted {model_type} model")
            return model
        
        except Exception as e:
            logger.error(f"Error Creating model : {str(e)}")
            raise
    
        
        
class PredictModel:
    def __init__(self,model_type:str):
        
        self.model_type = model_type
        self.model = None
        logger.info(f"Initiliazed Predict Model with type : {model_type}")
        
    def build(self)-> None:
        try :
            logger.info(f"Building {self.model_type} model")
            self.model = ModelFactory.create_model(self.model_type)
            logger.info("Model Built Successfully")
        except Exception as e:
            logger.error(f"Error building Model : {str(e)}")
            raise
    
    def get_params(self) -> Dict[str,Any]:
        if self.model is None :
            raise ValueError("Model not built yet")
        return self.model.get_params()