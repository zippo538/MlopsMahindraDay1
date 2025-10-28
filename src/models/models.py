from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from typing import Dict,Any, Type
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    @classmethod
    def create_model(cls,model_type:str)-> Any :
        pass 
    