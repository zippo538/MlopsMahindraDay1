import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optinal, Tuple
from src.utils.logger import default_logger as logger
from config.config import Config
from sklearn.preprocessing import train_test_split

class DataLoader:
    def __init__(self,data_path : Optinal[Path]=None) :
        self.data_path = data_path if data_path is not None else Path.cwd() / 'data' / 'NY-House_dataset.csv'
        logger.info(f"Initialized Dataloader with path : {self.data_path}")
    
    def load_data(self) -> pd.DataFrame :
        try : 
            logger.info("Loading data from csv")
            df = pd.read_csv(Config.DATA_PATH)
           
            logger.info(f"Data Successfully Load")
            
            return df   
        except Exception as e:
            logger.error(f"Error Feature Engineering : {e}")
            raise
    
    def validate_data(self,df:pd.DataFrame) -> pd.DataFrame:
        #delete value outlier
            df.drop(Config.DROP_VALUE_PRICE, inplace=True)  # PRICE outliers
            df.drop(Config.DROP_VALUE_BEDS, inplace=True)  # BEDS outliers
            df.drop(Config.DROP_VALUE_PROPERTYSQFT, inplace=True)  # PROPERTYSQFT outliers
            
            logger.info("Delete Outlier Value...")
            
            #drop columns 
            df.drop(Config.DROP_COLUMNS, axis=1, inplace=True)
            
            logger.info(f"Delete Columns {Config.DROP_COLUMNS}")
            
            
            # drop duplicate 
            df.drop_duplicates(inplace=True)
            
            logger.info(f"Drop Duplicate Value")
            
            
            #change type int
            df[Config.BATH] = df[Config.BATH].astype(int)
            df[Config.PROPERTYSQFT] = df[Config.PROPERTYSQFT].astype(int)
            
            logger.info(f"Change type int columns BATH and PROPERTYSQFT ")
    
    def split_features_target(self, df:pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        try : 
            logger.info("Splitting Features and Target")
            X = df.drop(Config.TARGET_COLUMN,axis=1)
            y = np.log(df[Config.TARGET_COLUMN])
            
            logger.info(f"Split Completedd. Features shape : {X.shape}, Target Shape : {y.shape}")
            
            return X,y
            
        except Exception as e:
            logger.error(f"Error Split Features : {e}")
        
        