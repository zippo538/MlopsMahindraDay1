import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from src.utils.logger import default_logger as logger
from src.utils.config import config

class DataLoader:
    def __init__(self,data_path : Optional[Path]=None) :
        self.data_path = data_path if data_path is not None else Path.cwd() / 'data' / 'NY-House_dataset.csv'
        logger.info(f"Initialized Dataloader with path : {self.data_path}")
    
    def load_data(self) -> pd.DataFrame :
        try : 
            logger.info("Loading data from csv")
            df = pd.read_csv(config.get('DATA_PATH'))
           
            logger.info(f"Data Successfully Load")
            
            return df   
        except Exception as e:
            logger.error(f"Error Feature Engineering : {e}")
            raise
    
    def validate_data(self,df:pd.DataFrame) -> True:
        try : 
            logger.info(f"Validate Data ")
            data_cleaning = config.get('data_cleaning')
            #delete value outlier
            df.drop(data_cleaning.get('DROP_VALUE_PRICE'), inplace=True)  # PRICE outliers
            df.drop(data_cleaning.get('DROP_VALUE_BEDS'), inplace=True)  # BEDS outliers
            df.drop(data_cleaning.get('DROP_VALUE_PROPERTYSQFT'), inplace=True)  # PROPERTYSQFT outliers
            
            logger.info("Delete Outlier Value...")
            
            #drop columns 
            df.drop(data_cleaning.get('DROP_COLUMNS'), axis=1, inplace=True)
            
            logger.info(f"Delete Columns {data_cleaning.get('DROP_COLUMNS')}")
            
            
            # drop duplicate 
            df.drop_duplicates(inplace=True)
            
            logger.info(f"Drop Duplicate Value")
            
            
            #change type int
            df['BATH'] = df['BATH'].astype(int)
            df['PROPERTYSQFT'] = df['PROPERTYSQFT'].astype(int)
            
            logger.info(f"Change type int columns BATH and PROPERTYSQFT ")
            
            return True
        except Exception as e : 
            logger.error(f"Error Validate Data {str(e)} ")
            raise
    
    def split_features_target(self, df:pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        try :
            model_parameters = config.get('model_parameters') 
            logger.info("Splitting Features and Target")
            X = df.drop(model_parameters.get('TARGET_COLUMN'),axis=1)
            y = np.log(df[model_parameters.get('TARGET_COLUMN')])
            
            logger.info(f"Split Completed. Features shape : {X.shape}, Target Shape : {y.shape}")
            
            return X,y
            
        except Exception as e:
            logger.error(f"Error Split Features : {e}")
        
        