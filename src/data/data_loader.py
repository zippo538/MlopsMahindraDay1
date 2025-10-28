import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optinal, Tuple

class DataLoader:
    def __init__(self,data_path : Optinal[Path]=None) :
        self.data_path = data_path if data_path is not None else Path.cwd() / 'data' / 'house_data.csv'
    
    def load_data(self) -> pd.DataFrame :
        try : 
            logger.info(f"Loading data from {self.data_path}")
            df = pd.read