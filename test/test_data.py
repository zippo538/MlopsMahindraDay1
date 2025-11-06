import pytest
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple

# ----------------------------------------------------
# A. MOCKING OBJEK EKSTERNAL
# ----------------------------------------------------

# 1. Mock Logger
# Menggunakan logging standar Python
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 2. Mock Config Class (Mencerminkan nilai-nilai dari config.py)
class MockConfig:
    # --- Data Paths (Perlu diubah dari string ke Path untuk pengujian) ---
    # Kita tidak ingin menggunakan path sebenarnya, jadi kita buat path dummy.
    BASE_DIR = Path("/mock/base")
    ARTIFACTS_DIR = BASE_DIR / "artifact"
    DATA_PATH = ARTIFACTS_DIR / "NY-House-Dataset.csv" # Path yang akan di-mock
    
    # --- Kolom & Nilai ---
    TARGET_COLUMN = "PRICE"
    BATH = 'BATH'
    PROPERTYSQFT = 'PROPERTYSQFT'

    # --- Drop Values (Indeks/Baris) ---
    # Kita hanya perlu beberapa untuk pengujian, bukan semua yang ada di config.py
    # Kita ambil 3 baris pertama untuk drop agar mudah diverifikasi: 0, 1, 2
    DROP_VALUE_PRICE = [0]
    DROP_VALUE_BEDS = [1]
    DROP_VALUE_PROPERTYSQFT = [2]
    
    # --- Drop Columns ---
    # Kita ambil 2 kolom dari daftar panjang untuk diuji
    DROP_COLUMNS = ['TYPE', 'STATE'] 

# Tetapkan Config global ke MockConfig
Config = MockConfig

# ----------------------------------------------------
# B. KELAS DATALOADER (Disertakan di sini untuk isolasi pengujian)
# ----------------------------------------------------

class DataLoader:
    def __init__(self, data_path: Optional[Path] = None):
        # Menggunakan Config.DATA_PATH jika tidak ada data_path yang diberikan
        self.data_path = data_path if data_path is not None else Config.DATA_PATH 
        logger.info(f"Initialized Dataloader with path : {self.data_path}")
    
    def load_data(self) -> pd.DataFrame:
        try : 
            logger.info("Loading data from csv")
            # Perubahan: Menggunakan self.data_path yang sudah benar
            df = pd.read_csv(self.data_path) 
           
            logger.info(f"Data Successfully Load")
            
            return df   
        except Exception as e:
            # Perubahan: Menyimpan error yang lebih deskriptif
            logger.error(f"Error Loading Data: {e}") 
            raise
    
    def validate_data(self,df:pd.DataFrame): # Tidak mengembalikan, karena inplace=True
        # delete value outlier
        df.drop(Config.DROP_VALUE_PRICE, inplace=True) 
        df.drop(Config.DROP_VALUE_BEDS, inplace=True) 
        df.drop(Config.DROP_VALUE_PROPERTYSQFT, inplace=True) 
        
        logger.info("Delete Outlier Value...")
        
        # drop columns 
        df.drop(Config.DROP_COLUMNS, axis=1, inplace=True)
        
        logger.info(f"Delete Columns {Config.DROP_COLUMNS}")
        
        # drop duplicate 
        df.drop_duplicates(inplace=True)
        
        logger.info(f"Drop Duplicate Value")
        
        # change type int
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
            raise

# ----------------------------------------------------
# C. FIXTURES DAN UNIT TESTS
# ----------------------------------------------------

@pytest.fixture
def mock_raw_data():
    """Fixture yang menyediakan DataFrame mock untuk simulasi data mentah."""
    data = {
        'PRICE': [100000.0, 200000.0, 300000.0, 200000.0, 400000.0, 400000.0], # Baris 5 adalah duplikat baris 4
        'BEDS': [3, 4, 2, 4, 5, 5],
        'BATH': [1.5, 2.0, 1.0, 2.5, 3.0, 3.0], 
        'PROPERTYSQFT': [1500.5, 2000.0, 1000.0, 2000.5, 3000.0, 3000.0], 
        'TYPE': ['T1', 'T2', 'T3', 'T4', 'T5', 'T5'], # Akan dihapus
        'STATE': ['NY', 'NJ', 'CA', 'FL', 'TX', 'TX'], # Akan dihapus
        'LOCALITY': ['A', 'B', 'C', 'D', 'E', 'E'] # Akan dipertahankan
    }
    # Indeks 0, 1, 2, 3, 4, 5
    return pd.DataFrame(data, index=range(6)).copy()

def test_dataloader_initialization():
    """Menguji inisialisasi DataLoader menggunakan path default dari Config."""
    loader = DataLoader()
    assert loader.data_path == Config.DATA_PATH

def test_load_data_success(mocker, mock_raw_data):
    """Menguji keberhasilan load_data dengan memock pd.read_csv."""
    # Mocking pd.read_csv agar mengembalikan data dummy
    mocker.patch('pandas.read_csv', return_value=mock_raw_data)
    
    loader = DataLoader()
    df = loader.load_data()
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape == mock_raw_data.shape
    
    # Pastikan pd.read_csv dipanggil dengan path yang benar
    # Perlu memverifikasi path yang digunakan oleh loader adalah Config.DATA_PATH
    pandas.read_csv.assert_called_once_with(Config.DATA_PATH)

def test_validate_data_transformation(mock_raw_data):
    """Menguji semua langkah transformasi dalam validate_data."""
    df_before = mock_raw_data.copy()
    initial_rows = df_before.shape[0] # 6 baris
    
    loader = DataLoader()
    loader.validate_data(df_before) # Perubahan dilakukan secara inplace
    
    # 1. Uji Penghapusan Outlier (Drop Baris 0, 1, dan 2)
    # Baris yang diharapkan tersisa: 3, 4, 5 (Total 3 baris)
    assert df_before.shape[0] == 3
    assert list(df_before.index) == [3, 4, 5] 
    
    # 2. Uji Penghapusan Duplikat (Baris 5 adalah duplikat baris 4, harus terhapus)
    # Total baris setelah duplikat dihapus harus 2 (baris 3 dan 4)
    assert df_before.shape[0] == 2
    
    # 3. Uji Penghapusan Kolom
    # Harusnya menghapus 'TYPE' dan 'STATE'
    assert 'TYPE' not in df_before.columns
    assert 'STATE' not in df_before.columns
    assert 'LOCALITY' in df_before.columns 
    
    # Kolom awal 7 - 2 (dihapus) = 5 kolom tersisa
    assert df_before.shape[1] == 5
    
    # 4. Uji Perubahan Tipe Data ke Int (BATH dan PROPERTYSQFT)
    assert df_before['BATH'].dtype == np.int64 
    assert df_before['PROPERTYSQFT'].dtype == np.int64
    
    # Pastikan nilai float dibulatkan ke bawah (diconvert ke int)
    assert df_before.loc[3, 'BATH'] == 2 # 2.5 -> 2
    assert df_before.loc[3, 'PROPERTYSQFT'] == 2000 # 2000.5 -> 2000


def test_split_features_target_success(mock_raw_data):
    """Menguji pemisahan fitur dan target, termasuk transformasi log."""
    df = mock_raw_data.copy()
    loader = DataLoader()
    
    # Lakukan validasi data terlebih dahulu agar data bersih (2 baris)
    loader.validate_data(df) 
    
    X, y = loader.split_features_target(df)
    
    # 1. Uji Dimensi
    # Data bersih memiliki 2 baris dan 5 kolom total. Target = 1 kolom, Features = 4 kolom.
    assert X.shape == (2, 4) 
    assert y.shape == (2,)
    
    # 2. Uji Kolom Target (PRICE harus di y dan tidak di X)
    assert Config.TARGET_COLUMN not in X.columns
    
    # 3. Uji Transformasi Logaritma pada Target
    # Ambil harga asli dari baris yang tersisa (baris 3 dan 4)
    original_price_3 = mock_raw_data.loc[3, 'PRICE']
    original_price_4 = mock_raw_data.loc[4, 'PRICE']
    
    # y harus merupakan logaritma dari nilai PRICE asli
    expected_log_price_3 = np.log(original_price_3)
    expected_log_price_4 = np.log(original_price_4)
    
    # Gunakan pytest.approx untuk membandingkan nilai float
    assert y.loc[3] == pytest.approx(expected_log_price_3) 
    assert y.loc[4] == pytest.approx(expected_log_price_4)