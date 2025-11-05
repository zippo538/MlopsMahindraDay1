import os
from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    ARTIFACTS_DIR = BASE_DIR / "artifact"
    LOGS_DIR = BASE_DIR / "logs"
    STATIC_DIR = BASE_DIR / "static"
    ASSETS_DIR = STATIC_DIR/ "assets"
    CSS_DIR = STATIC_DIR / "css"
    
    # Data paths
    DATA_PATH = ARTIFACTS_DIR / "NY-House-Dataset.csv"
    MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"
    ENCODING_PATH = ARTIFACTS_DIR / "encoder.pkl"
    METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
    ALL_METRICS_PATH = ARTIFACTS_DIR / "all_metrics.json"
    BEST_ESTIMATOR_PATH = ARTIFACTS_DIR / "best_estimators.json"
    
    #CSS PATH
    PROFILE_CSS = ASSETS_DIR/ "profile.css"
    
    # model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    TARGET_COLUMN = "PRICE"
    
    #feature coloumns for model 
    FEATURE_COLUMN = ['PRICE', 'BEDS', 'BATH', 'PROPERTYSQFT', 'LOCALITY']
    
    
    # feature LOCALITY 
    LOCALITY_COLUMN = ['New York', 'New York County', 'The Bronx', 'Kings County',
       'Bronx County', 'Queens County', 'Richmond County',
       'United States', 'Brooklyn', 'Queens', 'Flatbush']
    
    #feature description
    FEATURE_DESCRIPTIONS = {
    "PRICE": "Harga jual rumah dalam satuan dolar AS. Nilai ini menjadi target utama yang akan diprediksi untuk mengetahui estimasi harga properti.",
 
    "BEDS": "Jumlah kamar tidur yang tersedia di dalam rumah. Biasanya semakin banyak kamar tidur, semakin tinggi harga rumahnya — meskipun juga bergantung pada lokasi dan luas bangunan.",
    
    "BATH": "Jumlah kamar mandi yang terdapat di properti. Fitur ini sering memengaruhi kenyamanan dan menjadi salah satu faktor penting dalam menentukan nilai properti.",
    
    "PROPERTYSQFT": "Luas bangunan rumah (Property Square Feet) dalam satuan kaki persegi. Fitur ini menunjukkan seberapa besar ukuran properti, dan biasanya berkorelasi positif dengan harga rumah.",
    
    "LOCALITY": "Nama wilayah atau lingkungan tempat rumah berada. Faktor lokasi sering kali menjadi penentu utama dalam perbedaan harga antar properti, karena mencakup akses, fasilitas umum, dan nilai kawasan."
    }
    
    # Data validation rules
    DATA_VALIDATION = {
        'PRICE': {'min': 1000, 'max': 100000000},
        'BEDS': {'min': 1, 'max': 10},
        'BATH': {'min': 1, 'max': 5},
        'PROPERTYSQFT': {'min': 10, 'max': 10000},
    }
    
    
    #drop value
    DROP_VALUE_PRICE = [304, 1, 317, 310, 360, 463]
    DROP_VALUE_BEDS = [1143, 2653, 3276,2488,765,2265,3589,3603,4240,2564,1092,1410,1214,750,4191,4404]
    DROP_VALUE_PROPERTYSQFT = [4623,2146,2148,823,3130,141,2932,2054,917,1823,4353,2107,69,4,2171]
    
    # drop columns
    DROP_COLUMNS = ['TYPE','STATE','ADDRESS','MAIN_ADDRESS','ADMINISTRATIVE_AREA_LEVEL_2','STREET_NAME', 'LONG_NAME', 'FORMATTED_ADDRESS','LONGITUDE', 'LATITUDE','BROKERTITLE','SUBLOCALITY']
    
    # change type int 
    BATH = 'BATH'
    PROPERTYSQFT = 'PROPERTYSQFT'

    
    # Feature columns for model XGBoost 
    PARAMS = {
        'regressor__n_estimators' : [100,200,300,400],
        'regressor__max_depth'  : [2,4,6,8,10],
        'regressor__learning_rate' : [0.05, 0.01, 0.1], 
        }
    CV_FOLDS = 5
    SCORING = 'neg_mean_squared_error'
    
    
     # FastAPI settings
    API_TITLE = "New York House Price Prediction API"
    API_DESCRIPTION = "API for predicting house prices using the New York House Price Dataset"
    API_VERSION = "1.0.0"
    HOST = "0.0.0.0"
    PORT = 8000
    
    
    
    # Streamlit settings
    STREAMLIT_PORT = 8501
    PAGE_TITLE = "Dataset Summary Dashboard"
    PAGE_ICON = "🐋"
    LAYOUT = "wide"
    
    # path assets STREAMLIT
    RESUME_PATH = ASSETS_DIR / "CV Mahindra.pdf"
    PROFILE_PATH = ASSETS_DIR / "profile.jpeg" 
    PHOTO_1_PATH = ASSETS_DIR / "photo-1.jpg" 
    
    
    
    # --- PROFILE ME IN STREAMLIT ---
    NAME = "Mahindra Irvan Saputra"
    DESCRIPTION = """
    Recently, I was study dibimbing.id at Data Science & Articial Intelegence Batch 7 👨🏼‍🔧.
    """
    EMAIL = "mahindra.irvan538@gmail.com"
    PHONE_NUMBER = "088-690-0140"
    SOCIAL_MEDIA = {
        "GitHub": "https://github.com/hofmannj0n",
        "LinkedIn": "https://www.linkedin.com/in/mahindra-irvan-saputra-7925941aa/",
    }
    
    @classmethod
    def is_valid_feature_value(cls, feature, value):
        """Check if a feature value is within valid range."""
        ranges = cls.get_feature_range(feature)
        
        #jika fitur kategorikal (berupa list)
        if isinstance(ranges,list):
            return value in ranges
        
        #jika fitur memiliki min dan max
        elif isinstance(ranges,dict) and 'min' in ranges and 'max' in ranges : 
            try :
                value = float(value)
                return ranges['min'] <= value <= ranges['max']
            
            except (TypeError, ValueError):
                return False
        
        return False
    
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [cls.ARTIFACTS_DIR, cls.LOGS_DIR, cls.STATIC_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_feature_range(cls, feature):
        """Get the valid range for a feature."""
        range = {
            "PRICE" :  {'min' : 1000, 'max': 100000000},
            "BEDS" : {'min' : 1, 'max': 10},
            "BATH" : {'min' : 1, 'max': 5},
            "PROPERTYSQFT" :  {'min' : 100, 'max': 100000},
            "LOCALITY" : cls.LOCALITY_COLUMN
        }
        return range.get(feature,None)
    