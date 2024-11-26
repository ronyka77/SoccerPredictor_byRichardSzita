import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import tempfile
import shutil
from copy import deepcopy
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from scikeras.wrappers import KerasRegressor  # Ensure compatibility with Keras
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import RFE
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.regularizers import l2   
from keras.callbacks import EarlyStopping, Callback
import keras.backend as K
from keras.metrics import Metric
import h5py
from keras.backend import manual_variable_initialization 
import cloudpickle as cp
import dill
from lightgbm import LGBMRegressor 
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
manual_variable_initialization(True)
# Import utilities
import logging
from util_tools.logging_config import LoggerSetup
from util_tools.model_classes import CustomStackingRegressor, CustomReduceLROnPlateau, WithinRangeMetric, LoggingEstimator
from util_tools.model_functions import create_neural_network, prepare_data, prepare_new_data, within_range_evaluation, perform_feature_selection

model_types=['match_outcome', 'home_goals', 'away_goals']

# Set up logger with log file path
logger = LoggerSetup.setup_logger(
    name=f'Feature_selector_for_model',
    log_file=f'./log/Feature_selector_for_model.log',
    level=logging.INFO
)
# Create temp directory and ensure it exists
os.makedirs('./temp', exist_ok=True)
tempdir = tempfile.mkdtemp(dir='./temp')

# Clean up temp directory
if os.path.exists(tempdir):
    shutil.rmtree(tempdir)
os.makedirs(tempdir, exist_ok=True)

# Set environment variables for TensorFlow and temp directories
os.environ['TF_CONFIG'] = tempdir 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress detailed logs
os.environ['TEMP'] = tempdir
os.environ['TMP'] = tempdir
os.environ['TMPDIR'] = tempdir

# Log temp directory access permissions
logger.info(f"Temp directory writable: {os.access(tempdir, os.W_OK)}")
logger.info(f"Temp directory readable: {os.access(tempdir, os.R_OK)}")

# Set up model directory
model_dir = "./models/"
os.makedirs(model_dir, exist_ok=True)
logger.info("Model directory checked/created")

scaler = StandardScaler()
selected_features = []
numeric_features = []

# Load your feature-engineered data
base_data_path = './data/model_data_training_newPoisson.xlsx'


base_data = pd.read_excel(base_data_path)
logger.info(f"Data loaded from {base_data_path} with shape: {base_data.shape}")

# Filter data to only include away_goals from 0 to 5
base_data = base_data[(base_data['home_goals'] >= 0) & (base_data['home_goals'] <= 6) & (base_data['away_goals'] >= 0) & (base_data['away_goals'] <= 6) ]
base_data = base_data.drop(columns=['Unnamed: 0.1','Unnamed: 0.2','Unnamed: 0','Odd_Home','Odds_Draw','Odd_Away'], errors='ignore')
base_data.replace([np.inf, -np.inf], np.nan, inplace=True)
base_data = base_data.dropna()
logger.info(f"Data filtered to only include away_goals from 0 to 6. rows: {len(base_data)} Filtered data shape: {base_data.shape}")

# Drop unnecessary columns
data = base_data.drop(columns=['Unnamed: 0.1','Unnamed: 0.2','Unnamed: 0','match_outcome', 'home_goals','away_goals',  
                               'draw', 'away_win', 'home_win','away_points', 'home_points','HomeTeam_last_away_match','AwayTeam_last_home_match',
                               'home_points_rolling_avg','away_points_rolling_avg','home_advantage',
                               'Odd_Home','Odds_Draw','Odd_Away'], errors='ignore')

# Select all numeric features
numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
logger.info(f"Numeric features: {numeric_features}")

# After initial data preparation
logger.info(f"Initial number of features: {len(data.columns)}")
logger.info(f"Initial feature names: {data.columns.tolist()}")

for model_type in model_types:
    logger.info(f"\nPerforming feature selection for {model_type}")
    
    # Prepare data
    X = prepare_data(data, numeric_features, model_type, model_dir, logger)
    y = base_data[model_type]  # Target variable
    logger.info(f"Data prepared for modeling. Feature shape: {X.shape}, Target shape: {y.shape}")

    # Generate polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    logger.info("Polynomial feature transformation started.")
    X_poly = poly.fit_transform(X)

    # After polynomial feature generation
    logger.info(f"Number of polynomial features: {X_poly.shape[1]}")
    logger.info(f"Polynomial feature names: {poly.get_feature_names_out(X.columns)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
    logger.info(f"Data split into train and test. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Scaling
    logger.info("Data scaling started.")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Save the scaler
    scaler_file = os.path.join(model_dir, f'scaler_{model_type}.pkl')
    joblib.dump(scaler, scaler_file)
    logger.info(f"Scaler saved to {scaler_file}")

    # Perform feature selection
    selector = perform_feature_selection(X_train_scaled, y_train, model_dir, model_type, logger)

    # After feature selection
    selected_features = poly.get_feature_names_out(X.columns)[selector.support_]
    logger.info(f"Number of selected features after RFE: {len(selected_features)}")
    logger.info(f"Selected feature names: {selected_features.tolist()}")

    # Calculate feature importance if using RFE with RandomForest
    if hasattr(selector.estimator_, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': selector.estimator_.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        logger.info("\nFeature Importance:")
        logger.info(feature_importance.to_string())

logger.info("Feature selection completed for all model types")
