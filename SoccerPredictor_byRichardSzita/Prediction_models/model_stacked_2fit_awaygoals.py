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
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from scikeras.wrappers import KerasRegressor  # Ensure compatibility with Keras
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
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
manual_variable_initialization(True)
# Import utilities
import logging
from util_tools.logging_config import LoggerSetup
from util_tools.model_classes import CustomStackingRegressor, CustomReduceLROnPlateau, WithinRangeMetric, LoggingEstimator
from util_tools.model_functions import create_neural_network, prepare_data, prepare_new_data, within_range_evaluation

model_type='away_goals'

# Set up logger with log file path
logger = LoggerSetup.setup_logger(
    name=f'stacked_{model_type}_model',
    log_file=f'./log/stacked_{model_type}_model.log',
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

# Define model file paths
keras_nn_model_path = os.path.join(model_dir, 'nn_regressor_' + model_type + '_stacked_2fit.h5')
model_file = os.path.join(model_dir, 'model_stacked_2fit_' + model_type + '.pkl')

scaler = StandardScaler()
selected_features = []
numeric_features = []

# Load your feature-engineered data
base_data_path = './data/model_data_training_newPoisson.xlsx'
new_prediction_path = './data/merged_data_prediction.csv'
real_scores_path = './made_predictions/predictions_stacked_2fit_merged.xlsx'
base_data = pd.read_excel(base_data_path)
logger.info(f"Data loaded from {base_data_path} with shape: {base_data.shape}")

# Load data for prediction
new_prediction_data = pd.read_csv(new_prediction_path)
logger.info(f"Data loaded from {new_prediction_path} with shape: {new_prediction_data.shape}")
logger.info(f"Data loaded with {len(new_prediction_data)} rows")

# Filter data to only include away_goals from 0 to 5
base_data = base_data[(base_data['home_goals'] >= 0) & (base_data['home_goals'] <= 6) & (base_data['away_goals'] >= 0) & (base_data['away_goals'] <= 6) ]
base_data = base_data.drop(columns=['Unnamed: 0.1','Unnamed: 0.2','Unnamed: 0','Odd_Home','Odds_Draw','Odd_Away'], errors='ignore')
base_data = base_data.dropna()
logger.info(f"Data filtered to only include away_goals from 0 to 6. rows: {len(base_data)} Filtered data shape: {base_data.shape}")

# Drop unnecessary columns
data = base_data.drop(columns=['Unnamed: 0.1','Unnamed: 0.2','Unnamed: 0','match_outcome', 'home_goals','away_goals',  
                               'draw', 'away_win', 'home_win','away_points', 'home_points','HomeTeam_last_away_match','AwayTeam_last_home_match',
                               'home_points_rolling_avg','away_points_rolling_avg','home_advantage',
                               'Odd_Home','Odds_Draw','Odd_Away'], errors='ignore')

# Train the model
def train_model(base_data: pd.DataFrame, data: pd.DataFrame, model_type: str, model_dir: str, logger: logging.Logger) -> StackingRegressor:
    """
    Train a stacking regressor model with dual fitting approach.

    Args:
        base_data (pd.DataFrame): The base data containing target variable.
        data (pd.DataFrame): The feature data.
        model_type (str): The type of model to train.
        model_dir (str): Directory to save model artifacts.
        logger (logging.Logger): Logger for logging information.

    Returns:
        StackingRegressor: The trained stacking regressor model.
    """
    global selected_features
    global numeric_features
    # Select all numeric features
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    logger.info(f"Numeric features: {numeric_features}")

    X = prepare_data(data, numeric_features, model_type, model_dir, logger)
    y = base_data[model_type]  # Target variable
    logger.info(f"Data prepared for modeling. Feature shape: {X.shape}, Target shape: {y.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Data split into train and test. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    # Second fit on a different random split (or different configuration)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=123)
    logger.info(f"Data split into train2 and test2. Train2 shape: {X_train2.shape}, Test2 shape: {X_test2.shape}")

    # Scaling
    logger.info("Data scaling started.")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Save the scaler for future use
    scaler_file = os.path.join(model_dir, 'scaler_' + model_type + '.pkl')
    joblib.dump(scaler, scaler_file)
    logger.info(f"Scaler saved to {scaler_file}")
    
    X_test_scaled = scaler.transform(X_test)
    X_train2_scaled = scaler.transform(X_train2)
    X_test2_scaled = scaler.transform(X_test2)
    logger.info("Data scaling completed.")

    # Feature Selection (RFE)
    logger.info("Feature Selection started.")
    selector = RFE(estimator=RandomForestRegressor(), n_features_to_select=30)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    
    # Save the RFE selector for future use
    selector_file = os.path.join(model_dir, 'rfe_' + model_type + '_selector.pkl')
    joblib.dump(selector, selector_file)
    logger.info(f"RFE selector saved to {selector_file}")
    
    X_test_selected = selector.transform(X_test_scaled)
    X_train2_selected = selector.transform(X_train2_scaled)
    X_test2_selected = selector.transform(X_test2_scaled)

    # Log selected features
    selected_features = list(X_train.columns[selector.support_])
    logger.info("Feature selection using RFE completed.")
    logger.info(f"Selected Features: {selected_features}")
    
    # Define models for home goals prediction based on research for soccer prediction
    
    # LightGBM - Known for handling imbalanced data well and good performance on sports predictions
    lgb_regressor_home = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=12,
        num_leaves=31,
        random_state=42,
        force_col_wise=True,
        n_jobs=-1  # Use all available cores
    )
    
    # CatBoost - Handles categorical variables well and robust to overfitting
    # catboost_regressor_home = CatBoostRegressor(
    #     iterations=500,
    #     learning_rate=0.05,
    #     depth=12,
    #     verbose=2,
    #     random_state=42,
    #     thread_count=-1  # Use all available cores
    # )
    
    # XGBoost - Tuned for soccer prediction tasks
    xgb_regressor_home = XGBRegressor(
        n_estimators=500,
        max_depth=12,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        nthread=-1  # Use all available cores
    )
    
    # Neural Network with architecture suited for soccer prediction
    nn_regressor_home = KerasRegressor(
        model=create_neural_network,
        model__input_dim=X_train_selected.shape[1],
        epochs=250,
        batch_size=64,
        verbose=1,
        callbacks=[
            CustomReduceLROnPlateau(monitor='loss', factor=0.2, patience=15, min_lr=0.00001),
            EarlyStopping(monitor='loss', patience=25, restore_best_weights=True)
        ]
    )
    
    # Random Forest with parameters optimized for sports prediction
    rf_regressor_home = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    # Wrap each estimator with the LoggingEstimator
    estimators_home = [
        ('lgb', LoggingEstimator(lgb_regressor_home, 'LightGBM', logger)),
        # ('catboost', LoggingEstimator(catboost_regressor_home, 'CatBoost', logger)),
        ('xgb', LoggingEstimator(xgb_regressor_home, 'XGBoost', logger)),
        ('nn', LoggingEstimator(nn_regressor_home, 'Neural Network', logger)),
        ('rf', LoggingEstimator(rf_regressor_home, 'Random Forest', logger))
    ]
    
    logger.info('First fit started')
    stacking_regressor_home = StackingRegressor(estimators=estimators_home, final_estimator=Ridge())
    
    # Dual fitting approach - Perform two separate fits
    stacking_regressor_home.fit(X_train_selected, y_train)
    logger.info(f"Stacking model for {model_type} trained successfully on first split.")
    
    # Evaluation on the first test set
    y_pred_home = stacking_regressor_home.predict(X_test_selected)
    mse_home = mean_squared_error(y_test, y_pred_home)
    r2_home = r2_score(y_test, y_pred_home)
    mae_home = mean_absolute_error(y_test, y_pred_home)
    mape_home = np.mean(np.abs((y_test - y_pred_home) / y_test)) * 100
    within_range_home = within_range_evaluation(y_test, y_pred_home, tolerance=0.5)  # Convert to percentage

    logger.info(f"{model_type} (1st fit) Stacking Model MSE: {mse_home}, R2: {r2_home}, Stacking Model MAE: {mae_home}, Stacking Model MAPE: {mape_home}%")
    logger.info(f"{model_type} (1st fit) Stacking Model Within Range (±0.5): {within_range_home}%")
    
    # 2nd Fit of the model
    logger.info('Second fit started')
    stacking_regressor_home.fit(X_train2_selected, y_train2)
    logger.info(f"Stacking model for {model_type} trained successfully on second split.")
    
    # Evaluation on the second test set
    y_pred_home2 = stacking_regressor_home.predict(X_test2_selected)
    mse_home2 = mean_squared_error(y_test2, y_pred_home2)
    r2_home2 = r2_score(y_test2, y_pred_home2)
    mae_home2 = mean_absolute_error(y_test2, y_pred_home2)
    mape_home2 = np.mean(np.abs((y_test2 - y_pred_home2) / y_test2)) * 100
    within_range_home2 = within_range_evaluation(y_test2, y_pred_home2, tolerance=0.5)  # Convert to percentage

    logger.info(f"{model_type} (2nd fit) Stacking Model MSE: {mse_home2}, R2: {r2_home2}, Stacking Model MAE: {mae_home2}, Stacking Model MAPE: {mape_home2}%")
    logger.info(f"{model_type} (2nd fit) Stacking Model Within Range (±0.5): {within_range_home2}%")
    
    return stacking_regressor_home

# Make new predictions
def make_prediction(model_type: str, model: StackingRegressor, prediction_data: pd.DataFrame):
    """
    Make predictions using the trained model and save the results to an Excel file.

    Args:
        model_type (str): The type of model used for prediction.
        model (StackingRegressor): The trained stacking regressor model.
        prediction_data (pd.DataFrame): The data to make predictions on.
    """
    # Directory where the models are saved
    imputer_file = f'imputer_{model_type}.pkl'  # Imputer file from training
    # Load and apply the saved imputer from training
    imputer_path = os.path.join(model_dir, imputer_file)
    selector_file = os.path.join(model_dir, 'rfe_'+ model_type +'_selector.pkl')
    
    selector = joblib.load(selector_file)
    try:
        imputer = joblib.load(imputer_path)
        logger.info(f"Imputer loaded from {imputer_path}")
    except FileNotFoundError:
        logger.error(f"Imputer file not found at {imputer_path}")
        raise
    
    X_new_prepared = prepare_new_data(prediction_data, imputer, selector)
   
    # Predict the home goals using the original model
    prediction = model.predict(X_new_prepared)
    prediction_column_name= model_type + '_prediction'
    logger.info(f"{model_type} predictions: {prediction}")
    # Add predictions to the new data DataFrame
    prediction_data[prediction_column_name] = prediction
    
    # Save predictions to an Excel file
    output_file = f'./predictions_hybrid_2fit_{model_type}.xlsx'
    prediction_data.to_excel(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")

prediction_df = new_prediction_data.drop(columns=['Unnamed: 0.1','Unnamed: 0.2','Date', 'Unnamed: 0','match_outcome', 'home_goals','away_goals', 'draw', 'away_win', 'home_win','away_points', 'home_points','HomeTeam_last_away_match','AwayTeam_last_home_match','home_points_rolling_avg','away_points_rolling_avg','home_advantage'], errors='ignore')
prediction_df = prediction_df.replace(',', '.', regex=True)
prediction_df = prediction_df.apply(pd.to_numeric, errors='coerce')
prediction_df.replace([np.inf, -np.inf], np.nan, inplace=True)

logger.info(f"prediction_df length: {len(prediction_df)}")
stacking_regressor = train_model(base_data, data, model_type, model_dir, logger)
logger.info(f"Start {model_type} predictions")

# MAKE PREDICTIONS
try:
    make_prediction(model_type,stacking_regressor,prediction_df)   
        
except Exception as e:
    logger.error(f"Error occurred while making prediction: {e}")
    pass

# Initialize the wrappers
try:
    nn_model = stacking_regressor.named_estimators_['nn'].model_
    if nn_model is None:
        raise AttributeError("The underlying model is not accessible.")
except AttributeError as e:
    logger.error(f"Error occurred while accessing the model: {e}")
    raise

custom_model = CustomStackingRegressor(stacking_regressor, 
                                       nn_model,
                                       keras_nn_model_path)

# Save the model (this will save both the Keras model and the rest of the stacking regressor)
custom_model.save(model_file)