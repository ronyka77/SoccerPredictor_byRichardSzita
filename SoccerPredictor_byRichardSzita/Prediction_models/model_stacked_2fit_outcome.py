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

model_type='match_outcome'

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
new_prediction_data.replace([np.inf, -np.inf], np.nan, inplace=True)
logger.info(f"Data loaded from {new_prediction_path} with shape: {new_prediction_data.shape}")
logger.info(f"Data loaded with {len(new_prediction_data)} rows")

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

# Load previously predicted scores
try:
    real_scores = pd.read_excel(real_scores_path)
    real_scores = real_scores[real_scores['Date'] < '2024-11-06']
    real_scores = real_scores.dropna(subset=['real_score'])
    real_scores.replace([np.inf, -np.inf], np.nan, inplace=True)
    logger.info(f"Previously predicted scores loaded from {real_scores_path} with shape: {real_scores.shape}")
    # print(real_scores.head())
   
except Exception as e:
    logger.error(f"Error loading previously predicted scores: {e}")
    raise

# Prepare real scores data for additional fit
def prepare_real_scores_data(real_scores: pd.DataFrame, new_prediction_data: pd.DataFrame, base_data: pd.DataFrame, model_type: str, original_features: list) -> pd.DataFrame:
    """
    Prepare the real scores data for additional model fitting by merging with new prediction data.

    Args:
        real_scores (pd.DataFrame): The DataFrame containing real scores.
        new_prediction_data (pd.DataFrame): The DataFrame containing new prediction data with all features.
        model_type (str): The type of model to train.
        original_features (list): The list of original features used in the model.

    Returns:
        pd.DataFrame: The prepared feature data.
    """
    
    df_for_merge = new_prediction_data.drop(columns=['home_points','HomeTeam_last_away_match','AwayTeam_last_home_match',
                                                    'home_poisson_xG','away_poisson_xG',
                                                    'Odd_Home','Odds_Draw','Odd_Away','Unnamed: 0.1', 'Unnamed: 0','Date'], errors='ignore')
    # Merge real scores with new prediction data on common keys
    merged_data = pd.merge(
        real_scores,
        df_for_merge,
        on=['running_id'],  # Adjust these keys as necessary
        how='left'
    )
    merged_data[model_type] = merged_data['real_outcome']
  
    merged_data = merged_data.dropna(subset=[model_type])
    
    logger.info(f"merged data columns: {merged_data.columns.to_list()}")
    
    # Convert comma to dot for decimal conversion
    real_scores_data = merged_data.replace(',', '.', regex=True)
    real_scores_data = real_scores_data.apply(pd.to_numeric, errors='coerce')
    real_scores_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    logger.info(f"Real scores data prepared for additional fit. Shape: {real_scores_data.shape}")
    logger.debug(f"Real scores data columns: {real_scores_data.columns.tolist()}")
    print(real_scores_data.head())
    # real_scores_data.to_excel('./real_scores_data.xlsx', index=False)
    # logger.debug(f"Real scores data shape: {real_scores_data.shape}")
    return real_scores_data

# Prepare the real scores data
# real_scores_data = prepare_real_scores_data(real_scores, new_prediction_data, base_data, model_type, selected_features)

# Additional fit using real scores
def additional_fit_with_real_scores(model: StackingRegressor, real_scores_data: pd.DataFrame, model_type: str, logger: logging.Logger):
    """
    Perform an additional fit using the real scores data, only for incorrect predictions.

    Args:
        model (StackingRegressor): The trained stacking regressor model.
        real_scores_data (pd.DataFrame): The feature data for real scores.
        real_scores (pd.DataFrame): The DataFrame containing real scores.
        model_type (str): The type of model used for prediction.
        logger (logging.Logger): Logger for logging information.
    """
    try:
        # Load and apply the saved imputer from training
        selector_file = os.path.join(model_dir, f'rfe_{model_type}_selector.pkl')
        selector = joblib.load(selector_file)
        logger.info(f"Selector loaded from {selector_file}")
    except FileNotFoundError:
        logger.error(f"Selector file not found at {selector_file}")
        raise
    try:
        # Load and apply the saved scaler from training
        scaler_file = os.path.join(model_dir, f'scaler_{model_type}.pkl')
        scaler_loaded = joblib.load(scaler_file)
    except FileNotFoundError:
        logger.error(f"Scaler file not found at {scaler_file}")
        raise
    
    # Prepare the data
    X_real = prepare_data(real_scores_data, numeric_features, model_type, model_dir, logger)
    # Scale the data
    X_real_scaled = scaler_loaded.transform(X_real)

    # Feature selection
    X_real_selected = selector.transform(X_real_scaled)
    
    # Use the actual real score as the target variable
    y_real = real_scores_data[model_type]  # Adjust this to the correct column name for real scores

    # Predict using the current model
    y_pred = model.predict(X_real_selected)

    # Identify incorrect predictions
    incorrect_indices = y_pred != y_real

    # Filter data for incorrect predictions
    X_incorrect = X_real[incorrect_indices]
    y_incorrect = y_real[incorrect_indices]

    if len(X_incorrect) > 0:
        # Scale the data
        X_incorrect_scaled = scaler_loaded.transform(X_incorrect)

        # Feature selection
        X_incorrect_selected = selector.transform(X_incorrect_scaled)

        # Fit the model with incorrect predictions
        logger.info('Additional fit with real scores started for incorrect predictions')
        model.fit(X_incorrect_selected, y_incorrect)
        logger.info(f"Stacking model for {model_type} trained successfully with real scores for incorrect predictions.")
    else:
        logger.info("No incorrect predictions found; no additional fitting needed.")


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

    # Generate polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    logger.info("Polynomial feature transformation started.")
    X_poly = poly.fit_transform(X)

    # Update train-test split to use polynomial features
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_poly, y, test_size=0.3, random_state=123)

    # Scaling
    logger.info("Data scaling started.")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Save the scaler for future use
    scaler_file = os.path.join(model_dir, f'scaler_{model_type}.pkl')
    joblib.dump(scaler, scaler_file)
    logger.info(f"Scaler saved to {scaler_file}")
    
    X_test_scaled = scaler.transform(X_test)
    X_train2_scaled = scaler.transform(X_train2)
    X_test2_scaled = scaler.transform(X_test2)
    logger.info("Data scaling completed.")
    
    # Load the RFE selector for future use
    selector_file = os.path.join(model_dir, f'rfe_{model_type}_selector.pkl')
    selector = joblib.load(selector_file)
    logger.info(f"RFE selector loaded from {selector_file}")
    # selector = perform_feature_selection(X_train_scaled, y_train, model_dir, model_type, logger)

    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    X_train2_selected = selector.transform(X_train2_scaled)
    X_test2_selected = selector.transform(X_test2_scaled)

    # Log selected features
    # selected_features = list(X.columns[selector.support_])
    selected_features = list(poly.get_feature_names_out(X.columns)[selector.support_])
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
        epochs=150,
        batch_size=128,
        verbose=1,
        callbacks=[
            CustomReduceLROnPlateau(monitor='loss', factor=0.2, patience=15, min_lr=0.00001),
            EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        ]
    )
    
    # Improved Random Forest with potentially better parameters
    rf_regressor_home = RandomForestRegressor(
        n_estimators=1000,  # More trees
        max_depth=None,     # Allow trees to grow deeper
        min_samples_split=5,  # More flexibility in splitting
        min_samples_leaf=2,   # Smaller leaf size
        max_features='sqrt',  # Consider a subset of features at each split
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    # AdaBoost Regressor with optimized parameters for better performance
    ada_regressor_home = AdaBoostRegressor(
        base_estimator=DecisionTreeRegressor(max_depth=3, min_samples_split=5),
        n_estimators=500,
        learning_rate=0.05,
        loss='linear',
        random_state=42
    )

    # Wrap each estimator with the LoggingEstimator
    estimators_home = [
        ('lgb', LoggingEstimator(lgb_regressor_home, 'LightGBM', logger)),
        # ('catboost', LoggingEstimator(catboost_regressor_home, 'CatBoost', logger)),
        ('xgb', LoggingEstimator(xgb_regressor_home, 'XGBoost', logger)),
        ('nn', LoggingEstimator(nn_regressor_home, 'Neural Network', logger)),
        ('rf', LoggingEstimator(rf_regressor_home, 'Random Forest', logger)),
        ('ada', LoggingEstimator(ada_regressor_home, 'AdaBoost', logger))
    ]
    
    stacking_regressor_home = StackingRegressor(estimators=estimators_home, final_estimator=Ridge())
    
    # Perform two separate fits
    logger.info('First fit started')
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
    
    # try:
    #     # Perform the real scores fit
    #     logger.info('fit with real scores started')
    #     additional_fit_with_real_scores(stacking_regressor_home, real_scores_data, model_type, logger)
    # except Exception as e:
    #     logger.error(f"Error occurred while performing additional fit with real scores: {e}")
    #     raise
    
    return stacking_regressor_home

# Make new predictions
def make_prediction(model_type: str, model: StackingRegressor, prediction_data: pd.DataFrame, model_dir: str, logger: logging.Logger, numeric_features: list):
    """
    Make predictions using the trained model and save the results to an Excel file.

    Args:
        model_type (str): The type of model used for prediction.
        model (StackingRegressor): The trained stacking regressor model.
        prediction_data (pd.DataFrame): The data to make predictions on.
        model_dir (str): Directory where the model artifacts are stored.
        logger (logging.Logger): Logger for logging information.
        numeric_features (list): List of numeric features used in the model.
    """
    # Load and apply the saved imputer from training
    imputer_file = os.path.join(model_dir, f'imputer_{model_type}.pkl')
    selector_file = os.path.join(model_dir, f'rfe_{model_type}_selector.pkl')

    try:
        imputer = joblib.load(imputer_file)
        logger.info(f"Imputer loaded from {imputer_file}")
    except FileNotFoundError:
        logger.error(f"Imputer file not found at {imputer_file}")
        raise

    try:
        selector = joblib.load(selector_file)
        logger.info(f"Selector loaded from {selector_file}")
    except FileNotFoundError:
        logger.error(f"Selector file not found at {selector_file}")
        raise

    # Prepare new data for prediction
    X_new_prepared = prepare_new_data(prediction_data, imputer, selector, model_type, model_dir, logger, numeric_features)

    # Predict the home goals using the original model
    prediction = model.predict(X_new_prepared)
    prediction_column_name = f'{model_type}_prediction'
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
    make_prediction(model_type, stacking_regressor, prediction_df, model_dir, logger, numeric_features)   
        
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