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
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from scikeras.wrappers import KerasRegressor  # Ensure compatibility with Keras
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,brier_score_loss,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.calibration import CalibratedClassifierCV
# from skopt import BayesSearchCV
from scipy.stats import uniform, randint
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
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
# from lightgbm import LGBMError
from catboost import CatBoostRegressor

manual_variable_initialization(True)
# Import utilities
import logging
from util_tools.logging_config import LoggerSetup
from util_tools.model_classes import CustomStackingRegressor, CustomReduceLROnPlateau, WithinRangeMetric, LoggingEstimator, EarlyStoppingCallback
from util_tools.model_functions import (create_enhanced_neural_network, prepare_data,
        prepare_new_data, within_range_evaluation, perform_feature_selection, calculate_poisson_deviance, analyze_goals_difference,
        calculate_calibration_score, calculate_expected_value, evaluate_model, calculate_prediction_intervals, analyze_ensemble_diversity, apply_dynamic_weights,
        create_lgb_regressor, create_xgb_regressor, create_nn_regressor, create_rf_regressor, create_ada_regressor, create_elasticnet_regressor, create_gbm_regressor, optimize_hyperparameters)

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
recent_performance = pd.DataFrame(columns=['prediction', 'actual', 'abs_error'])


def enhance_predictions(base_predictions, model, recent_performance, data, logger):
    """
    Enhance predictions with uncertainty estimates and ensemble analysis.
    
    Args:
        base_predictions: Original model predictions
        model: Trained model
        data: Input data
        
    Returns:
        Dictionary containing enhanced predictions and metrics
    """
    # Calculate prediction intervals
    prediction_intervals = calculate_prediction_intervals(model, data)
    
    # Analyze ensemble diversity
    diversity_scores = analyze_ensemble_diversity(model.estimators_)  
    
    # Apply dynamic weighting
    weighted_predictions = apply_dynamic_weights(base_predictions, recent_performance)
    
    return {
        'predictions': weighted_predictions,
        'confidence': prediction_intervals['confidence'],
        'uncertainty': prediction_intervals['uncertainty'],
        'diversity_scores': diversity_scores
    }

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
    global recent_performance
    global selected_features
    global numeric_features
    global X_train, X_test, y_train, y_test
    
    # Select all numeric features
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    logger.info(f"Numeric features: {numeric_features}")
    
    # data = enhance_data_quality(data, logger)
    
    X = prepare_data(data, numeric_features, model_type, model_dir, logger)
    y = base_data[model_type]  # Target variable
    logger.info(f"Data prepared for modeling. Feature shape: {X.shape}, Target shape: {y.shape}")

    # Generate polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    logger.info("Polynomial feature transformation started.")
    X_poly = poly.fit_transform(X)
    
    # Optimize hyperparameters
    # best_params = optimize_hyperparameters(X_poly, y, logger)
    
    # # Update model parameters with optimized values
    # logger.info(f"Applying optimized hyperparameters: {list(best_params.keys())}")
    # for param, value in best_params.items():
    #     model_name, param_name = param.split('__')
    #     if model_name in stacking_regressor.named_estimators_:
    #         setattr(stacking_regressor.named_estimators_[model_name], param_name, value)
    #         logger.info(f"Setting {model_name} parameter {param_name} to {value}")
    #     elif model_name == 'final_estimator':
    #         setattr(stacking_regressor.final_estimator_, param_name, value)
    #         logger.info(f"Setting final estimator parameter {param_name} to {value}")
    
        
    # # Train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
    
    # # Second fit on a different random split (or different configuration)
    # X_train2, X_test2, y_train2, y_test2 = train_test_split(X_poly, y, test_size=0.3, random_state=123)
    
    # Before KFold splitting, reset the index of your data
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    # Use stratified k-fold cross validation for more robust model evaluation
    n_splits = 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize arrays to store multiple fold results
    X_trains, X_tests = [], []
    y_trains, y_tests = [], []
    
    # Split data into multiple folds
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_poly)):
        X_trains.append(X_poly[train_idx])
        X_tests.append(X_poly[test_idx]) 
        y_trains.append(y.iloc[train_idx])  # Use iloc instead of direct indexing
        y_tests.append(y.iloc[test_idx])    # Use iloc instead of direct indexing
        logger.info(f"Fold {fold+1} - Train shape: {X_trains[-1].shape}, Test shape: {X_tests[-1].shape}")
   
    # Use first two folds for dual fitting approach
    X_train, X_test = X_trains[0], X_tests[0]
    y_train, y_test = y_trains[0], y_tests[0]
    X_train2, X_test2 = X_trains[1], X_tests[1] 
    y_train2, y_test2 = y_trains[1], y_tests[1]
    
    logger.info("K-fold cross validation splits created for enhanced model validation")
    logger.info(f"Data split into train and test. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
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
    
    # Load the RFE selector for future use
    selector_file = os.path.join(model_dir, f'rfe_{model_type}_selector.pkl')
    selector = joblib.load(selector_file)
    logger.info(f"RFE selector loaded from {selector_file}")

    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    X_train2_selected = selector.transform(X_train2_scaled)
    X_test2_selected = selector.transform(X_test2_scaled)

    # Log selected features
    # selected_features = list(X_train.columns[selector.support_])
    selected_features = list(poly.get_feature_names_out(X.columns)[selector.support_])
    logger.info("Feature selection using RFE completed.")
    logger.info(f"Selected Features: {selected_features}")
    
    # Define models for home goals prediction based on research for soccer prediction
    lgb_regressor = create_lgb_regressor()

    xgb_regressor = create_xgb_regressor()

    nn_regressor = create_nn_regressor(X_train_selected)

    rf_regressor = create_rf_regressor()

    ada_regressor = create_ada_regressor()

    gbm_regressor = create_gbm_regressor()  

    elasticnet_regressor = create_elasticnet_regressor()
    
    estimators = [
        ('lgb', LoggingEstimator(lgb_regressor, 'LightGBM', logger)),
        ('xgb', LoggingEstimator(xgb_regressor, 'XGBoost', logger)),
        ('nn', LoggingEstimator(nn_regressor, 'Neural Network', logger)),
        ('rf', LoggingEstimator(rf_regressor, 'Random Forest', logger)),
        ('ada', LoggingEstimator(ada_regressor, 'AdaBoost', logger)),
        # Add to estimators with logging
        ('gbm', LoggingEstimator(gbm_regressor, 'GradientBoosting', logger)),
        ('elasticnet', LoggingEstimator(elasticnet_regressor, 'ElasticNet', logger)),
    ]

    # 2. Implement a more sophisticated final estimator
    final_estimator = VotingRegressor([
        ('ridge', Ridge(alpha=0.1)),
        ('lasso', Lasso(alpha=0.1)),
        ('elasticnet', ElasticNet(alpha=0.1, l1_ratio=0.5))
    ])
    
    stacking_regressor_home = StackingRegressor(estimators=estimators, final_estimator=Ridge())
    
    # Perform two separate fits
    logger.info('First fit started')
    stacking_regressor_home.fit(X_train_selected, y_train)
    logger.info(f"Stacking model for {model_type} trained successfully on first split.")
    
    # Evaluation on the first test set
    y_pred_home = stacking_regressor_home.predict(X_test_selected)
    
    evaluation_metrics = evaluate_model(y_test, y_pred_home, logger)
    
    # Get recent performance data
    recent_performance = pd.DataFrame({
        'prediction': y_pred_home,  # Last 100 predictions
        'actual': y_test,  # Last 100 actual values
        'abs_error': np.abs(y_pred_home - y_test)
    })
    # logger.info(f"Evaluation metrics for {model_type} (1st fit): {evaluation_metrics}")
    
    # mse_home = mean_squared_error(y_test, y_pred_home)
    # r2_home = r2_score(y_test, y_pred_home)
    # mae_home = mean_absolute_error(y_test, y_pred_home)
    # mape_home = np.mean(np.abs((y_test - y_pred_home) / y_test)) * 100
    # within_range_home = within_range_evaluation(y_test, y_pred_home, tolerance=0.5)  # Convert to percentage

    # logger.info(f"{model_type} (1st fit) Stacking Model MSE: {mse_home}, R2: {r2_home}, Stacking Model MAE: {mae_home}, Stacking Model MAPE: {mape_home}%")
    # logger.info(f"{model_type} (1st fit) Stacking Model Within Range (±0.5): {within_range_home}%")
    
    # 2nd Fit of the model
    logger.info('Second fit started')
    stacking_regressor_home.fit(X_train2_selected, y_train2)
    logger.info(f"Stacking model for {model_type} trained successfully on second split.")
    
    # Evaluation on the second test set
    y_pred_home2 = stacking_regressor_home.predict(X_test2_selected)
    
    # Get recent performance data
    recent_performance = pd.DataFrame({
        'prediction': y_pred_home2,  # Last 100 predictions
        'actual': y_test2,  # Last 100 actual values
        'abs_error': np.abs(y_pred_home2 - y_test2)
    })
    logger.info(f"Recent performance data: {recent_performance}")
    
    evaluation_metrics = evaluate_model(y_test2, y_pred_home2, logger)
    # logger.info(f"Evaluation metrics for {model_type} (2nd fit): {evaluation_metrics}")
    
    # mse_home2 = mean_squared_error(y_test2, y_pred_home2)
    # r2_home2 = r2_score(y_test2, y_pred_home2)
    # mae_home2 = mean_absolute_error(y_test2, y_pred_home2)
    # mape_home2 = np.mean(np.abs((y_test2 - y_pred_home2) / y_test2)) * 100
    # within_range_home2 = within_range_evaluation(y_test2, y_pred_home2, tolerance=0.5)  # Convert to percentage

    # logger.info(f"{model_type} (2nd fit) Stacking Model MSE: {mse_home2}, R2: {r2_home2}, Stacking Model MAE: {mae_home2}, Stacking Model MAPE: {mape_home2}%")
    # logger.info(f"{model_type} (2nd fit) Stacking Model Within Range (±0.5): {within_range_home2}%")
    
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
    global recent_performance
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
    base_predictions = model.predict(X_new_prepared)
    prediction_data[f'{model_type}_prediction_base'] = base_predictions
    try:
        # Calculate prediction intervals
        prediction_intervals = calculate_prediction_intervals(model, X_new_prepared, logger)
        prediction_data[f'{model_type}_confidence'] = prediction_intervals['confidence']
        prediction_data[f'{model_type}_uncertainty'] = prediction_intervals['uncertainty']
    except Exception as e:
        # Log the error and continue with base predictions only
        logger.error(f"Error occurred while calculate_prediction_intervals: {e}")
        prediction_data[f'{model_type}_confidence'] = 0.5  # Default confidence
        prediction_data[f'{model_type}_uncertainty'] = 1.0  # High uncertainty
    try:
        # Apply dynamic weighting
        weighted_predictions = apply_dynamic_weights(base_predictions, recent_performance, window=5, logger=logger)
        prediction_data[f'{model_type}_prediction_enhanced'] = weighted_predictions['weighted_predictions_positive']
        prediction_data[f'{model_type}_prediction_enhanced_negative'] = weighted_predictions['weighted_predictions_negative']
    except Exception as e:
        # Log the error and continue with base predictions only
        logger.error(f"Error occurred while apply_dynamic_weights: {e}")
        # Fallback to base predictions if enhancement fails
        prediction_data[f'{model_type}_prediction_enhanced'] = base_predictions
    
    # prediction_column_name = f'{model_type}_prediction'
    # logger.info(f"{model_type} predictions: {prediction}")

    # Add predictions to the new data DataFrame
    # prediction_data[prediction_column_name] = base_predictions

    
    # Save predictions to an Excel file
    output_file = f'./predictions_hybrid_2fit_{model_type}_enhanced.xlsx'
    prediction_data.to_excel(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")
    diversity_scores = analyze_ensemble_diversity(model.estimators_, X_new_prepared, logger)
    logger.info(f"Ensemble diversity scores: {diversity_scores}")

if __name__ == "__main__":
    prediction_df = new_prediction_data.drop(columns=['Unnamed: 0.1','Unnamed: 0.2','Date', 'Unnamed: 0','match_outcome', 'home_goals','away_goals', 'draw', 'away_win', 'home_win','away_points', 'home_points','HomeTeam_last_away_match','AwayTeam_last_home_match','home_points_rolling_avg','away_points_rolling_avg','home_advantage'], errors='ignore')
    prediction_df = prediction_df.replace(',', '.', regex=True)
    prediction_df = prediction_df.apply(pd.to_numeric, errors='coerce')
    prediction_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    logger.info(f"prediction_df length: {len(prediction_df)}")
    stacking_regressor = train_model(base_data, data, model_type, model_dir, logger)


    # MAKE PREDICTIONS
    try:
        logger.info(f"Start {model_type} predictions")
        make_prediction(model_type, stacking_regressor, prediction_df, model_dir, logger, numeric_features)          
    except Exception as e:
        logger.error(f"Error occurred while making prediction: {e}")
        pass

    # Initialize the wrappers
    try:
        nn_model = stacking_regressor.named_estimators_['nn'].model_
        if nn_model is None:
            raise AttributeError("The underlying model is not accessible.")
        
        # Save the neural network model separately first
        nn_model.save(keras_nn_model_path, save_format='h5')
        logger.info(f"Neural network model saved to {keras_nn_model_path}")
        
    except AttributeError as e:
        logger.error(f"Error occurred while accessing the model: {e}")
        raise

    # Create custom stacking regressor with saved model paths
    custom_model = CustomStackingRegressor(stacking_regressor, 
                                       nn_model,
                                       keras_nn_model_path)

    # Save the complete stacking model
    custom_model.save(model_file)
    logger.info(f"Complete stacking model saved to {model_file}")



# def load_real_scores():
#     # Load previously predicted scores
#     try:
#         real_scores = pd.read_excel(real_scores_path)
#         real_scores = real_scores[real_scores['Date'] < '2024-11-06']
#         real_scores = real_scores.dropna(subset=['real_score'])
#         real_scores.replace([np.inf, -np.inf], np.nan, inplace=True)
#         logger.info(f"Previously predicted scores loaded from {real_scores_path} with shape: {real_scores.shape}")
#         # print(real_scores.head())
#     except Exception as e:
#         logger.error(f"Error loading previously predicted scores: {e}")
#         raise

# # Prepare real scores data for additional fit
# def prepare_real_scores_data(real_scores: pd.DataFrame, new_prediction_data: pd.DataFrame, base_data: pd.DataFrame, model_type: str, original_features: list) -> pd.DataFrame:
#     """
#     Prepare the real scores data for additional model fitting by merging with new prediction data.

#     Args:
#         real_scores (pd.DataFrame): The DataFrame containing real scores.
#         new_prediction_data (pd.DataFrame): The DataFrame containing new prediction data with all features.
#         model_type (str): The type of model to train.
#         original_features (list): The list of original features used in the model.

#     Returns:
#         pd.DataFrame: The prepared feature data.
#     """
    
#     df_for_merge = new_prediction_data.drop(columns=['home_points','HomeTeam_last_away_match','AwayTeam_last_home_match',
#                                                     'home_poisson_xG','away_poisson_xG',
#                                                     'Odd_Home','Odds_Draw','Odd_Away','Unnamed: 0.1', 'Unnamed: 0','Date'], errors='ignore')
#     # Merge real scores with new prediction data on common keys
#     merged_data = pd.merge(
#         real_scores,
#         df_for_merge,
#         on=['running_id'],  # Adjust these keys as necessary
#         how='left'
#     )
#     merged_data[model_type] = merged_data['real_score'].str[2]
  
#     merged_data = merged_data.dropna(subset=[model_type])
    
#     logger.info(f"merged data columns: {merged_data.columns.to_list()}")
    
#     # Convert comma to dot for decimal conversion
#     real_scores_data = merged_data.replace(',', '.', regex=True)
#     real_scores_data = real_scores_data.apply(pd.to_numeric, errors='coerce')
#     real_scores_data.replace([np.inf, -np.inf], np.nan, inplace=True)
#     logger.info(f"Real scores data prepared for additional fit. Shape: {real_scores_data.shape}")
#     logger.debug(f"Real scores data columns: {real_scores_data.columns.tolist()}")
#     print(real_scores_data.head())
#     real_scores_data.to_excel('./real_scores_data.xlsx', index=False)
#     # logger.debug(f"Real scores data shape: {real_scores_data.shape}")
#     return real_scores_data

# # Prepare the real scores data
# # real_scores_data = prepare_real_scores_data(real_scores, new_prediction_data, base_data, model_type, selected_features)

# # Additional fit using real scores
# def additional_fit_with_real_scores(model: StackingRegressor, real_scores_data: pd.DataFrame, model_type: str, logger: logging.Logger):
#     """
#     Perform an additional fit using the real scores data, only for incorrect predictions.

#     Args:
#         model (StackingRegressor): The trained stacking regressor model.
#         real_scores_data (pd.DataFrame): The feature data for real scores.
#         real_scores (pd.DataFrame): The DataFrame containing real scores.
#         model_type (str): The type of model used for prediction.
#         logger (logging.Logger): Logger for logging information.
#     """
#     try:
#         # Load and apply the saved imputer from training
#         selector_file = os.path.join(model_dir, f'rfe_{model_type}_selector.pkl')
#         selector = joblib.load(selector_file)
#         logger.info(f"Selector loaded from {selector_file}")
#     except FileNotFoundError:
#         logger.error(f"Selector file not found at {selector_file}")
#         raise
#     try:
#         # Load and apply the saved scaler from training
#         scaler_file = os.path.join(model_dir, f'scaler_{model_type}.pkl')
#         scaler_loaded = joblib.load(scaler_file)
#     except FileNotFoundError:
#         logger.error(f"Scaler file not found at {scaler_file}")
#         raise
    
#     # Prepare the data
#     X_real = prepare_data(real_scores_data, numeric_features, model_type, model_dir, logger)
#     # Scale the data
#     X_real_scaled = scaler_loaded.transform(X_real)

#     # Feature selection
#     X_real_selected = selector.transform(X_real_scaled)
    
#     # Use the actual real score as the target variable
#     y_real = real_scores_data[model_type]  # Adjust this to the correct column name for real scores

#     # Predict using the current model
#     y_pred = model.predict(X_real_selected)

#     # Identify incorrect predictions
#     incorrect_indices = y_pred != y_real

#     # Filter data for incorrect predictions
#     X_incorrect = X_real[incorrect_indices]
#     y_incorrect = y_real[incorrect_indices]

#     if len(X_incorrect) > 0:
#         # Scale the data
#         X_incorrect_scaled = scaler_loaded.transform(X_incorrect)

#         # Feature selection
#         X_incorrect_selected = selector.transform(X_incorrect_scaled)

#         # Fit the model with incorrect predictions
#         logger.info('Additional fit with real scores started for incorrect predictions')
#         model.fit(X_incorrect_selected, y_incorrect)
#         logger.info(f"Stacking model for {model_type} trained successfully with real scores for incorrect predictions.")
#     else:
#         logger.info("No incorrect predictions found; no additional fitting needed.")
