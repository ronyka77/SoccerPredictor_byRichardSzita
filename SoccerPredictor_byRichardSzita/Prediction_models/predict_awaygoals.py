import os
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from util_tools.logging_config import LoggerSetup
from model_stacked_2fit_awaygoals import CustomStackingRegressor, prepare_new_data, WithinRangeMetric

# Set up logger
logger = LoggerSetup.setup_logger(
    name='predictor_awaygoals',
    log_file='./log/predictor_awaygoals.log',
    level=logging.INFO
)

# Define paths
model_type = 'away_goals'
model_dir = "./models/"
keras_nn_model_path = os.path.join(model_dir, f'nn_regressor_{model_type}_stacked_2fit.h5')
model_file = os.path.join(model_dir, f'model_stacked_2fit_{model_type}.pkl')

# Define custom objects
custom_objects = {
    'WithinRangeMetric': WithinRangeMetric,
    'within_range_metric': within_range_metric  # If you have a function for the metric
}

# Load the trained model
try:
    custom_model = CustomStackingRegressor.load(model_file, keras_nn_model_path, custom_objects=custom_objects)
    logger.info(f"Model loaded successfully for {model_type}.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Load and prepare new data for prediction
new_data_path = './data/merged_data_prediction.csv'
new_data = pd.read_csv(new_data_path)
logger.info(f"New data loaded from {new_data_path} with shape: {new_data.shape}")

# Prepare the new data
imputer_file = os.path.join(model_dir, f'imputer_{model_type}.pkl')
selector_file = os.path.join(model_dir, f'rfe_{model_type}_selector.pkl')

try:
    imputer = joblib.load(imputer_file)
    selector = joblib.load(selector_file)
    logger.info(f"Imputer and selector loaded successfully for {model_type}.")
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise

X_new_prepared = prepare_new_data(new_data, imputer, selector)

# Make predictions
predictions = custom_model.stacking_regressor.predict(X_new_prepared)
logger.info(f"Predictions made: {predictions}")

# Save predictions to a file
output_file = f'./predictions_{model_type}_new.xlsx'
pd.DataFrame(predictions, columns=[f'{model_type}_prediction']).to_excel(output_file, index=False)
logger.info(f"Predictions saved to {output_file}") 