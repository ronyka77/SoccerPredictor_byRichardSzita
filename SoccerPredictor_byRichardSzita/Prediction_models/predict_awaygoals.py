import os
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from util_tools.logging_config import LoggerSetup
from util_tools.model_classes import CustomStackingRegressor, CustomReduceLROnPlateau, WithinRangeMetric, LoggingEstimator
from util_tools.model_functions import create_neural_network, prepare_data, prepare_new_data
import cloudpickle
from keras.metrics import Metric

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

logger.info(f"Model paths set: keras_nn_model_path={keras_nn_model_path}, model_file={model_file}")

# Test serialization and deserialization
def test_serialization():
    # Create an instance of the custom metric
    original_metric = WithinRangeMetric()

    # Serialize the object
    with open('metric.pkl', 'wb') as f:
        cloudpickle.dump(original_metric, f)

    # Deserialize the object
    with open('metric.pkl', 'rb') as f:
        loaded_metric = cloudpickle.load(f)

    # Verify that the deserialized object is the same as the original
    assert original_metric.get_config() == loaded_metric.get_config(), "Config mismatch after deserialization"

    print("Serialization and deserialization test passed.")

class CustomMetric(Metric):
    def get_config(self):
        # Return a dictionary of the configuration
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        # Create an instance from the configuration
        return cls(**config)
    
# Run the test
test_serialization()

# Clean up
os.remove('metric.pkl')

# Define custom objects
custom_objects = {
    'WithinRangeMetric': WithinRangeMetric,
    'within_range_metric': WithinRangeMetric(),
    'CustomReduceLROnPlateau': CustomReduceLROnPlateau,
    '_tf_keras_metric': CustomMetric,
    'CustomStackingRegressor': CustomStackingRegressor,
    'LoggingEstimator': LoggingEstimator,
    # Add any other necessary custom objects here
}

logger.info("Custom objects defined for model loading.")

# Load the model
try:
    logger.info("Attempting to load the trained model.")
    custom_model = CustomStackingRegressor.load(model_file, keras_nn_model_path, custom_objects=custom_objects)
    logger.info(f"Model loaded successfully for {model_type}.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Load and prepare new data for prediction
new_data_path = './data/merged_data_prediction.csv'
logger.info(f"Loading new data from {new_data_path}.")
new_data = pd.read_csv(new_data_path)

# Drop unnecessary columns as done in training
new_data = new_data.drop(columns=['Unnamed: 0.1','Unnamed: 0.2','Date', 'Unnamed: 0','match_outcome', 
                                 'home_goals','away_goals', 'draw', 'away_win', 'home_win',
                                 'away_points', 'home_points','HomeTeam_last_away_match',
                                 'AwayTeam_last_home_match','home_points_rolling_avg',
                                 'away_points_rolling_avg','home_advantage'], errors='ignore')

# Convert data types and handle infinities
new_data = new_data.replace(',', '.', regex=True)
new_data = new_data.apply(pd.to_numeric, errors='coerce')
new_data.replace([np.inf, -np.inf], np.nan, inplace=True)

logger.info(f"New data loaded and preprocessed with shape: {new_data.shape}")

# Load preprocessing objects
imputer_file = os.path.join(model_dir, f'imputer_{model_type}.pkl')
selector_file = os.path.join(model_dir, f'rfe_{model_type}_selector.pkl')
scaler_file = os.path.join(model_dir, f'scaler_{model_type}.pkl')

logger.info("Loading preprocessing objects...")
try:
    imputer = joblib.load(imputer_file)
    selector = joblib.load(selector_file)
    scaler = joblib.load(scaler_file)
    logger.info("Preprocessing objects loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise

# Select numeric features
numeric_features = new_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
logger.info(f"Selected {len(numeric_features)} numeric features")

# Prepare data with polynomial features
logger.info("Preparing new data for prediction...")
X = prepare_data(new_data, numeric_features, model_type, model_dir, logger)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Scale the data
X_scaled = scaler.transform(X_poly)

# Apply feature selection
X_selected = selector.transform(X_scaled)
logger.info(f"Data prepared with final shape: {X_selected.shape}")

# Make predictions
logger.info("Making predictions with the loaded model...")
predictions = custom_model.predict(X_selected)
logger.info(f"Predictions made with shape: {predictions.shape}")

# Add predictions to original data and save
new_data[f'{model_type}_prediction'] = predictions
output_file = f'./made_predictions/predictions_{model_type}_new.xlsx'
logger.info(f"Saving predictions to {output_file}")
new_data.to_excel(output_file, index=False)
logger.info(f"Predictions saved successfully to {output_file}")






