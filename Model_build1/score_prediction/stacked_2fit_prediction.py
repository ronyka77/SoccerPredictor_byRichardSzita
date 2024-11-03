import os
import tempfile
import shutil
# Create a temporary directory under your project directory
# Create a temporary directory
tempdir = tempfile.mkdtemp(dir='./tempdir')
os.makedirs(tempdir, exist_ok=True)
# Clean up the temporary directory before starting TensorFlow
if os.path.exists(tempdir):
    shutil.rmtree(tempdir)
os.makedirs(tempdir, exist_ok=True)

os.environ['TF_CONFIG'] = tempdir 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # To suppress detailed logs
os.environ['TEMP'] = tempdir
os.environ['TMP'] = tempdir
os.environ['TMPDIR'] = tempdir
print(os.access(tempdir, os.W_OK))  # Check if writable
print(os.access(tempdir, os.R_OK))  # Check if readable

import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from scikeras.wrappers import KerasRegressor  # Ensure compatibility with Keras
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import RFE
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2   
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras.backend as K
import h5py
import dill
from keras.backend import manual_variable_initialization 
manual_variable_initialization(True)


# Define the within_range function (if needed)
def within_range(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    within_range = K.less_equal(diff, 0.3)
    return K.reduce_mean(K.cast(within_range, K.float32))

# Define the neural network architecture
def create_neural_network(input_dim):
    model = Sequential()
    
    # Increase neurons and add regularization
    model.add(Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile with MSE loss and Adam optimizer
    model.compile(loss='mse', optimizer=Adam(), metrics=['mae', within_range])
    
    return model

# Define directories
model_dir = './models/saved_models/stacked_2fit_models/'
model_type = 'match_outcome'  # Change this if your model_type is different
new_prediction_path = './Model_build1/data/merged_data_prediction.csv'

# Load the stacking model (without Keras)
model_file = os.path.join(model_dir, f"model_stacked_2fit{model_type}.pkl")
stacking_regressor = joblib.load(model_file)  # Load the stacking model

# Load the Keras neural network separately
keras_nn_model_path = os.path.join(model_dir, 'nn_regressor_'+ model_type +'_stacked_2fit')
keras_nn_model = load_model(keras_nn_model_path, custom_objects={'within_range': within_range})  # Load Keras model with custom metric


# Define base models for stacking, including the neural network (nn)
estimators = [
    ('rf', stacking_regressor.named_estimators_['rf']),  # Assuming RF was in the original stack
    ('svr', stacking_regressor.named_estimators_['svr']),  # Assuming SVR was in the original stack
    ('xgb', stacking_regressor.named_estimators_['xgb']),  # Assuming XGB was in the original stack
    ('nn', keras_nn_model)  # Reattach the Keras model
]

# Define the final estimator (Ridge or any other final model you used)
final_estimator = stacking_regressor.final_estimator_

# Rebuild the stacking regressor
stacking_regressor_rebuilt = StackingRegressor(estimators=estimators, final_estimator=final_estimator)

# Prepare the new data for prediction
# Load the new data (Ensure it matches the format of your training data)
prediction_data = pd.read_csv(new_prediction_path)

# Load the saved scaler and feature selector (assumed saved during training)
scaler_file = os.path.join(model_dir, 'scaler_' + model_type + '.pkl')
scaler = joblib.load(scaler_file)

selector_file = os.path.join(model_dir, 'rfe_'+ model_type +'_selector.pkl')
selector = joblib.load(selector_file)

# Preprocess the new data
X_new_scaled = scaler.transform(prediction_data)  # Apply the scaler
X_new_selected = selector.transform(X_new_scaled)  # Select features using RFE

# Make predictions with the rebuilt stacking model
predictions = stacking_regressor_rebuilt.predict(X_new_selected)

# Save the predictions to an Excel file
prediction_output = pd.DataFrame(predictions, columns=['predicted_goals'])
prediction_output.to_excel('predicted_goals_output.xlsx', index=False)

print("Predictions saved successfully.")
