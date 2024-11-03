import os
import cloudpickle as cp
import joblib
import numpy as np
from keras.callbacks import Callback
import keras.backend as K
from keras.backend import manual_variable_initialization
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.metrics import MeanMetricWrapper
import pandas as pd
manual_variable_initialization(True)

# Paths to the saved models and transformers
model_dir = './models/'
model_type = 'match_outcome'
model_file = os.path.join(model_dir, 'model_stacked_2fit_' + model_type + '.pkl')
keras_nn_model_path = os.path.join(model_dir, 'nn_regressor_' + model_type + '_stacked_2fit.h5')
imputer_file = os.path.join(model_dir, 'imputer_' + model_type + '.pkl')
scaler_file = os.path.join(model_dir, 'scaler_' + model_type + '.pkl')
selector_file = os.path.join(model_dir, 'rfe_' + model_type + '_selector.pkl')
new_prediction_path = './Model_build1/data/merged_data_prediction.csv'

# Ensure custom objects have the proper config methods
class CustomReduceLROnPlateau(Callback):
    def __init__(self, monitor='loss', factor=0.5, patience=10, verbose=0, min_lr=0.0001):
        super(CustomReduceLROnPlateau, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        self.best = np.Inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            print(f"Warning: CustomReduceLROnPlateau requires {self.monitor} to be available!")
            return

        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(K.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    K.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch + 1}: ReduceLROnPlateau reducing learning rate to {new_lr}.')
                self.wait = 0

    def get_config(self):
        return {
            'monitor': self.monitor,
            'factor': self.factor,
            'patience': self.patience,
            'verbose': self.verbose,
            'min_lr': self.min_lr
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CustomStackingRegressor:
    def __init__(self, stacking_regressor, keras_model, keras_model_path):
        self.stacking_regressor = stacking_regressor
        self.keras_model = keras_model
        self.keras_model_path = keras_model_path
    
    def save(self, model_path):
        # Save Keras model separately
        self.keras_model.save(self.keras_model_path, include_optimizer=True)
        
        # Remove Keras model from stacking regressor for pickling
        self.stacking_regressor.named_estimators_.pop('nn', None)
        
        # Save the rest of the stacking regressor
        with open(model_path, 'wb') as f:
            cp.dump(self.stacking_regressor, f)
        
        # Reassign Keras model for in-memory use
        self.stacking_regressor.named_estimators_['nn'] = self.keras_model
    
    @classmethod
    def load(cls, model_path, keras_model_path, custom_objects=None):
        # Load the stacking regressor
        with open(model_path, 'rb') as f:
            stacking_regressor = cp.load(f)
        
        # Load the Keras model
        keras_model = load_model(keras_model_path, custom_objects=custom_objects)
        
        # Reassign Keras model to stacking regressor
        stacking_regressor.named_estimators_['nn'] = keras_model
        
        # Return the wrapped object
        return cls(stacking_regressor, keras_model, keras_model_path)

# Define the function for the custom metric
def within_range_metric(y_true, y_pred):
    # Calculate the absolute difference between true and predicted values
    diff = K.abs(y_true - y_pred)
    # Check if the difference is less than or equal to 0.3
    within_range = K.less_equal(diff, 0.3)
    # Return the mean value (percentage of correct predictions within range)
    return K.mean(K.cast(within_range, K.floatx()))

# Wrap it using MeanMetricWrapper to be used as a Keras metric
WithinRangeMetric = MeanMetricWrapper(fn=within_range_metric, name='within_range_metric')

# Define custom objects for loading the model
custom_objects = {
    'within_range_metric': WithinRangeMetric,
    'WithinRangeMetric': WithinRangeMetric
}
prediction_data = pd.read_csv(new_prediction_path)
prediction_data.drop(columns=['Date','Datum','Unnamed: 0','Unnamed: 0.1','away_goals','away_points_rolling_avg','away_win','draw','home_goals','home_points','home_advantage','home_points_rolling_avg','home_win','match_outcome'], inplace=True, errors='ignore')

# Load the Keras neural network separately with custom objects
keras_nn_model = load_model(keras_nn_model_path, custom_objects=custom_objects)
# Reassign the custom metric
keras_nn_model.compile(optimizer=keras_nn_model.optimizer, loss=keras_nn_model.loss, metrics=[within_range_metric])
print("Keras neural network loaded successfully.")

# with open(model_file, 'rb') as f:
#     stacking_regressor = cp.load(f)

# print("Stacking regressor loaded successfully.")
# # Reinsert the Keras model into the stacking regressor
# stacking_regressor.named_estimators_['nn'] = keras_nn_model

# Load the pre-fitted imputer, scaler, and feature selector
imputer = joblib.load(imputer_file)
scaler = joblib.load(scaler_file)
selector = joblib.load(selector_file)

# Prepare new data for prediction
def prepare_new_data(new_data, imputer, scaler, selector):
    # Apply imputation
    model_data_imputed = imputer.transform(new_data)  # Use the saved imputer
    # Apply scaling
    model_data_scaled = scaler.transform(model_data_imputed)  # Use the saved scaler
    # Apply feature selection (RFE)
    model_data_selected = selector.transform(model_data_scaled)  # Use the saved selector
    return model_data_selected


# Prepare the new data for prediction
X_new_prepared = prepare_new_data(prediction_data, imputer, scaler, selector)

# Example: Making predictions with the Keras model
predictions = keras_nn_model.predict(X_new_prepared)

prediction_data['outcome_prediction'] = predictions
# Save the predictions to an Excel file
# prediction_output = pd.DataFrame(predictions, columns=['predicted_goals'])
prediction_data.to_excel('predicted_goals_output_CHECK.xlsx', index=False)
# Output the prediction
print(f"Predicted result: {predictions}")
