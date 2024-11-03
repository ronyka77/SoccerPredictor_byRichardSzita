import os
import tempfile
import shutil
from copy import deepcopy
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
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2   
from keras.callbacks import EarlyStopping, Callback
import keras.backend as K
from keras.metrics import MeanMetricWrapper
import h5py
from keras.backend import manual_variable_initialization 
import cloudpickle as cp
import dill
manual_variable_initialization(True)

model_type='home_goals'
# Set up logging
log_file_path = './Model_build1/score_prediction/log/stacked_' + model_type + '_model_training.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set directories and files
model_dir = "./models/"
os.makedirs(model_dir, exist_ok=True)
logging.info("Model directory checked/created.")

# Define model file paths
keras_nn_model_path = os.path.join(model_dir, 'nn_regressor_' + model_type + '_stacked_2fit.h5')
model_file = os.path.join(model_dir, 'model_stacked_2fit_' + model_type + '.pkl')

scaler = StandardScaler()
selected_features = []
numeric_features = []

# Load your feature-engineered data
base_data_path = './Model_build1/data/model_data_training_newPoisson.xlsx'
new_prediction_path = './Model_build1/data/merged_data_prediction.csv'
real_scores_path = './Model_build1/made_predictions/predictions_stacked_2fit_merged.xlsx'
base_data = pd.read_excel(base_data_path)
logging.info(f"Data loaded from {base_data_path} with shape: {base_data.shape}")

# Load data for prediction
new_prediction_data = pd.read_csv(new_prediction_path)
logging.info(f"Data loaded from {new_prediction_path} with shape: {new_prediction_data.shape}")
logging.info(f"Data loaded with {len(new_prediction_data)} rows")

# Filter data to only include away_goals from 0 to 5
base_data = base_data[(base_data['home_goals'] >= 0) & (base_data['home_goals'] <= 6) & (base_data['away_goals'] >= 0) & (base_data['away_goals'] <= 6) ]
base_data = base_data.dropna()
logging.info(f"Data filtered to only include away_goals from 0 to 6. rows: {len(base_data)} Filtered data shape: {base_data.shape}")

# Drop unnecessary columns
data = base_data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0','match_outcome', 'home_goals','away_goals',  
                               'draw', 'away_win', 'home_win','away_points', 'home_points','HomeTeam_last_away_match','AwayTeam_last_home_match',
                               'home_points_rolling_avg','away_points_rolling_avg','home_advantage'], errors='ignore')

# Import predicted scores
new_real_scores = pd.read_excel(real_scores_path)
new_real_scores.dropna(subset=['running_id','home_goals_prediction_rounded','away_goals_prediction_rounded','match_outcome_prediction_rounded','real_score'],inplace=True)
logging.info(f"new real scores length: {len(new_real_scores)}")

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

class CustomReduceLROnPlateau(Callback):
    def __init__(self, monitor='loss', factor=0.5, patience=10, verbose=0, min_lr=0.0001):
        super(CustomReduceLROnPlateau, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr

        self.best = None
        self.cooldown_counter = 0  # Cooldown counter
        self.wait = 0  # Wait counter
        self.monitor_op = None

        # Set the comparison operation for monitoring loss
        self.monitor_op = lambda a, b: np.less(a, b)
        self.best = np.Inf  # Initialize with the worst possible value for loss (Inf)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            print(f"Warning: CustomReduceLROnPlateau requires {self.monitor} to be available!")
            return

        if self.monitor_op(current, self.best):
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
        config = {
            'monitor': self.monitor,
            'factor': self.factor,
            'patience': self.patience,
            'verbose': self.verbose,
            'min_lr': self.min_lr,
        }
        return config

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
   
# Prepare data
def prepare_data(data, features, model_type):
   
    model_data = data.replace(',', '.', regex=True)
    model_data = model_data.apply(pd.to_numeric, errors='coerce')
    model_data = model_data[features]
    
    # Apply Iterative Imputation
    imputer = IterativeImputer(random_state=42)
    model_data_imputed = imputer.fit_transform(model_data)
    # Save the imputer
    imputer_file = os.path.join(model_dir, 'imputer_'+ model_type + '.pkl')
    joblib.dump(imputer, imputer_file)
    logging.info(f"Imputer saved to {imputer_file}")

    return pd.DataFrame(model_data_imputed, columns=features)

# Merge base data with predicted values
def add_predicted_values(new_real_scores,data):
    # Split the column into two based on '-'
    new_real_scores[['real_home_goals', 'real_away_goals']] = new_real_scores['real_score'].str.split('-', expand=True)
    # logging.info(f"new_scores_df: {new_real_scores}")
    new_real_scores.dropna(subset=['real_home_goals','real_away_goals'],inplace=True)
    new_real_scores['real_away_goals'] = new_real_scores['real_away_goals'].astype(int)
    new_real_scores['real_home_goals'] = new_real_scores['real_home_goals'].astype(int)
    new_real_scores['running_id'] = new_real_scores['running_id'].astype(int)
    # Calculate residuals (prediction errors)
    new_real_scores['home_error'] = new_real_scores['real_home_goals'] - new_real_scores['home_goals_prediction_rounded']
    new_real_scores['away_error'] = new_real_scores['real_away_goals'] - new_real_scores['away_goals_prediction_rounded']
    new_real_scores['outcome_error'] = new_real_scores['real_outcome'] - new_real_scores['match_outcome_prediction_rounded']
    # Calculate the mean squared error for home and away predictions
    mse_home_prediction = mean_squared_error(new_real_scores['real_home_goals'], new_real_scores['home_goals_prediction_rounded'])
    mse_away_prediction = mean_squared_error(new_real_scores['real_away_goals'], new_real_scores['away_goals_prediction_rounded'])
    mse_outcome_prediction = mean_squared_error(new_real_scores['real_outcome'], new_real_scores['match_outcome_prediction_rounded'])

    # Calculate R-squared score for home and away predictions
    r2_home_prediction = r2_score(new_real_scores['real_home_goals'], new_real_scores['home_goals_prediction_rounded'])
    r2_away_prediction = r2_score(new_real_scores['real_away_goals'], new_real_scores['away_goals_prediction_rounded'])
    r2_outcome_prediction = r2_score(new_real_scores['real_outcome'], new_real_scores['match_outcome_prediction_rounded'])
    
    logging.info(f"Home MSE: {mse_home_prediction}, Away MSE: {mse_away_prediction}")
    logging.info(f"Home R2: {r2_home_prediction}, Away R2: {r2_away_prediction}")
    logging.info(f"Outcome MSE: {mse_outcome_prediction}, Outcome R2: {r2_outcome_prediction}")
    
    score_df = new_real_scores[['running_id','home_goals_prediction_rounded','away_goals_prediction_rounded','match_outcome_prediction_rounded','real_home_goals','real_away_goals','real_outcome','home_error','away_error','outcome_error']]
    # logging.info(f"New score dataframe created with errors: {score_df.head(5)}")
    # print('Score_df: ' + str(len(score_df)))
    # Merge the predictions with the real scores
    merged_data = pd.merge(data, score_df, how='right', on='running_id')
    # numeric_features = merged_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # merged_data = merged_data[numeric_features]
    print('merge_df: ' + str(len(merged_data)))
    return merged_data

def prepare_new_data(new_data, imputer, selector):
    global selected_features
    global numeric_features
    
    model_data = new_data.replace(',', '.', regex=True)
    model_data = model_data.apply(pd.to_numeric, errors='coerce')
    logging.info(f"Prediction Selected Features: {numeric_features}")
    scaler_file = os.path.join(model_dir, 'scaler_' + model_type + '.pkl')
    # numeric_features = new_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    model_data = model_data[numeric_features]
    scaler_loaded = joblib.load(scaler_file)
    
    # Apply imputation
    model_data_imputed = imputer.transform(model_data)  # Use the imputer you saved during training
   
    # Apply scaling
    model_data_scaled = scaler_loaded.transform(model_data_imputed)  # Use the scaler you saved during training
    
    # Apply feature selection (RFE)
    model_data_selected = selector.transform(model_data_scaled)  # Use the RFE selector saved during training
    
    # logging.info(model_data_selected)
    return pd.DataFrame(model_data_selected)

def within_range(y_true, y_pred):
    # Calculate the absolute difference between the predicted and true values
    diff = K.abs(y_true - y_pred)
    
    # Check if the difference is less than or equal to 0.3
    within_range = K.less_equal(diff, 0.3)
    
    # Calculate the mean of the boolean values (i.e., percentage of correct predictions)
    return K.mean(K.cast(within_range, K.floatx()))

def within_range_evaluation(y_true, y_pred, tolerance=0.3):
    """Calculate the percentage of predictions within the given tolerance range."""
    within_range_count = np.sum(np.abs(y_true - y_pred) <= tolerance)
    return within_range_count / len(y_true)

# Define the neural network architecture
def create_neural_network(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer

    # Compile with MSE loss and Adam optimizer
    model.compile(loss='mse', optimizer=Adam(), metrics=['mae', WithinRangeMetric])
    return model

# Train the model
def train_model(base_data, data, model_type):
    global selected_features
    global numeric_features
    # # Count NaN values in each column
    # nan_counts = data.isna().sum()
    # print(nan_counts)
    # Select all numeric features
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    logging.info(f"Numeric features: {numeric_features}")

    X = prepare_data(data, numeric_features, model_type)
    y = base_data[model_type]  # Target variable
    logging.info(f"Data prepared for modeling. Feature shape: {X.shape}, Target shape: {y.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Data split into train and test. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    # Second fit on a different random split (or different configuration)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=123)
    logging.info(f"Data split into train2 and test2. Train2 shape: {X_train2.shape}, Test2 shape: {X_test2.shape}")

    # Scaling
    logging.info("Data scaling started.")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Save the scaler for future use
    scaler_file = os.path.join(model_dir, 'scaler_' + model_type + '.pkl')
    joblib.dump(scaler, scaler_file)
    logging.info(f"Scaler saved to {scaler_file}")
    
    X_test_scaled = scaler.transform(X_test)
    X_train2_scaled = scaler.transform(X_train2)
    X_test2_scaled = scaler.transform(X_test2)
    logging.info("Data scaling completed.")

    # Feature Selection (RFE)
    logging.info("Feature Selection started.")
    selector = RFE(estimator=RandomForestRegressor(), n_features_to_select=30)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    
    # Save the RFE selector for future use
    selector_file = os.path.join(model_dir, 'rfe_' + model_type + '_selector.pkl')
    joblib.dump(selector, selector_file)
    logging.info(f"RFE selector saved to {selector_file}")
    
    X_test_selected = selector.transform(X_test_scaled)
    X_train2_selected = selector.transform(X_train2_scaled)
    X_test2_selected = selector.transform(X_test2_scaled)

    # Log selected features
    selected_features = list(X_train.columns[selector.support_])
    logging.info("Feature selection using RFE completed.")
    logging.info(f"Selected Features: {selected_features}")
    
    # Set early stopping and learning rate scheduler
    # early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    # lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.0001)
    
    # Define models for home goals prediction
    rf_regressor_home = RandomForestRegressor(n_estimators=200, random_state=42)
    svr_regressor_home = SVR(kernel='rbf')
    nn_regressor_home = KerasRegressor(
        model=create_neural_network,
        model__input_dim=X_train_selected.shape[1],
        epochs=100,  # Set higher epochs but use early stopping
        batch_size=32,
        verbose=1,
        callbacks=[CustomReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.0001)]  # LR Scheduler
    )
    xgb_regressor_home = XGBRegressor()

    # Stacking Regressor with Ridge as final estimator
    estimators_home = [
        ('rf', rf_regressor_home),
        ('svr', svr_regressor_home),
        ('nn', nn_regressor_home),
        ('xgb', xgb_regressor_home)
    ]
    logging.info('First fit started')
    stacking_regressor_home = StackingRegressor(estimators=estimators_home, final_estimator=Ridge())
    
    # Dual fitting approach - Perform two separate fits
    stacking_regressor_home.fit(X_train_selected, y_train)
    logging.info(f"Stacking model for {model_type} trained successfully on first split.")
    
    # Evaluation on the first test set
    y_pred_home = stacking_regressor_home.predict(X_test_selected)
    mse_home = mean_squared_error(y_test, y_pred_home)
    r2_home = r2_score(y_test, y_pred_home)
    mae_home = mean_absolute_error(y_test, y_pred_home)
    mape_home = np.mean(np.abs((y_test - y_pred_home) / y_test)) * 100
    within_range_home = within_range_evaluation(y_test, y_pred_home, tolerance=0.3) * 100  # Convert to percentage

    logging.info(f"{model_type} (1st fit) Stacking Model MSE: {mse_home}, R2: {r2_home}, Stacking Model MAE: {mae_home}, Stacking Model MAPE: {mape_home}%")
    logging.info(f"{model_type} (1st fit) Stacking Model Within Range (±0.3): {within_range_home}%")
    
    # 2nd Fit of the model
    logging.info('Second fit started')
    stacking_regressor_home.fit(X_train2_selected, y_train2)
    logging.info(f"Stacking model for {model_type} trained successfully on second split.")
    
    # Evaluation on the second test set
    y_pred_home2 = stacking_regressor_home.predict(X_test2_selected)
    mse_home2 = mean_squared_error(y_test2, y_pred_home2)
    r2_home2 = r2_score(y_test2, y_pred_home2)
    mae_home2 = mean_absolute_error(y_test2, y_pred_home2)
    mape_home2 = np.mean(np.abs((y_test2 - y_pred_home2) / y_test2)) * 100
    within_range_home2 = within_range_evaluation(y_test2, y_pred_home2, tolerance=0.3) * 100  # Convert to percentage

    logging.info(f"{model_type} (2nd fit) Stacking Model MSE: {mse_home2}, R2: {r2_home2}, Stacking Model MAE: {mae_home2}, Stacking Model MAPE: {mape_home2}%")
    logging.info(f"{model_type} (2nd fit) Stacking Model Within Range (±0.3): {within_range_home2}%")
    
    return stacking_regressor_home

# Make new predictions
def make_prediction(model_type, model, prediction_data, residual_model):
    # MAKE NEW PREDICTION
    # Directory where the models are saved
    imputer_file = f'imputer_{model_type}.pkl'  # Imputer file from training
    # Load and apply the saved imputer from training
    imputer_path = os.path.join(model_dir, imputer_file)
    selector_file = os.path.join(model_dir, 'rfe_'+ model_type +'_selector.pkl')
    
    selector = joblib.load(selector_file)
    try:
        imputer = joblib.load(imputer_path)
        logging.info(f"Imputer loaded from {imputer_path}")
    except FileNotFoundError:
        logging.error(f"Imputer file not found at {imputer_path}")
        raise
    
    X_new_prepared = prepare_new_data(prediction_data, imputer, selector)
    # logging.info(f"X_new_prepared: {X_new_prepared}")
    # Predict the home goals using the original model
    prediction = model.predict(X_new_prepared)
    prediction_column_name= model_type + '_prediction'
    logging.info(f"{model_type} predictions: {prediction}")
    # Add predictions to the new data DataFrame
    prediction_data[prediction_column_name] = prediction
    
    if residual_model != 0:
        try:    
            # Predict the home error using the residual model
            error_pred = residual_model.predict(X_new_prepared)
            logging.info(f"{model_type} error predictions: {error_pred}")
            # Final adjusted prediction for goals
            prediction_with_error = prediction + error_pred
            prediction_data[model_type + '_prediction_with_error'] = prediction_with_error
        except Exception as e:
            logging.info(f"residual model prediction error: {str(e)}")
            pass
    
    # Save predictions to an Excel file
    output_file = f'./Model_build1/predictions_hybrid_2fit_{model_type}.xlsx'
    prediction_data.to_excel(output_file, index=False)
    print(f"Predictions saved to {output_file}")

data_with_error = add_predicted_values(new_real_scores,new_prediction_data)
data_with_error = data_with_error.drop(columns=['match_outcome', 'Datum','home_goals','away_goals',  
                               'draw', 'away_win', 'home_win','away_points', 'home_points','HomeTeam_last_away_match','AwayTeam_last_home_match',
                               'home_points_rolling_avg','away_points_rolling_avg','home_advantage'], errors='ignore')

# logging.info(f"data_with error columns: {data_with_error.columns}")

# Count NaN values in each column
nan_counts = data_with_error.isna().sum()
print(nan_counts)
# logging.info(nan_counts)

data_with_error = data_with_error.dropna()
logging.info(f"data_with_error length: {len(data_with_error)}")

prediction_df = new_prediction_data.drop(columns=['Unnamed: 0.1','Date', 'Unnamed: 0','match_outcome', 'home_goals','away_goals',   'draw', 'away_win', 'home_win','away_points', 'home_points','HomeTeam_last_away_match','AwayTeam_last_home_match','home_points_rolling_avg','away_points_rolling_avg','home_advantage'], errors='ignore')
prediction_df = prediction_df.replace(',', '.', regex=True)
prediction_df = prediction_df.apply(pd.to_numeric, errors='coerce')
prediction_df.replace([np.inf, -np.inf], np.nan, inplace=True)

logging.info(f"prediction_df length: {len(prediction_df)}")
stacking_regressor = train_model(base_data, data, model_type)
logging.info(f"Start {model_type} predictions")
# logging.info(f"columns of Prediction_df: {prediction_df.columns}")

# MAKE PREDICTIONS
try:
    if len(data_with_error)>0:
        # Train a residual model for error
        X_train_error = data_with_error.drop(columns=['real_away_goals','real_home_goals','real_outcome','home_error','away_error','outcome_error','home_goals_prediction_rounded','away_goals_prediction_rounded','match_outcome_prediction_rounded','Date'],errors='ignore')
        y_train_home_error = data_with_error['home_error']
        logging.info(f"Residual model train data selected")
        residual_model = RandomForestRegressor()
        # # Ensure feature names are preserved
        scaler_file = os.path.join(model_dir, 'scaler_' + model_type + '.pkl')
        scaler_loaded = joblib.load(scaler_file)
        logging.info(f"Residual model scaler loaded")
        X_new_scaled = scaler_loaded.transform(X_train_error)
        residual_model.fit(X_new_scaled, y_train_home_error)
        logging.info(f"Residual model successfully fitted")
        
        make_prediction(model_type,stacking_regressor,prediction_df, residual_model)
    else: 
        make_prediction(model_type,stacking_regressor,prediction_df, 0)
        
except Exception as e:
    logging.error(f"Error occurred while making prediction: {e}")
    make_prediction(model_type,stacking_regressor,prediction_df, 0)
    pass

# Initialize the wrappers
custom_model = CustomStackingRegressor(stacking_regressor, 
                                       stacking_regressor.named_estimators_['nn'].model_,
                                       keras_nn_model_path)

# Save the model (this will save both the Keras model and the rest of the stacking regressor)
custom_model.save(model_file)