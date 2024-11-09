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
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2   
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras.backend as K
import h5py
from keras.backend import manual_variable_initialization 
manual_variable_initialization(True)

model_type='away_goals'
# Set up logging
log_file_path = './SoccerPredictor_byRichardSzita/Prediction_models/log/stacked_' + model_type + '_model_training.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set directories and files
model_dir = "./models/"
os.makedirs(model_dir, exist_ok=True)
logging.info("Model directory checked/created.")

scaler = StandardScaler()
selected_features = []
numeric_features = []
# Load your feature-engineered data
base_data_path = './SoccerPredictor_byRichardSzita/data/model_data_training_withPoisson.xlsx'
new_prediction_path = './SoccerPredictor_byRichardSzita/data/merged_data_prediction.csv'
real_scores_path = './SoccerPredictor_byRichardSzita/made_predictions/predictions_stacked_merged_latest.xlsx'
base_data = pd.read_excel(base_data_path)
logging.info(f"Data loaded from {base_data_path} with shape: {base_data.shape}")

new_prediction_data = pd.read_csv(new_prediction_path)
logging.info(f"Data loaded from {new_prediction_path} with shape: {new_prediction_data.shape}")
# new_prediction_data = new_prediction_data[(new_prediction_data['season_encoded'] == 4)]
logging.info(f"Data loaded with {len(new_prediction_data)} rows")

# Filter data to only include away_goals from 0 to 5
base_data = base_data[(base_data['home_goals'] >= 0) & (base_data['home_goals'] <= 6) & (base_data['away_goals'] >= 0) & (base_data['away_goals'] <= 6) ]
logging.info(f"Data filtered to only include away_goals from 0 to 6. rows: {len(base_data)} Filtered data shape: {base_data.shape}")
# Drop unnecessary columns
data = base_data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0','match_outcome', 'home_goals','away_goals',  
                               'draw', 'away_win', 'home_win','away_points', 'home_points','HomeTeam_last_away_match','AwayTeam_last_home_match',
                               'home_points_rolling_avg','away_points_rolling_avg','home_advantage'], errors='ignore')

# Import predicted scores
new_real_scores = pd.read_excel(real_scores_path)
new_real_scores.dropna(subset=['running_id','home_goals_prediction_rounded','away_goals_prediction_rounded','real_score'],inplace=True)
logging.info(f"new real scores length: {len(new_real_scores)}")

# Prepare data
def prepare_data(data, features, model_type):
   
    model_data = data[features]

    # Apply Iterative Imputation
    imputer = IterativeImputer(random_state=42)
    model_data_imputed = imputer.fit_transform(model_data)
    # Save the imputer
    imputer_file = os.path.join(model_dir, 'imputer_'+ model_type + '.pkl')
    joblib.dump(imputer, imputer_file)
    logging.info(f"Imputer saved to {imputer_file}")

    return pd.DataFrame(model_data_imputed, columns=features)

# Define Keras Neural Network model
# def create_neural_network(input_dim):
#     model = Sequential()
#     model.add(Dense(128, input_dim=input_dim, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(1))  # Output layer for regression (no activation function)
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
#     return model

# Merge base data with predicted values
def add_predicted_values(new_real_scores,data):
    # Split the column into two based on '-'
    new_real_scores[['real_home_goals', 'real_away_goals']] = new_real_scores['real_score'].str.split('-', expand=True)
    new_real_scores['real_away_goals'] = new_real_scores['real_away_goals'].astype(int)
    new_real_scores['real_home_goals'] = new_real_scores['real_home_goals'].astype(int)
    new_real_scores['running_id'] = new_real_scores['running_id'].astype(int)
    # Calculate residuals (prediction errors)
    new_real_scores['home_error'] = new_real_scores['real_home_goals'] - new_real_scores['home_goals_prediction_rounded']
    new_real_scores['away_error'] = new_real_scores['real_away_goals'] - new_real_scores['away_goals_prediction_rounded']
    # new_real_scores['outcome_error'] = new_real_scores['real_match_outcome'] - new_real_scores['outcome_prediction']
    # Calculate the mean squared error for home and away predictions
    mse_home_prediction = mean_squared_error(new_real_scores['real_home_goals'], new_real_scores['home_goals_prediction_rounded'])
    mse_away_prediction = mean_squared_error(new_real_scores['real_away_goals'], new_real_scores['away_goals_prediction_rounded'])
    # mse_outcome_prediction = mean_squared_error(new_real_scores['real_match_outcome'], new_real_scores['outcome_prediction'])

    # Calculate R-squared score for home and away predictions
    r2_home_prediction = r2_score(new_real_scores['real_home_goals'], new_real_scores['home_goals_prediction_rounded'])
    r2_away_prediction = r2_score(new_real_scores['real_away_goals'], new_real_scores['away_goals_prediction_rounded'])
    # r2_outcome_prediction = r2_score(new_real_scores['real_away_goals'], new_real_scores['away_goals_prediction_rounded'])
    
    print(f"Home MSE: {mse_home_prediction}, Away MSE: {mse_away_prediction}")
    print(f"Home R2: {r2_home_prediction}, Away R2: {r2_away_prediction}")
    # print(f"Outcome MSE: {mse_outcome_prediction}, Outcome R2: {r2_outcome_prediction}")
    
    score_df = new_real_scores[['running_id','home_goals_prediction_rounded','away_goals_prediction_rounded','real_home_goals','real_away_goals','home_error','away_error']]
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
    logging.info(f"Prediction Selected Features: {numeric_features}")
    scaler_file = os.path.join(model_dir, 'scaler_' + model_type + '.pkl')
    # numeric_features = new_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    model_data = new_data[numeric_features]
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



# Define the neural network architecture (assuming the function already exists)
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
    model.compile(loss='mse', optimizer=Adam(), metrics=['mae',within_range])
    
    return model

# Train the model
def train_model(base_data, data, model_type):
    global selected_features
    global numeric_features

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
    logging.info("Data scaling1 completed.")

    
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
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.0001)
    
    # Define models for home goals prediction
    rf_regressor_home = RandomForestRegressor(n_estimators=200, random_state=42)
    svr_regressor_home = SVR(kernel='rbf')
    nn_regressor_home = KerasRegressor(
        model=create_neural_network,
        model__input_dim=X_train_selected.shape[1],
        epochs=100,  # Set higher epochs but use early stopping
        batch_size=32,
        verbose=1,
        callbacks=[lr_scheduler]  # LR Scheduler
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
    logging.info(f"{model_type} (2nd fit) Stacking Model Within Range (±0.3): {within_range_home}%")
    
    # # 2nd Fit of the model
    # logging.info('Second fit started')
    # stacking_regressor_home.fit(X_train2_selected, y_train2)
    # logging.info(f"Stacking model for {model_type} trained successfully on second split.")
    
    # # Evaluation on the second test set
    # y_pred_home2 = stacking_regressor_home.predict(X_test2_selected)
    # mse_home2 = mean_squared_error(y_test2, y_pred_home2)
    # r2_home2 = r2_score(y_test2, y_pred_home2)
    # mae_home2 = mean_absolute_error(y_test2, y_pred_home2)
    # mape_home2 = np.mean(np.abs((y_test2 - y_pred_home2) / y_test2)) * 100
    # within_range_home2 = within_range_evaluation(y_test2, y_pred_home2, tolerance=0.3) * 100  # Convert to percentage

    # logging.info(f"{model_type} (2nd fit) Stacking Model MSE: {mse_home2}, R2: {r2_home2}, Stacking Model MAE: {mae_home2}, Stacking Model MAPE: {mape_home2}%")
    # logging.info(f"{model_type} (2nd fit) Stacking Model Within Range (±0.3): {within_range_home2}%")

    
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
    logging.info(f"home goal predictions: {prediction}")
    # Add predictions to the new data DataFrame
    prediction_data[prediction_column_name] = prediction
    
    if residual_model != 0:
        # Predict the home error using the residual model
        error_pred = residual_model.predict(prediction_data)
        logging.info(f"home goal error predictions: {error_pred}")
        # Final adjusted prediction for goals
        prediction_with_error = prediction + error_pred
        prediction_data[model_type + '_prediction_with_error'] = prediction_with_error
    
    # Save predictions to an Excel file
    output_file = f'./SoccerPredictor_byRichardSzita/predictions_hybrid_{model_type}_3.xlsx'
    prediction_data.to_excel(output_file, index=False)
    print(f"Predictions saved to {output_file}")

data_with_error = add_predicted_values(new_real_scores,data)
data_with_error = data_with_error.drop(columns=['match_outcome', 'Datum','home_goals','away_goals',  
                               'draw', 'away_win', 'home_win','away_points', 'home_points','HomeTeam_last_away_match','AwayTeam_last_home_match',
                               'home_points_rolling_avg','away_points_rolling_avg','home_advantage'], errors='ignore')
print(len(data_with_error))
data_with_error = data_with_error.dropna()
logging.info(f"data_with_error length: {len(data_with_error)}")

prediction_df = new_prediction_data.drop(columns=['Unnamed: 0.1','Date', 'Unnamed: 0','match_outcome', 'home_goals','away_goals',  
                               'draw', 'away_win', 'home_win','away_points', 'home_points','HomeTeam_last_away_match','AwayTeam_last_home_match',
                               'home_points_rolling_avg','away_points_rolling_avg','home_advantage'], errors='ignore')

prediction_df.replace([np.inf, -np.inf], np.nan, inplace=True)
# prediction_df = prediction_df[(prediction_df['season_encoded'] == 4)]
# prediction_df = prediction_df.dropna()
# print(len(prediction_df))
logging.info(f"prediction_df length: {len(prediction_df)}")
stacking_regressor = train_model(base_data, data, model_type)
logging.info('Start away_goals predictions')
logging.info(f"columns of Prediction_df: {prediction_df.columns}")

try:
    if len(data_with_error)>0:
        # Train a residual model for error
        X_train_error = data_with_error.drop(columns=['real_away_goals','real_home_goals','home_error','away_error','home_goals_prediction_rounded','away_goals_prediction_rounded'],errors='ignore')
        y_train_home_error = data_with_error['away_error']
        residual_model = RandomForestRegressor()
        # # Ensure feature names are preserved
        X_new_scaled = scaler.transform(X_train_error)
        residual_model.fit(X_new_scaled, y_train_home_error)
        make_prediction(model_type,stacking_regressor,prediction_df, residual_model)
    else: 
        make_prediction(model_type,stacking_regressor,prediction_df, 0)
except Exception as e:
    logging.error(f"Error occurred while making prediction: {e}")
    pass

# Make a deep copy of the stacking regressor to avoid side effects
# stacking_regressor_copy = deepcopy(stacking_regressor)

# Define file paths
model_file = os.path.join(model_dir, 'model_stacked_2fit_' + model_type + '.pkl')
keras_nn_model_path = os.path.join(model_dir, 'nn_regressor_' + model_type + '_stacked_2fit')

try:
    # Separate and save the Keras neural network model
    trained_keras_model = stacking_regressor.named_estimators_['nn'].model_
    
    # Save the Keras model using the `.save()` method
    trained_keras_model.save(keras_nn_model_path)  # Save Keras model in SavedModel format
    logging.info(f"Keras model saved successfully at {keras_nn_model_path}")
    
    # Remove the Keras model from the stacking regressor before saving
    stacking_regressor = stacking_regressor.named_estimators_.pop('nn')
    
    # Save the stacking model (without the neural network part) using joblib
    joblib.dump(stacking_regressor, model_file)
    logging.info(f"Stacking model saved successfully at {model_file}")
    
except Exception as e:
    logging.error(f"Error occurred while saving models: {e}")
    raise


# Optionally, zip the SavedModel folder for easier portability
shutil.make_archive(keras_nn_model_path, 'zip', keras_nn_model_path)
logging.info(f"Keras model zipped successfully at {keras_nn_model_path}.zip")