import os
import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from scikeras.wrappers import KerasRegressor  # Ensure compatibility with Keras
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from keras.models import Sequential
from keras.layers import Dense
import h5py
from keras.backend import manual_variable_initialization 
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

scaler = StandardScaler()

# Load your feature-engineered data
base_data_path = './Model_build1/data/model_data_training.csv'
new_prediction_path = './Model_build1/data/merged_data_prediction.xlsx'
real_scores_path = './Model_build1/hybrid_predictions_with_names.xlsx'
base_data = pd.read_csv(base_data_path)
logging.info(f"Data loaded from {base_data_path} with shape: {base_data.shape}")

new_prediction_data = pd.read_excel(new_prediction_path)
logging.info(f"Data loaded from {new_prediction_path} with shape: {new_prediction_data.shape}")

# Filter data to only include home_goals from 0 to 5
base_data = base_data[(base_data['home_goals'] >= 0) & (base_data['home_goals'] <= 6) & (base_data['away_goals'] >= 0) & (base_data['away_goals'] <= 6) & (base_data['league_encoded'] == 3)]
logging.info(f"Data filtered to only include home_goals from 0 to 6. rows: {len(base_data)} Filtered data shape: {base_data.shape}")
# Drop unnecessary columns
data = base_data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0','match_outcome', 'home_goals','away_goals',  
                               'draw', 'away_win', 'home_win','away_points', 'home_points','HomeTeam_last_away_match','AwayTeam_last_home_match',
                               'home_points_rolling_avg','away_points_rolling_avg','home_advantage'], errors='ignore')

# Import predicted scores
new_real_scores = pd.read_excel(real_scores_path)
new_real_scores.dropna(subset=['running_id','home_prediction','away_prediction','outcome_prediction','real_home_goals','real_away_goals','real_match_outcome'],inplace=True)
new_real_scores = new_real_scores[(new_real_scores['league_encoded'] == 3)]

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
def create_neural_network(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))  # Output layer for regression (no activation function)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Merge base data with predicted values
def add_predicted_values(new_real_scores,data):
  
    # Calculate residuals (prediction errors)
    new_real_scores['home_error'] = new_real_scores['real_home_goals'] - new_real_scores['home_prediction']
    new_real_scores['away_error'] = new_real_scores['real_away_goals'] - new_real_scores['away_prediction']
    new_real_scores['outcome_error'] = new_real_scores['real_match_outcome'] - new_real_scores['outcome_prediction']
    # Calculate the mean squared error for home and away predictions
    mse_home_prediction = mean_squared_error(new_real_scores['real_home_goals'], new_real_scores['home_prediction'])
    mse_away_prediction = mean_squared_error(new_real_scores['real_away_goals'], new_real_scores['away_prediction'])
    mse_outcome_prediction = mean_squared_error(new_real_scores['real_match_outcome'], new_real_scores['outcome_prediction'])

    # Calculate R-squared score for home and away predictions
    r2_home_prediction = r2_score(new_real_scores['real_home_goals'], new_real_scores['home_prediction'])
    r2_away_prediction = r2_score(new_real_scores['real_away_goals'], new_real_scores['away_prediction'])
    r2_outcome_prediction = r2_score(new_real_scores['real_away_goals'], new_real_scores['away_prediction'])
    
    print(f"Home MSE: {mse_home_prediction}, Away MSE: {mse_away_prediction}")
    print(f"Home R2: {r2_home_prediction}, Away R2: {r2_away_prediction}")
    print(f"Outcome MSE: {mse_outcome_prediction}, Outcome R2: {r2_outcome_prediction}")
    
    score_df = new_real_scores[['running_id','home_prediction','away_prediction','real_home_goals','real_away_goals','real_match_outcome','home_error','away_error','outcome_error']]
    logging.info(f"New score dataframe created with errors: {score_df.head(5)}")
    # Merge the predictions with the real scores
    merged_data = pd.merge(data, score_df, on='running_id')
    numeric_features = merged_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    merged_data = merged_data[numeric_features]
    return merged_data

# Assuming `new_data` is your new data in a pandas DataFrame
def prepare_new_data(new_data, imputer, selector):
    numeric_features = new_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    model_data = new_data[numeric_features]
    
    # Apply imputation
    model_data_imputed = imputer.transform(model_data)  # Use the imputer you saved during training
    
    # Apply scaling
    model_data_scaled = scaler.transform(model_data_imputed)  # Use the scaler you saved during training
    
    # Apply feature selection (RFE)
    model_data_selected = selector.transform(model_data_scaled)  # Use the RFE selector saved during training
    logging.info(model_data_selected)
    return pd.DataFrame(model_data_selected)

# Train the model
def train_model(base_data, data, model_type):
    # Select all numeric features
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    logging.info(f"Numeric features selected: {numeric_features}")

    X = prepare_data(data, numeric_features, model_type)
    y = base_data[model_type]  # Target variable
    logging.info(f"Data prepared for modeling. Feature shape: {X.shape}, Target shape: {y.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Data split into train and test. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Scaling
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logging.info("Data scaling completed.")

    # Save the scaler for future use
    scaler_file = os.path.join(model_dir, 'scaler_'+ model_type +'.pkl')
    joblib.dump(scaler, scaler_file)
    logging.info(f"Scaler saved to {scaler_file}")

    # Feature Selection (RFE)
    selector = RFE(estimator=RandomForestRegressor(), n_features_to_select=30)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    logging.info("Feature selection using RFE completed.")

    # Save the RFE selector for future use
    selector_file = os.path.join(model_dir, 'rfe_'+ model_type +'_selector.pkl')
    joblib.dump(selector, selector_file)
    logging.info(f"RFE selector saved to {selector_file}")

    # Define models for home goals prediction
    rf_regressor_home = RandomForestRegressor(n_estimators=200, random_state=42)
    svr_regressor_home = SVR(kernel='rbf')
    nn_regressor_home = KerasRegressor(
        model=create_neural_network,  # Passing the function reference, not calling it
        model__input_dim=X_train_selected.shape[1],  # Arguments for create_neural_network
        epochs=30,
        batch_size=32,
        verbose=1
    )
    xgb_regressor_home = XGBRegressor()

    # Stacking Regressor
    estimators_home = [
        ('rf', rf_regressor_home),
        ('svr', svr_regressor_home),
        ('nn', nn_regressor_home),
        ('xgb', xgb_regressor_home)
    ]
    stacking_regressor_home = StackingRegressor(estimators=estimators_home, final_estimator=LinearRegression())
    stacking_regressor_home.fit(X_train_selected, y_train)
    logging.info(f"Stacking model for {model_type} trained successfully.")

    # Evaluation
    y_pred_home = stacking_regressor_home.predict(X_test_selected)
    mse_home = mean_squared_error(y_test, y_pred_home)
    r2_home = r2_score(y_test, y_pred_home)
    mae_home = mean_absolute_error(y_test, y_pred_home)
    mape_home = np.mean(np.abs((y_test - y_pred_home) / y_test)) * 100
    logging.info(f"{model_type} Stacking Model MSE: {mse_home}, R2: {r2_home}, Stacking Model MAE: {mae_home}, Stacking Model MAPE: {mape_home}%")

    scores = cross_val_score(stacking_regressor_home, X_train_selected, y_train, scoring='neg_mean_squared_error', cv=5)
    print(f"Cross-validated MSE: {-scores.mean()}")
    
    # Extract the trained Keras model from the stacking regressor
    trained_keras_model = stacking_regressor_home.named_estimators_['nn'].model_
    # Save the stacking model without the Keras model
    stacking_regressor_home = stacking_regressor_home.named_estimators_.pop('nn')  # Remove the Keras model from the stacking regressor
    stacking_model_file = os.path.join(model_dir, model_type +'_stacking_model.pkl')
    joblib.dump(stacking_regressor_home, stacking_model_file)
    logging.info(f"Stacking model for home goals (without Keras) saved to {stacking_model_file}")

    # Save the trained Keras model in the SavedModel format (no need for h5py)
    keras_nn_model_path = os.path.join(model_dir, 'nn_regressor_'+ model_type +'_model')
    trained_keras_model.save(keras_nn_model_path)  # Save the Keras model in SavedModel format
    logging.info(f"Keras neural network model saved to {keras_nn_model_path}")
    
    return stacking_regressor_home

# Make new predictions
def make_prediction(model_type, model, residual_model, prediction_data):
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
 
    # Predict the home goals using the original model
    prediction = model.predict(X_new_prepared)
    prediction_column_name= model_type + '_prediction'
    logging.info(f"home goal predictions: {prediction}")
    
    # Predict the home error using the residual model
    error_pred = residual_model.predict(prediction_data)
    logging.info(f"home goal error predictions: {error_pred}")
    # Final adjusted prediction for goals
    prediction_with_error = prediction + error_pred
    
    # Add predictions to the new data DataFrame
    prediction_data[prediction_column_name] = prediction
    prediction_data[model_type + '_prediction_with_error'] = prediction_with_error

    # Save predictions to an Excel file
    output_file = f'./Model_build1/predictions_hybrid_{model_type}_1.xlsx'
    prediction_data.to_excel(output_file, index=False)
    print(f"Predictions saved to {output_file}")

data_with_error = add_predicted_values(new_real_scores,data)
data_with_error = data_with_error.drop(columns=['match_outcome', 'home_goals','away_goals',  
                               'draw', 'away_win', 'home_win','away_points', 'home_points','HomeTeam_last_away_match','AwayTeam_last_home_match',
                               'Home_shot_on_target_mean','away_shot_on_target_mean','home_points_rolling_avg','away_points_rolling_avg','home_advantage'], errors='ignore')

logging.info(f"data_with_error length: {len(data_with_error)}")
prediction_df = new_prediction_data.drop(columns=['match_outcome', 'home_goals','away_goals',  
                               'draw', 'away_win', 'home_win','away_points', 'home_points','HomeTeam_last_away_match','AwayTeam_last_home_match',
                               'home_points_rolling_avg','away_points_rolling_avg','Home_shot_on_target_mean','away_shot_on_target_mean','home_advantage'], errors='ignore')

prediction_df.replace([np.inf, -np.inf], np.nan, inplace=True)
prediction_df = prediction_df[(prediction_df['league_encoded'] == 3)]
prediction_df = prediction_df.dropna()
logging.info(f"prediction_df length: {len(prediction_df)}")
# Train a residual model for error
X_train_error = data_with_error.drop(columns=['real_match_outcome','real_away_goals','real_home_goals','home_error','away_error','outcome_error','home_prediction','away_prediction','outcome_prediction'],errors='ignore')
y_train_home_error = data_with_error['home_error']

stacking_regressor_home = train_model(base_data, data, 'home_goals')

residual_model_home = RandomForestRegressor()

# Ensure feature names are preserved
X_new_scaled = scaler.transform(X_train_error)

residual_model_home.fit(X_new_scaled, y_train_home_error)
logging.info('Start home_goals predictions')
logging.info(f"columns of Prediction_df: {prediction_df.columns}")
make_prediction('home_goals',stacking_regressor_home,residual_model_home,prediction_df)

