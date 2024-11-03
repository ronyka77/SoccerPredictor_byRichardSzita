import os
import pandas as pd
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from scikeras.wrappers import KerasRegressor  # Ensure compatibility with Keras
from sklearn.metrics import mean_squared_error, r2_score
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

# Set up logging
log_file_path = './Model_build1/Outcome_prediction/log/hybrid_outcome_model_training.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set directories and files
model_dir = "./models/"
os.makedirs(model_dir, exist_ok=True)
logging.info("Model directory checked/created.")

# Load your feature-engineered data
file_path = './Model_build1/data/model_data_training.csv'
data = pd.read_csv(file_path)
logging.info(f"Data loaded from {file_path} with shape: {data.shape}")

# Drop unnecessary columns
data = data.drop(columns=['home_goals', 'away_goals', 'away_points', 'draw', 'away_win', 'home_points', 'home_win'], errors='ignore')

# Select all numeric features
numeric_features = data.drop(columns=['match_outcome']).select_dtypes(include=['float64', 'int64']).columns.tolist()
logging.info(f"Numeric features selected: {numeric_features}")

# Prepare data
def prepare_data(data, features):
   
    model_data = data[features]

    # Apply Iterative Imputation
    imputer = IterativeImputer(random_state=42)
    model_data_imputed = imputer.fit_transform(model_data)
    # Save the imputer
    imputer_file = os.path.join(model_dir, 'imputer_outcome.pkl')
    joblib.dump(imputer, imputer_file)
    logging.info(f"Imputer saved to {imputer_file}")

    return pd.DataFrame(model_data_imputed, columns=features)

X = prepare_data(data, numeric_features)
y = data['match_outcome']  # Target variable
logging.info(f"Data prepared for modeling. Feature shape: {X.shape}, Target shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f"Data split into train and test. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logging.info("Data scaling completed.")

# Save the scaler for future use
scaler_file = os.path.join(model_dir, 'scaler_outcome.pkl')
joblib.dump(scaler, scaler_file)
logging.info(f"Scaler saved to {scaler_file}")

# Feature Selection (RFE)
selector = RFE(estimator=RandomForestRegressor(), n_features_to_select=20)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
logging.info("Feature selection using RFE completed.")

# Save the RFE selector for future use
selector_file = os.path.join(model_dir, 'rfe_outcome_selector.pkl')
joblib.dump(selector, selector_file)
logging.info(f"RFE selector saved to {selector_file}")

# Define Keras Neural Network model
def create_neural_network(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))  # Output layer for regression (no activation function)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

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
logging.info("Stacking model for home Goals trained successfully.")

# Evaluation
y_pred_home = stacking_regressor_home.predict(X_test_selected)
mse_home = mean_squared_error(y_test, y_pred_home)
r2_home = r2_score(y_test, y_pred_home)
logging.info(f"Home Goals Stacking Model MSE: {mse_home}, R2: {r2_home}")

# Extract the trained Keras model from the stacking regressor
trained_keras_model = stacking_regressor_home.named_estimators_['nn'].model_
# Save the stacking model without the Keras model
stacking_regressor_home = stacking_regressor_home.named_estimators_.pop('nn')  # Remove the Keras model from the stacking regressor
stacking_model_file = os.path.join(model_dir, 'homescore_stacking_model.pkl')
joblib.dump(stacking_regressor_home, stacking_model_file)
logging.info(f"Stacking model for home goals (without Keras) saved to {stacking_model_file}")

# Save the trained Keras model in the SavedModel format (no need for h5py)
keras_nn_model_path = os.path.join(model_dir, 'nn_regressor_home_model')
trained_keras_model.save(keras_nn_model_path)  # Save the Keras model in SavedModel format
logging.info(f"Keras neural network model saved to {keras_nn_model_path}")


# MAKE PREDICTION
# Directory where the models are saved
model_dir = "./models/"
model_type = 'outcome'
imputer_file = f'imputer_{model_type}.pkl'  # Imputer file from training
# Load and apply the saved imputer from training
imputer_path = os.path.join(model_dir, imputer_file)
try:
    imputer = joblib.load(imputer_path)
    logging.info(f"Imputer loaded from {imputer_path}")
except FileNotFoundError:
    logging.error(f"Imputer file not found at {imputer_path}")
    raise
# Load your feature-engineered data for testing
file_path = './Model_build1/data/merged_data_prediction.xlsx'
new_data = pd.read_excel(file_path)
logging.info(f"New data loaded from {file_path} with shape: {new_data.shape}")

# Assuming `new_data` is your new data in a pandas DataFrame
def prepare_new_data(new_data):
    numeric_features = new_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    model_data = new_data[numeric_features]
    
    # Apply imputation
    model_data_imputed = imputer.transform(model_data)  # Use the imputer you saved during training
    
    # Apply scaling
    model_data_scaled = scaler.transform(model_data_imputed)  # Use the scaler you saved during training
    
    # Apply feature selection (RFE)
    model_data_selected = selector.transform(model_data_scaled)  # Use the RFE selector saved during training
    
    return model_data_selected

X_new_prepared = prepare_new_data(new_data)
# Make predictions using the trained stacking model
predictions = stacking_regressor_home.predict(X_new_prepared)

# Add predictions to the new data DataFrame
new_data['Outcome Predictions'] = predictions

# Save predictions to an Excel file
output_file = f'./Model_build1/predictions_hybrid_{model_type}_1.xlsx'
new_data.to_excel(output_file, index=False)
print(f"Predictions saved to {output_file}")


