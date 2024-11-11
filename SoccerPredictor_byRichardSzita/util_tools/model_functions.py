import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.regularizers import l2
from util_tools.model_classes import WithinRangeMetric

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def prepare_data(data: pd.DataFrame, features: list, model_type: str, model_dir: str, logger) -> pd.DataFrame:
    """
    Prepares the data by replacing commas, converting to numeric, and applying imputation.

    Args:
        data (pd.DataFrame): The input data.
        features (list): List of features to be used.
        model_type (str): The type of model.
        model_dir (str): Directory to save the imputer.
        logger: Logger for logging information.

    Returns:
        pd.DataFrame: The imputed data.
    """
    model_data = data.replace(',', '.', regex=True)
    model_data = model_data.apply(pd.to_numeric, errors='coerce')
    model_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    model_data = model_data[features]

    # Apply Iterative Imputation
    imputer = KNNImputer(n_neighbors=10, weights='uniform')
    model_data_imputed = imputer.fit_transform(model_data)

    # Save the imputer
    imputer_file = os.path.join(model_dir, f'imputer_{model_type}.pkl')
    joblib.dump(imputer, imputer_file)
    logger.info(f"Imputer saved to {imputer_file}")

    return pd.DataFrame(model_data_imputed, columns=features)

def prepare_new_data(new_data: pd.DataFrame, imputer, selector, model_type: str, model_dir: str, logger, numeric_features: list) -> pd.DataFrame:
    """
    Prepares new data for prediction by applying imputation, scaling, and feature selection.

    Args:
        new_data (pd.DataFrame): The new data for prediction.
        imputer: The imputer used for missing data.
        selector: The feature selector.
        model_type (str): The type of model.
        model_dir (str): Directory to load the scaler.
        logger: Logger for logging information.
        numeric_features (list): List of numeric features.

    Returns:
        pd.DataFrame: The prepared data ready for prediction.
    """
    model_data = new_data.replace(',', '.', regex=True)
    model_data = model_data.apply(pd.to_numeric, errors='coerce')
    logger.info(f"Prediction Selected Features: {numeric_features}")

    scaler_file = os.path.join(model_dir, f'scaler_{model_type}.pkl')
    model_data = model_data[numeric_features]
    scaler_loaded = joblib.load(scaler_file)

    # Apply imputation
    model_data_imputed = imputer.transform(model_data)

    # Apply scaling
    model_data_scaled = scaler_loaded.transform(model_data_imputed)

    # Apply feature selection (RFE)
    model_data_selected = selector.transform(model_data_scaled)

    return pd.DataFrame(model_data_selected)

def create_neural_network(input_dim: int) -> Sequential:
    """
    Creates a neural network model.

    Args:
        input_dim (int): The input dimension for the model.

    Returns:
        Sequential: The compiled Keras model.
    """
    model = Sequential()

    # First layer
    model.add(Dense(128, input_dim=input_dim, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    # Second layer
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    # Third layer
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae', WithinRangeMetric()]
    )

    return model

def within_range_evaluation(y_true, y_pred, tolerance=0.5) -> float:
    """
    Evaluate the percentage of predictions within a specified range of the true values.

    Args:
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.
        tolerance (float): The acceptable range for predictions to be considered correct.

    Returns:
        float: The percentage of predictions within the specified range.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    diff = np.abs(y_true - y_pred)
    within_range = diff <= tolerance
    percentage_within_range = np.mean(within_range) * 100

    return percentage_within_range
