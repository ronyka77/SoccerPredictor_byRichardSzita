import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
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

    # Apply polynomial features transformation first
    poly = PolynomialFeatures(degree=2, include_bias=False)
    model_data_poly = poly.fit_transform(model_data_imputed)
    
    # Then apply scaling to polynomial features
    model_data_scaled = scaler_loaded.transform(model_data_poly)
    
    # Apply feature selection (RFE)
    model_data_selected = selector.transform(model_data_scaled)

    return pd.DataFrame(model_data_selected)

def perform_feature_selection(X_train_scaled: np.ndarray, y_train: pd.Series, model_dir: str, model_type: str, logger: logging.Logger) -> RFE:
    """
    Perform feature selection using Recursive Feature Elimination (RFE) with XGBoost, which handles 
    polynomial features well and can capture non-linear relationships.

    Args:
        X_train_scaled (np.ndarray): Training data with polynomial features, already scaled.
        y_train (pd.Series): Training target variable.
        model_dir (str): Directory to save model artifacts.
        model_type (str): The type of model to train.
        logger (logging.Logger): Logger for logging information.

    Returns:
        RFE: The fitted RFE selector.
    """
    logger.info("Feature Selection started.")
    
    # Using XGBoost as base estimator since it handles polynomial features well
    base_estimator = XGBRegressor(
        n_estimators=500,  # Increased from 100 for better model convergence
        learning_rate=0.05,  # Reduced to allow for finer parameter updates
        max_depth=8,  # Increased from 5 to capture more complex relationships
        min_child_weight=3,  # Added to prevent overfitting on noisy data
        subsample=0.8,  # Added to reduce overfitting
        colsample_bytree=0.8,  # Added to reduce overfitting
        gamma=0.1,  # Added to control tree growth
        reg_alpha=0.1,  # L1 regularization to reduce model complexity
        reg_lambda=1.0,  # L2 regularization for stability
        random_state=42
    )
    
    # Perform RFE with XGBoost
    selector = RFE(
        estimator=base_estimator,
        n_features_to_select=120,  # ~5% of total 2400 features for optimal balance
        step=0.1  # Remove 10% of features at each iteration
    )
    logger.info("RFE fitting started.")
    selector.fit(X_train_scaled, y_train)
    
    # Save the RFE selector for future use
    selector_file = os.path.join(model_dir, 'rfe_' + model_type + '_selector.pkl')
    joblib.dump(selector, selector_file)
    logger.info(f"RFE selector saved to {selector_file}")
    
    # Log feature selection results
    n_selected = sum(selector.support_)
    logger.info(f"Selected {n_selected} features out of {X_train_scaled.shape[1]}")
    
    return selector

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
