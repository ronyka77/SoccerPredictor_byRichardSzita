import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE, RFECV, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, cross_validate, KFold, LeaveOneOut, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasRegressor
from scipy.stats import uniform, randint
from util_tools.model_classes import WithinRangeMetric, EarlyStoppingCallback, LoggingEstimator, CustomReduceLROnPlateau
import optuna
import optuna.visualization as vis
import mlflow
import psutil
from multiprocessing import cpu_count
import gc


# Set this before running your code
os.environ["MLFLOW_TRACKING_URI"] = "file:///192.168.0.77/Betting/Chatgpt/SoccerPredictor_byRichardSzita/mlruns"

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

def create_enhanced_neural_network(input_dim: int, n_layers: int = 3, units: int = 256) -> Sequential:
    """
    Creates an enhanced neural network with configurable layers and units.
    
    Args:
        input_dim (int): Input dimension
        n_layers (int): Number of hidden layers
        units (int): Number of units in first layer (halved for each subsequent layer)
        
    Returns:
        Sequential: Compiled Keras model
    """
    model = Sequential()
    
    # Input layer
    model.add(Dense(units, input_dim=input_dim, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    # Hidden layers
    for i in range(n_layers - 1):
        units = units // 2
        model.add(Dense(units, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(max(0.1, 0.3 - i * 0.1)))
    
    # Output layer
    model.add(Dense(1, activation='relu'))
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='huber',
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

def calculate_poisson_deviance(y_true, y_pred):
    """Calculate Poisson deviance for count data like goals"""
    # Avoid division by zero and log(0) by adding small constant
    eps = 1e-10
    y_pred = np.maximum(eps, y_pred)
    dev = 2 * (y_true * np.log((y_true + eps) / y_pred) - (y_true - y_pred))
    return np.mean(dev)

def analyze_goals_difference(y_true, y_pred):
    """Analyze distribution of differences between predicted and actual goals"""
    diff = y_pred - y_true
    metrics = {
        'mean_diff': np.mean(diff),
        'std_diff': np.std(diff),
        'within_0.5': np.mean(np.abs(diff) <= 0.5) * 100,
        'within_1.0': np.mean(np.abs(diff) <= 1.0) * 100
    }
    return metrics

def calculate_calibration_score(y_true, y_pred):
    """Calculate calibration score for goal predictions"""
    # Bin predictions and calculate average actual values per bin
    bins = np.linspace(0, max(y_pred), 10)
    bin_means = np.zeros(len(bins)-1)
    bin_true = np.zeros(len(bins)-1)
    
    for i in range(len(bins)-1):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
        if np.sum(mask) > 0:
            bin_means[i] = np.mean(y_pred[mask])
            bin_true[i] = np.mean(y_true[mask])
    
    # Calculate RMSE between bin predictions and actuals
    valid_bins = (bin_means != 0)
    if np.sum(valid_bins) > 0:
        return np.sqrt(mean_squared_error(bin_true[valid_bins], bin_means[valid_bins]))
    return None

def calculate_expected_value(y_pred, betting_odds=None):
    """Calculate expected value of predictions"""
    if betting_odds is None:
        # Return basic statistics if no odds available
        return {
            'mean': np.mean(y_pred),
            'median': np.median(y_pred),
            'std': np.std(y_pred)
        }
    else:
        # Add betting value calculations here if odds are available
        pass

def evaluate_model(y_true, y_pred, logger: logging.Logger = None):
    """
    Evaluate model performance using multiple metrics
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        betting_odds: Optional betting odds data
        logger: Optional logger instance. If None, logging will be skipped
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create default logger if none provided
    if logger is None:
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    # Calculate metrics one by one with logging
    poisson_deviance = calculate_poisson_deviance(y_true, y_pred)
    logger.info(f"Poisson deviance: {poisson_deviance}")
    
    goals_diff_dist = analyze_goals_difference(y_true, y_pred)
    logger.info(f"Goals difference distribution: {goals_diff_dist}")
    
    calibration = calculate_calibration_score(y_true, y_pred)
    logger.info(f"Calibration score: {calibration}")
    
    # exp_value = calculate_expected_value(y_pred, betting_odds)
    # logger.info(f"Expected value metrics: {exp_value}")

    metrics = {
        'poisson_deviance': poisson_deviance,
        'goals_difference_distribution': goals_diff_dist, 
        'calibration_score': calibration
        # 'expected_value': exp_value
    }
    return metrics

def calculate_prediction_intervals(model, data, confidence=0.95, logger: logging.Logger = None):
    """
    Calculate confidence intervals for predictions using bootstrap resampling.
    
    Args:
        model: The trained model
        data: Input data for predictions (pandas DataFrame or numpy array)
        confidence: Confidence level (default 0.95 for 95% intervals)
        logger: Logger instance for logging messages
        
    Returns:
        Dictionary containing confidence and uncertainty metrics
    """
    try:
        # Convert data to numpy array if DataFrame
        if hasattr(data, 'values'):
            data = data.values
            
        n_iterations = 100
        n_samples = data.shape[0]
        predictions = np.zeros((n_iterations, n_samples))
        
        for i in range(n_iterations):
            # Bootstrap sample with replacement
            indices = np.random.randint(0, n_samples, n_samples)
            sample = data[indices]
            predictions[i,:] = model.predict(sample)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Calculate confidence as inverse of prediction standard deviation
        confidence_scores = 1 / (1 + std_pred)
        
        # Calculate uncertainty as the width of the prediction interval
        uncertainty = np.percentile(predictions, 97.5, axis=0) - np.percentile(predictions, 2.5, axis=0)
        
        return {
            'confidence': confidence_scores,
            'uncertainty': std_pred
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Error occurred while calculate_prediction_intervals: {str(e)}")
        
        print(f"Error occurred while calculate_prediction_intervals: {str(e)}")
        # Return default values instead of None
        n_samples = data.shape[0] if hasattr(data, 'shape') else len(data)
        return {
            'confidence': np.full(n_samples, 0.5),  # Medium confidence
            'uncertainty': np.full(n_samples, 1.0)   # High uncertainty
        }

def analyze_ensemble_diversity(estimators, test_data=None, logger: logging.Logger = None):
    """
    Analyze diversity among ensemble members using correlation and predictions.
    
    Args:
        estimators: List of estimators in the ensemble
        test_data: Test data to use for predictions. If None, returns empty metrics.
        
    Returns:
        Dictionary of diversity metrics
    """
    if test_data is None:
        if logger:
            logger.error("Test data is None, returning empty diversity metrics")
        
        print(f"Test data is None, returning empty diversity metrics")
        return {
            'mean_correlation': 0.0,
            'min_correlation': 0.0, 
            'max_correlation': 0.0
        }
        
    n_estimators = len(estimators)
    correlations = np.zeros((n_estimators, n_estimators))
    
    # Calculate pairwise correlations between estimator predictions
    for i in range(n_estimators):
        for j in range(i+1, n_estimators):
            pred_i = estimators[i].predict(test_data)
            pred_j = estimators[j].predict(test_data)
            correlations[i,j] = np.corrcoef(pred_i, pred_j)[0,1]
            correlations[j,i] = correlations[i,j]
            
    return {
        'mean_correlation': np.mean(correlations[np.triu_indices(n_estimators, k=1)]),
        'min_correlation': np.min(correlations[np.triu_indices(n_estimators, k=1)]),
        'max_correlation': np.max(correlations[np.triu_indices(n_estimators, k=1)])
    }

def apply_dynamic_weights(base_predictions, recent_performance, window=5, logger: logging.Logger = None):
    """
    Apply dynamic weights to predictions based on recent model performance.
    
    Args:
        base_predictions: Original model predictions
        recent_performance: DataFrame containing last 100 predictions with their errors
        window: Number of recent predictions to consider
        
    Returns:
        Dictionary containing weighted predictions
    """
    # Validate window parameter
    if not isinstance(window, int) or window < 0:
        if logger:
            logger.error("window must be an integer 0 or greater")
        print("window must be an integer 0 or greater")
        return base_predictions

    # No need to validate lengths since recent_performance only contains last 100 predictions
    # while base_predictions contains all predictions to be made
    if len(recent_performance) == 0:
        if logger:
            logger.error("Recent performance data is empty")
        print("Recent performance data is empty") 
        return base_predictions
        
    # Calculate mean absolute error on last 100 predictions
    rolling_mae = recent_performance['abs_error'].mean()
    
    # Convert errors to weights (lower error = higher weight)
    # weights = 1 / (rolling_mae + 1e-6)  # Add small constant to avoid division by zero
    # weights = weights / weights.sum()  # Normalize weights
    
    # # Ensure weights array matches predictions shape by padding with mean weight
    # if len(weights) < len(base_predictions):
    #     mean_weight = weights.mean()
    #     weights = pd.Series([mean_weight] * len(base_predictions))
    
    # Apply weights to predictions
    weighted_predictions_positive = base_predictions + rolling_mae
    weighted_predictions_negative = base_predictions - rolling_mae
    
    return {
            'weighted_predictions_positive': weighted_predictions_positive,
            'weighted_predictions_negative': weighted_predictions_negative
        }

def get_recent_performance(model_type: str, window: int = 10, logger: logging.Logger = None) -> pd.DataFrame:
    """
    Get recent model performance data from saved predictions and actual results.
    
    Args:
        model_type (str): Type of model (e.g. 'away_goals', 'home_goals')
        window (int): Number of recent predictions to analyze
        logger (logging.Logger): Logger for logging information and errors
        
    Returns:
        pd.DataFrame: DataFrame containing recent predictions and actual values with columns:
            - prediction: Model's predicted value
            - actual: Actual observed value 
            - abs_error: Absolute error between prediction and actual
    """
    try:
        if logger is None:
            logger = logging.getLogger(__name__)
            
        # Load previously saved predictions file
        predictions_file = f'./made_predictions/predictions_stacked_2fit_merged.xlsx'
        if not os.path.exists(predictions_file):
            logger.error(f"Predictions file not found at {predictions_file}")
            return pd.DataFrame(columns=['prediction', 'actual', 'abs_error'])
            
        recent_data = pd.read_excel(predictions_file)
        
        # Validate required columns exist
        required_cols = ['real_score']
        if model_type == 'away_goals':
            required_cols.extend(['away_goals_prediction', 'away_goals'])
        if model_type == 'home_goals':
            required_cols.extend(['home_goals_prediction', 'home_goals'])
        if model_type == 'match_outcome':
            required_cols.extend(['match_outcome_prediction', 'match_outcome'])
        
        missing_cols = [col for col in required_cols if col not in recent_data.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame(columns=['prediction', 'actual', 'abs_error'])
        
        # Filter for rows with actual results
        recent_data = recent_data.dropna(subset=['real_score'])
        
        if len(recent_data) == 0:
            logger.warning("No records found with actual results")
            return pd.DataFrame(columns=['prediction', 'actual', 'abs_error'])
            
        # Get last n predictions
        recent_data = recent_data.tail(window)
        
        # Calculate absolute error
        if model_type == 'away_goals':
            recent_data['prediction'] = recent_data['away_goals_prediction'] 
            recent_data['actual'] = recent_data['away_goals']
        elif model_type == 'home_goals':
            recent_data['prediction'] = recent_data['home_goals_prediction']
            recent_data['actual'] = recent_data['home_goals']
        if model_type == 'match_outcome':
            recent_data['prediction'] = recent_data['match_outcome_prediction']
            recent_data['actual'] = recent_data['match_outcome']
            
        recent_data['abs_error'] = np.abs(recent_data['prediction'] - recent_data['actual'])
        
        return recent_data[['prediction', 'actual', 'abs_error']]
        
    except Exception as e:
        if logger:
            logger.error(f"Error getting recent performance data: {str(e)}")
        # Return empty DataFrame with required columns if error
        return pd.DataFrame(columns=['prediction', 'actual', 'abs_error'])

def enhance_data_quality(data: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Implement sophisticated data cleaning and validation.
    
    Args:
        data (pd.DataFrame): Input DataFrame to clean
        logger (logging.Logger): Logger instance
        
    Returns:
        pd.DataFrame: Cleaned and enhanced DataFrame
    """
    try:
        initial_shape = data.shape
        logger.info(f"Starting data quality enhancement. Initial shape: {initial_shape}")
        
        # 1. Remove statistical outliers
        numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
        for column in numerical_features:
            if column not in ['running_id', 'season_encoded', 'league_encoded', 
                            'home_encoded', 'away_encoded', 'venue_encoded']:
                mean = data[column].mean()
                std = data[column].std()
                data = data[
                    (data[column] >= mean - 3*std) & 
                    (data[column] <= mean + 3*std)
                ]
        
        logger.info(f"Outlier removal complete. New shape: {data.shape}")
        
        # 2. Smart imputation
        categorical_features = data.select_dtypes(include=['object']).columns
        
        # Categorical imputation
        for col in categorical_features:
            data[col].fillna(data[col].mode()[0], inplace=True)
            
        # Numerical imputation using KNN
        imputer = KNNImputer(n_neighbors=5)
        data[numerical_features] = imputer.fit_transform(data[numerical_features])
        
        logger.info(f"Imputation complete. Missing values: {data.isnull().sum().sum()}")
        
        # 3. Feature validation
        # Remove features with zero variance
        variance = data[numerical_features].var()
        zero_variance_features = variance[variance == 0].index
        data = data.drop(columns=zero_variance_features)
        
        # 4. Add data quality metrics
        quality_metrics = {
            'rows_removed': initial_shape[0] - data.shape[0],
            'missing_values_filled': data.isnull().sum().sum(),
            'zero_variance_features': len(zero_variance_features)
        }
        
        logger.info(f"Data quality metrics: {quality_metrics}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error in enhance_data_quality: {str(e)}")
        raise
    
    
    
    
    
    # def enhance_predictions(base_predictions, model, recent_performance, data, logger):

def prepare_data_for_optimization(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Prepare data for hyperparameter optimization by handling missing values,
    scaling features, and validating data integrity.
    
    Args:
        X (np.ndarray): Input features array
        y (np.ndarray): Target values array
        
    Returns:
        tuple: Processed (X, y) arrays ready for optimization
    """
    # Handle missing values
    if np.isnan(X).any():
        imputer = KNNImputer(n_neighbors=5)
        X = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Ensure y is 1D array
    y = np.ravel(y)
    
    # Remove any remaining invalid values
    mask = ~(np.isnan(y) | np.isinf(y))
    X = X[mask]
    y = y[mask]
    
    return X, y

def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, logger: logging.Logger) -> dict:
    """Enhanced hyperparameter optimization"""
    try:
        logger.info("Starting hyperparameter optimization")
        gc.collect()
        
        # Configure resources
        resources = configure_resources(X.shape)
        
        # Prepare data
        X, y = prepare_data_for_optimization(X, y)
        
        # Create pipeline components
        estimators = create_estimators(X)
        cv_strategy = create_cv_strategy(X, y)
        pipeline = create_pipeline(estimators, resources)
        
        # Create study with tracking
        study, callback = create_study_with_tracking()
        
        # Define objective with enhanced parameter space
        def objective(trial):
            params = get_parameter_space(trial)
            pipeline.set_params(**params)
            
            # Quick evaluation
            # quick_scores = evaluate_quick(pipeline, X, y)  # This line causes the NameError
            # Replace or remove the above line
            # Example: quick_scores = some_other_evaluation_function(pipeline, X, y)
            
            # Full evaluation
            scores = evaluate_full(pipeline, X, y, cv_strategy)
            return compute_objective_value(scores)
        
        # Run optimization with improved error handling
        best_params = safe_optimize(objective, study, n_trials=50, logger=logger)
        
        # Analyze and log results
        analyze_and_log_results(study, logger)
        
        return best_params
        
    except Exception as e:
        logger.error(f"Error in optimize_hyperparameters: {str(e)}")
        raise

def analyze_optimization_results(study, logger):
    """Analyze and visualize optimization results."""
    try:
        
        
        # Plot optimization history
        history_plot = vis.plot_optimization_history(study)
        
        # Plot parameter importances
        param_importance = vis.plot_param_importances(study)
        
        # Plot parallel coordinate plot
        parallel_plot = vis.plot_parallel_coordinate(study)
        
        # Plot slice plot
        slice_plot = vis.plot_slice(study)
        
        # Save visualization results
        plots = {
            'history': history_plot,
            'importance': param_importance,
            'parallel': parallel_plot,
            'slice': slice_plot
        }
        
        # Log optimization results
        logger.info(f"Best trial value: {study.best_value}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return plots
        
    except Exception as e:
        logger.error(f"Error in optimization analysis: {str(e)}")
        return None

# Define MLflow callback here instead
def mlflow_callback(study, trial):
    trial_value = trial.value if trial.value is not None else float('nan')
    mlflow.log_params(trial.params)
    mlflow.log_metrics({'value': trial_value})

def create_study_with_tracking():
    """Create Optuna study with MLflow tracking."""
    
    # Custom sampler for better exploration
    sampler = optuna.samplers.CmaEsSampler(
        seed=42,
        n_startup_trials=10,
        independent_sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=50,
            reduction_factor=3
        )
    )
    
    # Return both study and callback
    return study, mlflow_callback

def safe_optimize(objective, study, n_trials, logger):
    """Safer optimization with better error handling"""
    try:
        mlflow.set_tracking_uri("file:///192.168.0.77/Betting/Chatgpt/SoccerPredictor_byRichardSzita/mlruns")
        
        for _ in range(n_trials):
            with mlflow.start_run():
                study.optimize(
                    objective,
                    n_trials=1,  # Run one trial at a time
                    callbacks=[
                        mlflow_callback,
                    ],
                    n_jobs=3,
                    catch=(ValueError, RuntimeError),
                    timeout=7200
                )
                
                # Log detailed results
                mlflow.log_metrics({
                    'best_value': study.best_value,
                    'n_trials': len(study.trials),
                    'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                    'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                })
                
        return study.best_params
            
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        if len(study.trials) > 0:
            return study.best_params
        raise

def create_cv_strategy(X, y):
    """Create adaptive cross-validation strategy"""
    n_samples = len(X)
    
    if n_samples < 1000:
        # Use leave-one-out for very small datasets
        return LeaveOneOut()
    else:
        # Use standard k-fold for larger datasets
        return KFold(n_splits=5, shuffle=True, random_state=42)

def configure_resources(X_shape):
    """Smart resource allocation"""
    total_memory = psutil.virtual_memory().total
    data_size = X_shape[0] * X_shape[1] * 4  # Assuming float32
    
    # Adjust batch size based on available memory
    max_batch_size = min(256, max(16, int(total_memory * 0.1 / data_size)))
    
    # Adjust parallel jobs based on CPU cores and memory
    n_cores = cpu_count()
    memory_factor = min(1.0, (total_memory - data_size) / total_memory)
    n_jobs = max(1, min(4, int(n_cores * memory_factor)))
    
    return {
        'batch_size': max_batch_size,
        'n_jobs': n_jobs
    }

def check_early_stopping(scores, trial, study):
    """More sophisticated early stopping logic"""
    train_mse = -scores['train_mse'].mean()
    val_mse = -scores['test_mse'].mean()
    
    # Dynamic threshold based on study progress
    n_trials = len(study.trials)
    threshold_multiplier = max(1.2, 2.0 * np.exp(-n_trials / 20))  # Relaxes over time
    
    # Multiple stopping conditions
    conditions = [
        (val_mse > study.best_value * threshold_multiplier, "poor performance"),
        (train_mse < val_mse * 0.3, "severe overfitting"),
        (np.isnan(val_mse), "numerical instability")
    ]
    
    for condition, reason in conditions:
        if condition:
            return True, reason
    return False, None

def get_parameter_space(trial): 
    """Define a more sophisticated parameter space"""
    params = {
        # LightGBM parameters with more nuanced ranges
        'stacking__lgb__n_estimators': trial.suggest_int('lgb_n_estimators', 100, 1000, step=50),
        'stacking__lgb__learning_rate': trial.suggest_float('lgb_lr', 1e-4, 0.1, log=True),
        'stacking__lgb__max_depth': trial.suggest_int('lgb_depth', 3, 15),
        'stacking__lgb__min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 100),
        'stacking__lgb__subsample': trial.suggest_float('lgb_subsample', 0.5, 1.0),
        
        # XGBoost with additional parameters
        'stacking__xgb__n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000, step=50),
        'stacking__xgb__learning_rate': trial.suggest_float('xgb_lr', 1e-4, 0.1, log=True),
        'stacking__xgb__max_depth': trial.suggest_int('xgb_max_depth', 3, 15),
        'stacking__xgb__subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
        'stacking__xgb__colsample_bytree': trial.suggest_float('xgb_colsample', 0.5, 1.0),
        
        # Neural Network parameters updated to match the new structure
        'stacking__nn__model__n_layers': trial.suggest_int('nn_layers', 2, 5),
        'stacking__nn__model__units': trial.suggest_int('nn_units', 32, 512, log=True),
        'stacking__nn__fit__epochs': trial.suggest_int('nn_epochs', 50, 200),
        'stacking__nn__fit__batch_size': trial.suggest_int('nn_batch', 16, 256, log=True),
    }
    return params

def create_estimators(X: np.ndarray) -> list:
    """
    Create a list of estimators for the stacked model.
    
    Args:
        X (np.ndarray): Input features array to determine input dimensions for NN
        
    Returns:
        list: List of tuples containing (name, estimator) pairs
    """
    try:
        # Create base estimators
        lgb_regressor = create_lgb_regressor()
        xgb_regressor = create_xgb_regressor()
        nn_regressor = create_nn_regressor(X)
        rf_regressor = create_rf_regressor()
        ada_regressor = create_ada_regressor()
        gbm_regressor = create_gbm_regressor()
        elasticnet_regressor = create_elasticnet_regressor()
        
        # Create estimator list with logging wrappers
        estimators = [
            ('lgb', lgb_regressor),
            ('xgb', xgb_regressor),
            ('nn', nn_regressor),
            ('rf', rf_regressor),
            ('ada', ada_regressor),
            ('gbm', gbm_regressor),
            ('elasticnet', elasticnet_regressor)
        ]
        
        return estimators
        
    except Exception as e:
        logger.error(f"Error creating estimators: {str(e)}")
        raise

def create_final_estimator() -> VotingRegressor:
    """
    Creates the final estimator for the stacking ensemble.
    
    Returns:
        VotingRegressor: A voting regressor that combines Ridge, Lasso, and ElasticNet
    """
    return VotingRegressor([
        ('ridge', Ridge(alpha=0.1)),
        ('lasso', Lasso(alpha=0.1)),
        ('elasticnet', ElasticNet(alpha=0.1, l1_ratio=0.5))
    ])

def create_lgb_regressor():
    """Create and return a configured LightGBM regressor"""
    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=12,
        num_leaves=31,
        random_state=42,
        force_col_wise=True,
        n_jobs=-1,
        min_split_gain=1e-5,          # Keep only this one
        min_child_samples=20,          
        min_child_weight=1e-3,         
        reg_alpha=0.1,                
        reg_lambda=0.1,               
        max_bin=255,                  
        colsample_bytree=0.8,         
        subsample=0.8,                
        subsample_freq=5,             
        metric='l2',                  
        verbose=1
    )

def create_xgb_regressor():
    """Create and return a configured XGBoost regressor"""
    return XGBRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        nthread=-1
    )

def create_nn_regressor(X):
    """Create and return a configured Neural Network regressor"""
    # Create callbacks explicitly
    reduce_lr = CustomReduceLROnPlateau(
        monitor='loss', 
        factor=0.2, 
        patience=15, 
        min_lr=0.00001
    )
    
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    )
    
    return KerasRegressor(
        model=create_enhanced_neural_network,
        model__input_dim=X.shape[1],
        model__n_layers=3,  # Default value
        model__units=256,   # Default value
        fit__epochs=200,
        fit__batch_size=128,
        fit__validation_split=0.2,
        fit__callbacks=[reduce_lr, early_stopping],
        fit__verbose=1
    )

def create_rf_regressor():
    """Create and return a configured Random Forest regressor"""
    return RandomForestRegressor(
        n_estimators=2000,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

def create_ada_regressor():
    """Create and return a configured AdaBoost regressor"""
    return AdaBoostRegressor(
        base_estimator=DecisionTreeRegressor(
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=4
        ),
        n_estimators=1000,
        learning_rate=0.01,
        loss='square',
        random_state=42
    )

def create_elasticnet_regressor():
    """Create and return an optimized ElasticNet regressor for regression tasks.
    
    The ElasticNet combines L1 and L2 regularization to handle correlated features
    and prevent overfitting. This implementation uses optimized hyperparameters
    and performance settings.
    
    Returns:
        ElasticNet: Configured ElasticNet regressor instance
    
    Note:
        - alpha=0.1 provides moderate regularization
        - l1_ratio=0.5 balances L1/L2 penalties
        - precompute=True speeds up fitting for dense data
        - max_iter increased to 1000 to ensure convergence
    """
    try:
        return ElasticNet(
            alpha=0.1,  # Regularization strength
            l1_ratio=0.5,  # Balance between L1/L2
            max_iter=1000,  # Increased to ensure convergence
            tol=0.0001,  # Tighter tolerance for better accuracy
            random_state=42,
            selection='cyclic',  # More reliable than random
            precompute=True,  # Speeds up on dense data
            warm_start=True,  # Reuse previous solution
            fit_intercept=True,
            normalize=False,  # Data should be scaled externally
            positive=False  # Allow negative coefficients
        )
    except Exception as e:
        logger.error(f"Error creating ElasticNet regressor: {str(e)}")
        raise

def create_gbm_regressor():
    """Create and return a configured Gradient Boosting regressor"""
    return GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )

def create_pipeline(estimators: list, resources: dict) -> Pipeline:
    """
    Create a scikit-learn pipeline with the given estimators and resources.
    
    Args:
        estimators (list): List of (name, estimator) tuples
        resources (dict): Dictionary containing resource configurations
        
    Returns:
        Pipeline: Configured scikit-learn pipeline
    """
    try:
        # Create the stacking regressor with the estimators
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=create_final_estimator(),
            n_jobs=resources['n_jobs'],
            cv=5,
            verbose=1
        )
        # Create and return the full pipeline
        return Pipeline([
            ('scaler', StandardScaler()),
            ('stacking', stacking)
        ])
        
    except Exception as e:
        logger.error(f"Error creating pipeline: {str(e)}")
        raise  

def evaluate_full(pipeline, X, y, cv_strategy, logger=None):
    """
    Perform full model evaluation using cross-validation.
    
    Args:
        pipeline: The sklearn pipeline to evaluate
        X: Input features
        y: Target values
        cv_strategy: Cross-validation strategy
        logger: Optional logger instance
        
    Returns:
        dict: Dictionary containing evaluation scores
    """
    try:
        # Create default logger if none provided
        if logger is None:
            logger = logging.getLogger(__name__)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
        # Perform cross-validation with multiple metrics
        scores = cross_validate(
            pipeline,
            X, 
            y,
            cv=cv_strategy,
            scoring={
                'mse': 'neg_mean_squared_error',
                'mae': 'neg_mean_absolute_error',
                'r2': 'r2'
            },
            return_train_score=True,
            n_jobs=-1,
            error_score='raise'  # Added to help with debugging
        )
        
        return scores
        
    except Exception as e:
        if logger:
            logger.error(f"Error in evaluate_full: {str(e)}")
        # Return worst possible scores instead of None
        return {
            'train_mse': np.array([-float('inf')]),
            'test_mse': np.array([-float('inf')]),
            'train_mae': np.array([-float('inf')]),
            'test_mae': np.array([-float('inf')]),
            'train_r2': np.array([-float('inf')]),
            'test_r2': np.array([-float('inf')])
        }

def compute_objective_value(scores):
    """
    Compute the objective value to minimize from cross-validation scores.
    
    Args:
        scores: Dictionary of cross-validation scores
        
    Returns:
        float: The objective value to minimize
    """
    # Convert negative MSE back to positive
    test_mse = -scores['test_mse'].mean()
    train_mse = -scores['train_mse'].mean()
    
    # Add regularization to prevent overfitting
    # Penalize large gaps between train and test performance
    regularization = abs(train_mse - test_mse) * 0.1
    
    return test_mse + regularization

# Update the objective function to use these new functions
def objective(trial):
    """Objective function for hyperparameter optimization"""
    params = get_parameter_space(trial)
    pipeline.set_params(**params)
    
    # Full evaluation
    scores = evaluate_full(pipeline, X, y, cv_strategy)
    return compute_objective_value(scores)



#     Returns:
#         Dictionary containing enhanced predictions and metrics
#     """
#     # Calculate prediction intervals
#     prediction_intervals = calculate_prediction_intervals(model, data)
    
#     # Analyze ensemble diversity
#     diversity_scores = analyze_ensemble_diversity(model.estimators_)  
    
#     # Apply dynamic weighting
#     weighted_predictions = apply_dynamic_weights(base_predictions, recent_performance)
    
#     return {
#         'base_predictions': base_predictions,
#         'weighted_predictions': weighted_predictions,
#         'prediction_intervals': prediction_intervals,
#         'diversity_scores': diversity_scores
#     }