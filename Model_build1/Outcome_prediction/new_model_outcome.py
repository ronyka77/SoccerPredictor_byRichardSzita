import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBClassifier
import joblib
import numpy as np
import logging
import os
from collections import Counter

# Setup logging
log_file_path = './Model_build1/Outcome_prediction/log/outcome_model_training.log'
logging.basicConfig(filename=log_file_path,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the important features for the model
important_features = [
    'league_encoded','season_encoded',
    'home_points_rolling_avg', 'away_points_rolling_avg',
    'home_points_cumulative', 'away_points_cumulative',
    'home_goal_difference_rolling_avg', 'away_goal_difference_rolling_avg',
    'home_goal_diff_cumulative', 'away_goal_diff_cumulative',
    'home_goals_rolling_avg', 'away_goals_rolling_avg',
    'home_win_rate', 'away_win_rate',
    'home_draw_rate', 'away_draw_rate',
    'home_saves_accuracy_rolling_avg', 'away_saves_accuracy_rolling_avg',
    'home_team_average_points', 'away_team_average_points',
    'home_strength_index', 'away_strength_index', 
    'home_shots_on_target_ratio', 'away_shots_on_target_ratio',
    'home_xG_rolling_avg', 'away_xG_rolling_avg',
    'home_shots_on_target_rolling_avg', 'away_shots_on_target_rolling_avg'
]

# Load data
def load_data(file_path):
    data = pd.read_excel(file_path)
    logging.info(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data

# Prepare data for model input
def prepare_data_for_model(data, important_features):
    model_data = data[important_features]
    imputer = IterativeImputer(random_state=42)
    model_data_imputed = imputer.fit_transform(model_data)
    return pd.DataFrame(model_data_imputed, columns=important_features)

# Train/Test Split
def split_data(data, target_column):
    X = prepare_data_for_model(data, important_features)
    y = data[target_column]
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})  # Map the target labels for XGBoost
    X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=42)
    logging.info(f"Training set size: {X_train.shape[0]} rows, Test set size: {X_test.shape[0]} rows.")
    return X_train, X_test, y_train, y_test

# Train Random Forest Classifier
def train_random_forest(X_train, y_train):
    logging.info("Training baseline Random Forest Classifier...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy}")
    logging.info(f"Classification Report:\n{report}")
    return accuracy, report

# Cross-Validation
def perform_cross_validation(model, X_train, y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    logging.info(f"Cross-Validation Accuracy Scores: {cv_scores}")
    return cv_scores

# Hyperparameter Tuning with GridSearchCV for Random Forest
def tune_random_forest(X_train, y_train):
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    logging.info("Starting GridSearchCV for Random Forest...")
    rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
    grid_search_rf.fit(X_train, y_train)
    best_rf_classifier = grid_search_rf.best_estimator_
    logging.info(f"Best Hyperparameters from GridSearchCV: {grid_search_rf.best_params_}")
    return best_rf_classifier

# Hyperparameter Tuning with GridSearchCV for XGBoost
def tune_xgboost(X_train, y_train):
    param_grid_xgb = {
        'n_estimators': [200, 300],
        'max_depth': [6, 10, 20],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3]
    }
    logging.info("Starting GridSearchCV for XGBoost...")
    xgb_classifier = XGBClassifier(objective='multi:softmax', num_class=3)
    grid_search_xgb = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid_xgb, cv=5, n_jobs=-1, verbose=2)
    grid_search_xgb.fit(X_train, y_train)
    best_xgb_classifier = grid_search_xgb.best_estimator_
    logging.info(f"Best Hyperparameters from XGBoost GridSearchCV: {grid_search_xgb.best_params_}")
    return best_xgb_classifier

# Main Function
def main():
    # Load the data
    file_path = './Model_build1/data/model_data.xlsx'
    data = load_data(file_path)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data, 'match_outcome')

    # Train the Random Forest Classifier
    rf_classifier = train_random_forest(X_train, y_train)

    # Evaluate Random Forest Classifier
    accuracy_rf, report_rf = evaluate_model(rf_classifier, X_test, y_test)

    # Perform Cross-Validation
    perform_cross_validation(rf_classifier, X_train, y_train)

    # Hyperparameter Tuning for Random Forest
    tuned_rf_classifier = tune_random_forest(X_train, y_train)

    # Evaluate Tuned Random Forest Model
    accuracy_rf_tuned, report_rf_tuned = evaluate_model(tuned_rf_classifier, X_test, y_test)

    # Hyperparameter Tuning for XGBoost
    tuned_xgb_classifier = tune_xgboost(X_train, y_train)

    # Evaluate Tuned XGBoost Model
    accuracy_xgb, report_xgb = evaluate_model(tuned_xgb_classifier, X_test, y_test)

    # Save the models
    joblib.dump(tuned_rf_classifier, './models/outcome_rf_classifier_model.pkl')
    joblib.dump(tuned_xgb_classifier, './models/outcome_xgb_model.pkl')

    logging.info("Models saved successfully.")

if __name__ == "__main__":
    main()
