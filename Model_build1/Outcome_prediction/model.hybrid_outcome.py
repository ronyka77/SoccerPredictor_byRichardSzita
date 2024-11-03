import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasRegressor  # Ensure compatibility with Keras
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import joblib
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


# Set up logging
log_file_path = './Model_build1/Outcome_prediction/log/hybrid_model_training.log'
logging.basicConfig(filename=log_file_path, 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set directories and files
model_dir = "./models/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

logging.info("Model directory checked/created.")

# Load your feature engineered data
file_path = './Model_build1/data/model_data_training.csv'
data = pd.read_csv(file_path)
logging.info(f"Data loaded from {file_path} with shape: {data.shape}")

data = data.drop(columns=['running_id','home_goals','away_goals','away_points', 'draw', 'away_win', 'home_points', 'home_win'],errors='ignore')

# Select all numeric features dynamically from the dataset
numeric_features = data.drop(columns=['match_outcome']).select_dtypes(include=['float64', 'int64']).columns.tolist()
logging.info(f"Numeric features selected: {numeric_features}")

# Prepare data
def prepare_data(data, features):
    model_data = data[features]
    imputer = IterativeImputer(random_state=42)
    model_data_imputed = imputer.fit_transform(model_data)
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

# Save the scaler
joblib.dump(scaler, os.path.join(model_dir, 'scaler_outcome.pkl'))
logging.info("Scaler saved for future use.")

# Feature Selection (RFE)
selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=20)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
logging.info("Feature selection using RFE completed.")

# Save the RFE selector for later use in predictions
joblib.dump(selector, os.path.join(model_dir, 'rfe_selector.pkl'))  # <--- Save the RFE selector
logging.info("RFE selector saved.")

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
logging.info(f"SMOTE applied to handle class imbalance. Resampled train shape: {X_train_resampled.shape}")

# Define models
# 1. Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# 2. Support Vector Machine (SVM)
svm_classifier = SVC(probability=True, random_state=42, class_weight='balanced')

# 3. Neural Network (wrapped with KerasClassifier to be compatible with scikit-learn)
def create_neural_network():
    model = Sequential()
    model.add(Dense(128, input_dim=X_train_selected.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrapping the Keras model
nn_classifier = KerasClassifier(build_fn=create_neural_network, epochs=20, batch_size=32, verbose=1)

# 4. XGBoost Classifier
xgb_classifier = XGBClassifier()

# nn_regressor_outcome = KerasRegressor(
#     model=create_neural_network,  # Passing the function reference, not calling it
#     model__input_dim=X_train_selected.shape[1],  # Arguments for create_neural_network
#     epochs=30,
#     batch_size=32,
#     verbose=1
# )

# Stacking Ensemble
estimators = [
    ('rf', rf_classifier),
    ('svm', svm_classifier),
    ('xgb', xgb_classifier)
]

stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_model.fit(X_train_resampled, y_train_resampled)
logging.info("Stacking model trained successfully.")

# Train Neural Network separately and save
nn_model = create_neural_network()
nn_model.fit(X_train_resampled, y_train_resampled, epochs=20, batch_size=32)
nn_model.save(os.path.join(model_dir, 'keras_nn_model_outcome.h5'))  # Save neural network model separately
logging.info("Neural Network model trained and saved.")

# Evaluation
y_pred = stacking_model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
logging.info(f"Stacking Model Accuracy: {accuracy}")
logging.info(f"Stacking Model Classification Report:\n{report}")

# Cross-validation with Repeated Stratified K-Fold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
cv_scores = cross_val_score(stacking_model, X_train_resampled, y_train_resampled, cv=cv)
logging.info(f"Cross-Validation Scores: {cv_scores}")

# Hyperparameter tuning for Random Forest and SVM using GridSearch
param_grid = {
    'rf__n_estimators': [200],
    'rf__max_depth': [20],
    'svm__C': [0.1, 1],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': [0.1]
}

# Set `n_jobs=1` to avoid Keras model pickling issues
grid_search = GridSearchCV(estimator=stacking_model, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)
logging.info("GridSearchCV completed for hyperparameter tuning.")

best_params = grid_search.best_params_
logging.info(f"Best Params from GridSearch: {best_params}")

# Save models
joblib.dump(stacking_model, os.path.join(model_dir, 'stacking_model_outcome.pkl'))
joblib.dump(grid_search.best_estimator_, os.path.join(model_dir, 'stacking_model_tuned_outcome.pkl'))
logging.info("Models saved successfully.")
