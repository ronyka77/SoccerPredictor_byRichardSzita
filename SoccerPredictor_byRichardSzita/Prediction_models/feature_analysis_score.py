import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Load your data
file_path = f'./data/model_data_training_newPoisson.xlsx'  # Adjust path as needed
data = pd.read_excel(file_path)

if data.empty:
    print("Error: The dataset is empty after cleaning. Please check the clean_data() function.")
else:
    print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")

# Drop non-numeric columns
df_numeric = data.select_dtypes(include=[np.number])
# Specify the columns to drop
columns_to_drop = ['Unnamed: 0.1','Unnamed: 0.2','Unnamed: 0','match_outcome',  
                    'draw', 'away_win', 'home_win','away_points', 'home_points','HomeTeam_last_away_match','AwayTeam_last_home_match',
                    'home_points_rolling_avg','away_points_rolling_avg','home_advantage',
                    'Odd_Home','Odds_Draw','Odd_Away']
df_numeric = df_numeric.drop(columns=columns_to_drop, errors='ignore')
df_numeric = df_numeric.dropna()
print(df_numeric.isna().sum())
# Compute correlation matrix for home goals and away goals
correlation_matrix_home = df_numeric.corr()['home_goals'].sort_values(ascending=False)
correlation_matrix_away = df_numeric.corr()['away_goals'].sort_values(ascending=False)

# correlation_df_home = 

# Plot heatmap for home goals
# plt.figure(figsize=(12, 8))
# sns.heatmap(df_numeric2.corr(), annot=False, cmap='coolwarm')
# plt.title("Correlation Heatmap - Home Goals")

# Plot heatmap for away goals
plt.figure(figsize=(12, 8))
correlation_matrix = df_numeric.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap - Away Goals")
plt.show()

# Define correlation threshold for selecting important features
correlation_threshold = 0.05

# Filter features based on correlation threshold for home goals
important_features_home = correlation_matrix_home[abs(correlation_matrix_home) > correlation_threshold].index.tolist()

# Filter features based on correlation threshold for away goals
important_features_away = correlation_matrix_away[abs(correlation_matrix_away) > correlation_threshold].index.tolist()

print("Important Features for Home Goals:\n", important_features_home)
print("Important Features for Away Goals:\n", important_features_away)

# Ensure there are features to train on
if len(important_features_home) <= 1:
    print("No significant features found for home goals prediction. Adjust the correlation threshold or check data.")
else:
    # Create datasets for home goals prediction
    X_home = df_numeric[important_features_home].drop(columns=['home_goals', 'away_goals'], errors='ignore')
    y_home = df_numeric['home_goals']

    # Train/test split for home goals model
    X_train_home, X_test_home, y_train_home, y_test_home = train_test_split(X_home, y_home, test_size=0.3, random_state=42)

    # Build Random Forest for home goals
    rf_home = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_home.fit(X_train_home, y_train_home)

    # Evaluate Home Goals Model
    y_pred_home = rf_home.predict(X_test_home)
    mse_home = mean_squared_error(y_test_home, y_pred_home)
    r2_home = r2_score(y_test_home, y_pred_home)
    print("Home Goals Prediction - Mean Squared Error:", mse_home)
    print("Home Goals Prediction - R2 Score:", r2_home)

# Create datasets for away goals prediction
X_away = df_numeric[important_features_away].drop(columns= ['home_goals','away_goals'],errors='ignore')
y_away = df_numeric['away_goals']

# Train/test split for away goals model
X_train_away, X_test_away, y_train_away, y_test_away = train_test_split(X_away, y_away, test_size=0.2, random_state=42)

# Build Random Forest for away goals
rf_away = RandomForestRegressor(n_estimators=100, random_state=42)
rf_away.fit(X_train_away, y_train_away)
y_pred_away = rf_away.predict(X_test_away)

# Evaluate Away Goals Model
mse_away = mean_squared_error(y_test_away, y_pred_away)
r2_away = r2_score(y_test_away, y_pred_away)
print("Away Goals Prediction - Mean Squared Error:", mse_away)
print("Away Goals Prediction - R2 Score:", r2_away)

# Feature Importance for Home Goals Model
feature_importances_home = pd.Series(rf_home.feature_importances_, index=X_train_home.columns)
print("Feature Importances for Home Goals Model:")
print(feature_importances_home.sort_values(ascending=False))

# Feature Importance for Away Goals Model
feature_importances_away = pd.Series(rf_away.feature_importances_, index=X_train_away.columns)
print("Feature Importances for Away Goals Model:")
print(feature_importances_away.sort_values(ascending=False))
