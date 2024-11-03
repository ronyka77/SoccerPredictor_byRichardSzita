import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
# Add the folder containing your script to the Python path
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
from data_scaling import scale_data

# Load your data
file_path = f'C:/Users/T480/Documents/Betting/Chatgpt/Model_build1/data/model_data.xlsx'  # Adjust path as needed
data = pd.read_excel(file_path)

if data.empty:
    print("Error: The dataset is empty after cleaning. Please check the clean_data() function.")
else:
    print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")

# Select features and target for classification (predicting match outcome)
X = data.drop(columns=['match_outcome','_id','Home','Away','Home_xG','Away_xG',
                       'unique_id','home_win','away_win','draw','away_points','home_points',
                       'home_goal_difference','away_goal_difference','home_goals','away_goals',
                       'home_saves_accuracy','away_saves_accuracy'])  # Adjust as needed
# Check if all columns are numeric
# print(X.dtypes)

y = data['match_outcome']  # Target variable for classification

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

# 2. Train a Random Forest Classifier for feature importance analysis
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# 3. Extract and display feature importances
feature_importances = rf_classifier.feature_importances_
feature_names = X.columns

# Create a sorted list of features and their importance scores
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_names = [feature_names[idx] for idx in sorted_indices]
sorted_importances = feature_importances[sorted_indices]

# Display the top N important features
N = 20  # Adjust N to see more/less features
print(f"Top {N} Important Features:\n")
for i in range(N):
    print(f"{sorted_feature_names[i]}: {sorted_importances[i]:.4f}")
for i in range(N):   
    print(f"{sorted_feature_names[i]}")
    
# Plot feature importances
# plt.figure(figsize=(10, 6))
# plt.barh(sorted_feature_names[:N], sorted_importances[:N], color='skyblue')
# plt.xlabel("Feature Importance Score")
# plt.title(f"Top {N} Important Features")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()

# 5. Save the updated dataset with new features
data.to_csv('./Model_build1/updated_data_with_new_features.csv', index=False)
print("\nFeature engineering completed and dataset saved with new features!")

