import pandas as pd
import statsmodels.api as sm
import os

# Set directories and files
model_dir = "./models/"
os.makedirs(model_dir, exist_ok=True)
global merged_matches 

def add_poisson_merged(matches):
    global merged_matches 
    # Prepare the feature set for home and away team goals
    X_home = matches[['home_league_position', 'away_league_position', 'Home_goal_difference_cum', 'Away_goal_difference_cum','home_win_rate','away_win_rate','Home_points_cum','Away_points_cum']]
    X_away = matches[['away_league_position', 'home_league_position', 'Home_goal_difference_cum', 'Away_goal_difference_cum','home_win_rate','away_win_rate','Home_points_cum','Away_points_cum']]  # Note the reversal of home and away elo
    X_merged_home = merged_matches[['away_league_position', 'home_league_position', 'Home_goal_difference_cum', 'Away_goal_difference_cum','home_win_rate','away_win_rate','Home_points_cum','Away_points_cum']]  # Note the reversal of home and away elo
    X_merged_away = merged_matches[['away_league_position', 'home_league_position', 'Home_goal_difference_cum', 'Away_goal_difference_cum','home_win_rate','away_win_rate','Home_points_cum','Away_points_cum']]  # Note the reversal of home and away elo
   
    # Add constant to the feature set (intercept term)
    X_home = sm.add_constant(X_home)
    X_away = sm.add_constant(X_away)
    X_merged_home = sm.add_constant(X_merged_home)
    X_merged_away = sm.add_constant(X_merged_away)

    # Fit Poisson regression for home goals
    poisson_home = sm.GLM(matches['home_goals'], X_home, family=sm.families.Poisson())
    poisson_home_results = poisson_home.fit()

    # Fit Poisson regression for away goals
    poisson_away = sm.GLM(matches['away_goals'], X_away, family=sm.families.Poisson())
    poisson_away_results = poisson_away.fit()

    # Display the summary of the results for home and away goal predictions
    print("Poisson Regression for Home Goals:")
    print(poisson_home_results.summary())

    print("\nPoisson Regression for Away Goals:")
    print(poisson_away_results.summary())

    # Predict expected goals (xG) for home and away teams
    matches['home_poisson_xG'] = poisson_home_results.predict(X_home)
    matches['away_poisson_xG'] = poisson_away_results.predict(X_away)
    merged_matches['home_poisson_xG'] = poisson_home_results.predict(X_merged_home)
    merged_matches['away_poisson_xG'] = poisson_away_results.predict(X_merged_away)
    return matches

def add_poisson(matches):
    # Prepare the feature set for home and away team goals
    X_home = matches[['home_league_position', 'away_league_position', 'Home_goal_difference_cum', 'Away_goal_difference_cum','home_win_rate','away_win_rate','Home_points_cum','Away_points_cum']]
    X_away = matches[['away_league_position', 'home_league_position', 'Home_goal_difference_cum', 'Away_goal_difference_cum','home_win_rate','away_win_rate','Home_points_cum','Away_points_cum']]  # Note the reversal of home and away elo
   
    # Add constant to the feature set (intercept term)
    X_home = sm.add_constant(X_home)
    X_away = sm.add_constant(X_away)
 
    # Fit Poisson regression for home goals
    poisson_home = sm.GLM(matches['home_goals'], X_home, family=sm.families.Poisson())
    poisson_home_results = poisson_home.fit()

    # Fit Poisson regression for away goals
    poisson_away = sm.GLM(matches['away_goals'], X_away, family=sm.families.Poisson())
    poisson_away_results = poisson_away.fit()

    # Display the summary of the results for home and away goal predictions
    print("Poisson Regression for Home Goals:")
    print(poisson_home_results.summary())

    print("\nPoisson Regression for Away Goals:")
    print(poisson_away_results.summary())

    # Predict expected goals (xG) for home and away teams
    matches['home_poisson_xG'] = poisson_home_results.predict(X_home)
    matches['away_poisson_xG'] = poisson_away_results.predict(X_away)
    return matches

# Load your feature-engineered data
training_data_path = './data/model_data_training.csv'
prediction_data_path = './data/model_data_prediction.csv'
merged_data_path = './data/merged_data_prediction.csv'
training_data = pd.read_csv(training_data_path)
prediction_data = pd.read_csv(prediction_data_path)
merged_matches = pd.read_csv(merged_data_path)

training_data = training_data.sort_values(by='Datum')
prediction_data = prediction_data.sort_values(by='Datum')
merged_matches = merged_matches.sort_values(by='Date')
merged_matches = merged_matches.replace(',', '.', regex=True)
merged_matches['Date'] = pd.to_datetime(merged_matches['Date'], errors='coerce')
# Select columns that are of object type (excluding 'Date' which is now datetime)
cols_to_convert = merged_matches.select_dtypes(include=['object']).columns
# Apply pd.to_numeric to these columns
merged_matches[cols_to_convert] = merged_matches[cols_to_convert].apply(pd.to_numeric, errors='coerce')
# print(merged_matches.dtypes)

training_data = add_poisson(training_data)
prediction_data = add_poisson_merged(prediction_data)
training_export_path = './data/model_data_training_newPoisson.xlsx'
prediction_export_path = './data/model_data_prediction_newPoisson.xlsx'
merged_export_path = './data/merged_data_prediction.csv'
# View the dataset with ELO ratings
print("\nExporting data:")
training_data.to_excel(training_export_path)
prediction_data.to_excel(prediction_export_path)
merged_matches.to_csv(merged_export_path)

