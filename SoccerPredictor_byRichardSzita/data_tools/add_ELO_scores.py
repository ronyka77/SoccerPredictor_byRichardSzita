import pandas as pd
import os
# Set directories and files
model_dir = "./models/"
os.makedirs(model_dir, exist_ok=True)

# Load your feature-engineered data
training_data_path =  './SoccerPredictor_byRichardSzita/data/model_data_training_newPoisson.xlsx'
prediction_data_path =  './SoccerPredictor_byRichardSzita/data/model_data_prediction_newPoisson.xlsx'
merged_data_path = './SoccerPredictor_byRichardSzita/data/merged_data_prediction.csv'
training_data = pd.read_excel(training_data_path)
prediction_data = pd.read_excel(prediction_data_path)
merged_data = pd.read_csv(merged_data_path)

training_data = training_data.sort_values(by='Datum')
prediction_data = prediction_data.sort_values(by='Datum')
merged_data = merged_data.sort_values(by='Date')

# Initialize the ELO ratings for each team
INITIAL_ELO = 1500
K_FACTOR = 40  # K-Factor for ELO updates

# Create a dictionary to store the ELO ratings for each team
elo_ratings = {}

# Function to calculate the expected score for team A
def calculate_expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

# Function to update the ELO rating based on the match result
def update_elo(elo_a, elo_b, score_a):
    expected_a = calculate_expected_score(elo_a, elo_b)
    new_elo_a = elo_a + K_FACTOR * (score_a - expected_a)
    return new_elo_a

# Example historical match data (home_team, away_team, home_goals, away_goals, match_date)

def add_elo_scores(matches):
    # Initialize all teams with the initial ELO rating
    for team in pd.concat([matches['home_encoded'], matches['away_encoded']]).unique():
        elo_ratings[team] = INITIAL_ELO

    # Add new columns to store ELO ratings before the match
    matches['home_team_elo'] = 0
    matches['away_team_elo'] = 0

    # Process each match and update ELO ratings
    for index, row in matches.iterrows():
        home_team = row['home_encoded']
        away_team = row['away_encoded']
        home_goals = row['home_goals']
        away_goals = row['away_goals']

        # Get current ELO ratings for both teams before the match
        home_elo = elo_ratings[home_team]
        away_elo = elo_ratings[away_team]

        # Store the ELO ratings before the match in the dataset
        matches.at[index, 'home_team_elo'] = home_elo
        matches.at[index, 'away_team_elo'] = away_elo

        # Determine the outcome of the match
        if home_goals > away_goals:
            home_score = 1  # Home team wins
            away_score = 0  # Away team loses
        elif home_goals < away_goals:
            home_score = 0  # Home team loses
            away_score = 1  # Away team wins
        else:
            home_score = 0.5  # Draw
            away_score = 0.5  # Draw

        # Update ELO ratings based on the match result
        new_home_elo = update_elo(home_elo, away_elo, home_score)
        new_away_elo = update_elo(away_elo, home_elo, away_score)

        # Save the new ELO ratings for the next match
        elo_ratings[home_team] = new_home_elo
        elo_ratings[away_team] = new_away_elo

        # Log the updated ratings for each team (optional)
        # print(f"{home_team} ELO: {new_home_elo:.2f}, {away_team} ELO: {new_away_elo:.2f}")

    # Final ELO ratings after processing all matches
    print("\nFinal ELO Ratings:")
    for team, rating in elo_ratings.items():
        print(f"{team}: {rating:.2f}")
    return matches

def add_elo_scores_to_merged(matches):
   
    # Process each match and update ELO ratings
    for index, row in matches.iterrows():
        home_team = row['home_encoded']
        away_team = row['away_encoded']
        home_goals = row['home_poisson_xG']
        away_goals = row['away_poisson_xG']

        # Get current ELO ratings for both teams before the match
        home_elo = row['home_team_elo']
        away_elo = row['away_team_elo']

        # Store the ELO ratings before the match in the dataset
        # matches.at[index, 'home_team_elo'] = home_elo
        # matches.at[index, 'away_team_elo'] = away_elo

        # Determine the outcome of the match
        if home_goals > away_goals:
            home_score = 1  # Home team wins
            away_score = 0  # Away team loses
        elif home_goals < away_goals:
            home_score = 0  # Home team loses
            away_score = 1  # Away team wins
        else:
            home_score = 0.5  # Draw
            away_score = 0.5  # Draw

        # Update ELO ratings based on the match result
        new_home_elo = update_elo(home_elo, away_elo, home_score)
        new_away_elo = update_elo(away_elo, home_elo, away_score)

        # Save the new ELO ratings for the next match
        row['home_team_elo'] = new_home_elo
        row['away_team_elo'] = new_away_elo

        # Log the updated ratings for each team (optional)
        # print(f"{home_team} ELO: {new_home_elo:.2f}, {away_team} ELO: {new_away_elo:.2f}")

    # Final ELO ratings after processing all matches
    print("\nFinal ELO Ratings:")
    for team, rating in elo_ratings.items():
        print(f"{team}: {rating:.2f}")
    return matches

training_data = add_elo_scores(training_data)
prediction_data = add_elo_scores(prediction_data)
# merged_data = add_elo_scores_to_merged(merged_data)
training_export_path = './SoccerPredictor_byRichardSzita/data/model_data_training_newPoisson.xlsx'
prediction_export_path = './SoccerPredictor_byRichardSzita/data/model_data_prediction_newPoisson.xlsx'
merged_export_path = './SoccerPredictor_byRichardSzita/data/merged_data_prediction.csv'
# View the dataset with ELO ratings
print("\nExporting data:")
training_data.to_excel(training_export_path)
prediction_data.to_excel(prediction_export_path)
merged_data.to_csv(merged_export_path)