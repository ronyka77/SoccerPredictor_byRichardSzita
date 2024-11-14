import pandas as pd
import os

class ELOCalculator:
    """
    A utility class for calculating and adding ELO scores to soccer match data.
    """
    def __init__(self):
        """Initialize the ELOCalculator with required paths and settings."""
        # Set directories and files
        self.model_dir = "./models/"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Define data paths
        self.training_data_path = './data/model_data_training_newPoisson.xlsx'
        self.prediction_data_path = './data/model_data_prediction_newPoisson.xlsx'
        self.merged_data_path = './data/merged_data_prediction.csv'
        
        # Define export paths
        self.training_export_path = './data/model_data_training_newPoisson.xlsx'
        self.prediction_export_path = './data/model_data_prediction_newPoisson.xlsx'
        self.merged_export_path = './data/merged_data_prediction.csv'
        
        # ELO settings
        self.INITIAL_ELO = 1500
        self.K_FACTOR = 40  # K-Factor for ELO updates
        self.elo_ratings = {}  # Dictionary to store ELO ratings

    def calculate_expected_score(self, elo_a, elo_b):
        """Calculate expected score for team A based on ELO ratings."""
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def update_elo(self, elo_a, elo_b, score_a):
        """Update ELO rating based on match result."""
        expected_a = self.calculate_expected_score(elo_a, elo_b)
        new_elo_a = elo_a + self.K_FACTOR * (score_a - expected_a)
        return new_elo_a

    def add_elo_scores(self, matches):
        """Add ELO scores to training and prediction data."""
        # Initialize all teams with the initial ELO rating
        for team in pd.concat([matches['home_encoded'], matches['away_encoded']]).unique():
            self.elo_ratings[team] = self.INITIAL_ELO

        # Add new columns to store ELO ratings before the match
        matches['home_team_elo'] = 0.0
        matches['away_team_elo'] = 0.0

        # Process each match and update ELO ratings
        for index, row in matches.iterrows():
            home_team = row['home_encoded']
            away_team = row['away_encoded']
            home_goals = row['home_goals']
            away_goals = row['away_goals']

            # Get current ELO ratings
            home_elo = self.elo_ratings[home_team]
            away_elo = self.elo_ratings[away_team]

            # Store pre-match ELO ratings
            matches.at[index, 'home_team_elo'] = home_elo
            matches.at[index, 'away_team_elo'] = away_elo

            # Determine match outcome
            if home_goals > away_goals:
                home_score, away_score = 1, 0
            elif home_goals < away_goals:
                home_score, away_score = 0, 1
            else:
                home_score = away_score = 0.5

            # Update ELO ratings
            new_home_elo = self.update_elo(home_elo, away_elo, home_score)
            new_away_elo = self.update_elo(away_elo, home_elo, away_score)

            # Save new ratings
            self.elo_ratings[home_team] = new_home_elo
            self.elo_ratings[away_team] = new_away_elo

        return matches

    def add_elo_scores_to_merged(self, matches):
        """Add ELO scores to merged data using Poisson xG."""
        for index, row in matches.iterrows():
            home_team = row['home_encoded']
            away_team = row['away_encoded']
            home_goals = row['home_poisson_xG']
            away_goals = row['away_poisson_xG']

            # Get current ELO ratings
            home_elo = row['home_team_elo']
            away_elo = row['away_team_elo']

            # Determine match outcome based on Poisson xG
            if home_goals > away_goals:
                home_score, away_score = 1, 0
            elif home_goals < away_goals:
                home_score, away_score = 0, 1
            else:
                home_score = away_score = 0.5

            # Update ELO ratings
            new_home_elo = self.update_elo(home_elo, away_elo, home_score)
            new_away_elo = self.update_elo(away_elo, home_elo, away_score)

            # Save new ratings
            row['home_team_elo'] = new_home_elo
            row['away_team_elo'] = new_away_elo

        return matches

    def process_training_data(self):
        """Process training data with ELO calculations."""
        # Load and sort training data
        training_data = pd.read_excel(self.training_data_path)
        training_data = training_data.sort_values(by='Datum')
        
        # Calculate ELO scores
        training_data = self.add_elo_scores(training_data)
        
        # Export results
        print("\nExporting training data:")
        training_data.to_excel(self.training_export_path)
        
    def process_prediction_data(self):
        """Process prediction data with ELO calculations."""
        # Load and sort prediction data
        prediction_data = pd.read_excel(self.prediction_data_path)
        prediction_data = prediction_data.sort_values(by='Datum')
        
        # Calculate ELO scores
        prediction_data = self.add_elo_scores(prediction_data)
        
        # Export results
        print("\nExporting prediction data:")
        prediction_data.to_excel(self.prediction_export_path)
        
    def process_merged_data(self):
        """Process merged data with ELO calculations."""
        # Load and sort merged data
        merged_data = pd.read_csv(self.merged_data_path)
        merged_data = merged_data.sort_values(by='Date')
        
        # Calculate ELO scores
        merged_data = self.add_elo_scores_to_merged(merged_data)
        
        # Export results
        print("\nExporting merged data:")
        merged_data.to_csv(self.merged_export_path)
        
if __name__ == "__main__":
    calculator = ELOCalculator()
    calculator.process_training_data()
    calculator.process_prediction_data()
    calculator.process_merged_data()
