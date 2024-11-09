import pandas as pd
import statsmodels.api as sm
import os

class PoissonXGCalculator:
    """
    A utility class for calculating Poisson expected goals (xG) for soccer matches.
    """
    def __init__(self):
        """Initialize the PoissonXGCalculator with required paths and directories."""
        self.model_dir = "./models/"
        os.makedirs(self.model_dir, exist_ok=True)
        self.merged_matches = None
        
        # Define data paths
        self.training_data_path = './data/model_data_training.csv'
        self.prediction_data_path = './data/model_data_prediction.csv' 
        self.merged_data_path = './data/merged_data_prediction.csv'
        
        # Define export paths
        self.training_export_path = './data/model_data_training_newPoisson.xlsx'
        self.prediction_export_path = './data/model_data_prediction_newPoisson.xlsx'
        self.merged_export_path = './data/merged_data_prediction.csv'

    def add_poisson_merged(self, matches):
        """
        Calculate Poisson xG for both training and merged prediction data.
        
        Args:
            matches: DataFrame containing match data
            
        Returns:
            DataFrame with added Poisson xG columns
        """
        # Prepare feature sets
        X_home = matches[['home_league_position', 'away_league_position', 'Home_goal_difference_cum', 
                         'Away_goal_difference_cum', 'home_win_rate', 'away_win_rate',
                         'Home_points_cum', 'Away_points_cum']]
        X_away = matches[['away_league_position', 'home_league_position', 'Home_goal_difference_cum',
                         'Away_goal_difference_cum', 'home_win_rate', 'away_win_rate',
                         'Home_points_cum', 'Away_points_cum']]
                         
        X_merged_home = self.merged_matches[['away_league_position', 'home_league_position', 
                                           'Home_goal_difference_cum', 'Away_goal_difference_cum',
                                           'home_win_rate', 'away_win_rate', 
                                           'Home_points_cum', 'Away_points_cum']]
        X_merged_away = self.merged_matches[['away_league_position', 'home_league_position',
                                           'Home_goal_difference_cum', 'Away_goal_difference_cum',
                                           'home_win_rate', 'away_win_rate',
                                           'Home_points_cum', 'Away_points_cum']]

        # Add constants
        X_home = sm.add_constant(X_home)
        X_away = sm.add_constant(X_away)
        X_merged_home = sm.add_constant(X_merged_home)
        X_merged_away = sm.add_constant(X_merged_away)

        # Fit Poisson models
        poisson_home = sm.GLM(matches['home_goals'], X_home, family=sm.families.Poisson())
        poisson_away = sm.GLM(matches['away_goals'], X_away, family=sm.families.Poisson())
        
        poisson_home_results = poisson_home.fit()
        poisson_away_results = poisson_away.fit()

        # Print model summaries
        print("Poisson Regression for Home Goals:")
        print(poisson_home_results.summary())
        print("\nPoisson Regression for Away Goals:")
        print(poisson_away_results.summary())

        # Add predictions
        matches['home_poisson_xG'] = poisson_home_results.predict(X_home)
        matches['away_poisson_xG'] = poisson_away_results.predict(X_away)
        self.merged_matches['home_poisson_xG'] = poisson_home_results.predict(X_merged_home)
        self.merged_matches['away_poisson_xG'] = poisson_away_results.predict(X_merged_away)
        
        return matches

    def add_poisson(self, matches):
        """
        Calculate Poisson xG for training data only.
        
        Args:
            matches: DataFrame containing match data
            
        Returns:
            DataFrame with added Poisson xG columns
        """
        # Prepare feature sets
        X_home = matches[['home_league_position', 'away_league_position', 'Home_goal_difference_cum',
                         'Away_goal_difference_cum', 'home_win_rate', 'away_win_rate',
                         'Home_points_cum', 'Away_points_cum']]
        X_away = matches[['away_league_position', 'home_league_position', 'Home_goal_difference_cum',
                         'Away_goal_difference_cum', 'home_win_rate', 'away_win_rate',
                         'Home_points_cum', 'Away_points_cum']]

        # Add constants
        X_home = sm.add_constant(X_home)
        X_away = sm.add_constant(X_away)

        # Fit Poisson models
        poisson_home = sm.GLM(matches['home_goals'], X_home, family=sm.families.Poisson())
        poisson_away = sm.GLM(matches['away_goals'], X_away, family=sm.families.Poisson())
        
        poisson_home_results = poisson_home.fit()
        poisson_away_results = poisson_away.fit()

        # Print model summaries
        print("Poisson Regression for Home Goals:")
        print(poisson_home_results.summary())
        print("\nPoisson Regression for Away Goals:") 
        print(poisson_away_results.summary())

        # Add predictions
        matches['home_poisson_xG'] = poisson_home_results.predict(X_home)
        matches['away_poisson_xG'] = poisson_away_results.predict(X_away)
        
        return matches

    def process_data(self):
        """Load, process and export data with Poisson xG calculations."""
        # Load data
        training_data = pd.read_csv(self.training_data_path)
        prediction_data = pd.read_csv(self.prediction_data_path)
        self.merged_matches = pd.read_csv(self.merged_data_path)

        # Sort data
        training_data = training_data.sort_values(by='Datum')
        prediction_data = prediction_data.sort_values(by='Datum')
        self.merged_matches = self.merged_matches.sort_values(by='Date')

        # Process merged matches
        self.merged_matches = self.merged_matches.replace(',', '.', regex=True)
        self.merged_matches['Date'] = pd.to_datetime(self.merged_matches['Date'], errors='coerce')
        
        # Convert object columns to numeric
        cols_to_convert = self.merged_matches.select_dtypes(include=['object']).columns
        self.merged_matches[cols_to_convert] = self.merged_matches[cols_to_convert].apply(pd.to_numeric, errors='coerce')
        print(training_data.columns)
        # Calculate Poisson xG
        training_data = self.add_poisson(training_data)
        prediction_data = self.add_poisson_merged(prediction_data)

        # Export results
        print("\nExporting data:")
        training_data.to_excel(self.training_export_path)
        prediction_data.to_excel(self.prediction_export_path)
        self.merged_matches.to_csv(self.merged_export_path)

if __name__ == "__main__":
    calculator = PoissonXGCalculator()
    calculator.process_data()
