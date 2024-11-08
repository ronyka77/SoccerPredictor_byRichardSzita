import os
import pandas as pd
import logging
from pymongo import MongoClient
import joblib
import datetime

# Set up logging
log_file_path = './SoccerPredictor_byRichardSzita/data_tools/log/merged_data.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define important features for the model
important_features = [
    'season_encoded', 'league_encoded', 'venue_encoded',
    'away_league_position', 'Away_points_cum', 'Away_goal_difference_cum', 'Away_draws', 'away_draw_rate', 'away_average_points', 
    'Away_team_matches', 'away_win_rate', 'Away_wins', 'away_xG_rolling_avg', 'away_shots_on_target_accuracy_rolling_avg', 'away_shots_on_target_rolling_avg',
    'away_saves_accuracy_rolling_avg', 'away_points_rolling_avg', 'away_goals_rolling_avg', 'away_goal_difference_rolling_avg', 'home_average_points',
    'home_draw_rate',
    'Home_draws', 
    'Home_goal_difference_cum',
    'home_league_position', 
    'Home_team_matches', 
    'home_win_rate',
    'Home_wins', 
    'HomeTeam_last_away_match', 
    'home_xG_rolling_avg',
    'home_shots_on_target_rolling_avg',
    'home_shots_on_target_accuracy_rolling_avg',
    'home_saves_accuracy_rolling_avg', 
    'home_points_rolling_avg', 
    'Home_points_cum', 
    'home_goals_rolling_avg', 
    'home_goal_difference_rolling_avg',
    'referee_foul_rate', 'referee_encoded', 
    'home_advantage',
    'Home_saves_mean',	'Away_saves_mean',	
    'Home_shot_on_target_mean',	'away_shot_on_target_mean',
    'Home_fouls_mean',	'Away_fouls_mean',	
    'Home_possession_mean',	'away_possession_mean',
    'home_team_elo',	'away_team_elo',
    'home_poisson_xG',	'away_poisson_xG'
]
# Python list of the columns
column_order = [
    'running_id', 'season_encoded', 'league_encoded', 'home_encoded', 'away_encoded','venue_encoded', 'Date',
    'home_league_position','away_league_position', 
    'Home_points_cum','Away_points_cum', 
    'Home_goal_difference_cum','Away_goal_difference_cum',
    'Home_draws','Away_draws', 
    'home_draw_rate','away_draw_rate', 
    'home_average_points', 'away_average_points', 
    'Home_team_matches','Away_team_matches',
    'home_win_rate', 'away_win_rate',
    'Home_wins', 'Away_wins', 
    'home_xG_rolling_avg','away_xG_rolling_avg', 
    'home_shots_on_target_accuracy_rolling_avg','away_shots_on_target_accuracy_rolling_avg', 
    'home_shots_on_target_rolling_avg', 'away_shots_on_target_rolling_avg', 
    'away_saves_accuracy_rolling_avg',
    'home_points_rolling_avg','away_points_rolling_avg', 
    'home_goals_rolling_avg','away_goals_rolling_avg', 
    'away_goal_difference_rolling_avg',
    'home_saves_accuracy_rolling_avg', 
    'home_goal_difference_rolling_avg', 
    'Home_saves_mean',	'Away_saves_mean',
    'Home_shot_on_target_mean',	'away_shot_on_target_mean',
    'Home_fouls_mean',	'Away_fouls_mean',
    'Home_possession_mean',	'away_possession_mean',
    'home_team_elo',	'away_team_elo',
    'home_poisson_xG',	'away_poisson_xG',
    'referee_foul_rate', 'referee_encoded'
]

# Load historical data from an Excel file
def load_historical_data(file_path):
    logging.info(f"Loading historical data from {file_path}...")
    try:
        historical_data = pd.read_excel(file_path)
        logging.info("Historical data loaded successfully!")
    except Exception as e:
        logging.error(f"Error loading historical data: {e}")
        raise
    return historical_data

# Define the cutoff date
cutoff_date = datetime.date(2024, 10, 22)
# Retrieve future match data from MongoDB and match team names with encoded values
def get_future_matches(collection, historical_data):
    logging.info("Querying future matches from MongoDB...")
    # future_matches = pd.DataFrame(list(collection.find({"match_outcome": {"$eq": None}})))
    # Query MongoDB for future matches (after 2024-10-22) where match_outcome is still None
    future_matches = pd.DataFrame(list(collection.find({
        "Date": {"$gt": '2024-10-21'}
    })))
    
    if future_matches.empty:
        logging.warning("No future matches found.")
    else:
        logging.info(f"Found {future_matches.shape[0]} future matches.")
    
    future_matches['Date'] = pd.to_datetime(future_matches['Date'], errors='coerce')
    logging.info("Extracted date features successfully.")
    
    columns_to_drop = [ 'Round','_id_stats'
                        ,'Home_xG', 'Attendance', 'Score', 'Away_xG', '_id', '_id_match', 'Day', 'Time',
                       'Match Report', 'season', 'league', 'unique_id', 'match', 'team_stats', 'team_stats_extra',
                       'url', 'match_outcome']
    
    future_matches = future_matches.drop(columns=columns_to_drop, errors='ignore')
    logging.info("Dropped unnecessary columns.")
    future_matches.to_excel('./SoccerPredictor_byRichardSzita/data_to_merge.xlsx')
    print('exported')
    # Merging team names with encoded values from historical data
    logging.info("Merging future matches with historical data for encoded team values...")
    try:
        future_matches = pd.merge(future_matches, historical_data[['Home', 'home_encoded']], on='Home', how='left')
        future_matches = pd.merge(future_matches, historical_data[['Away', 'away_encoded']], on='Away', how='left')
        logging.info("Merging completed successfully.")
    except Exception as e:
        logging.error(f"Error merging team encoded values: {e}")
        raise
    
    return future_matches

# Perform feature engineering on future matches
def feature_engineering(future_matches, historical_data):
    logging.info("Starting feature engineering for future matches...")
    
    for feature in important_features:
        if feature not in historical_data.columns:
            logging.warning(f"Feature '{feature}' not found in historical data.")
            continue
        
        encoded_col = 'home_encoded' if 'home' in feature or 'Home' in feature else 'away_encoded'
        
        try:
            historical_data_sorted = historical_data.sort_values(by=[encoded_col, 'running_id'])
            historical_data_grouped = historical_data_sorted.groupby(encoded_col).last().reset_index()
            
            future_matches = future_matches.merge(historical_data_grouped[[encoded_col, feature]], 
                                                  on=encoded_col, how='left', suffixes=('', f'_{feature}'))
        except Exception as e:
            logging.error(f"Error processing feature '{feature}': {e}")
            raise
    
    logging.info("Feature engineering completed.")
    return future_matches

# Prepare data for predictions by selecting only important features
def prepare_data_for_model(future_matches, important_features):
    missing_features = [feat for feat in important_features if feat not in future_matches.columns]
    if missing_features:
        logging.warning(f"Missing features in future matches data: {missing_features}")
    
    try:
        model_data = future_matches[important_features]
        logging.info("Data prepared for model prediction.")
    except Exception as e:
        logging.error(f"Error preparing data for model: {e}")
        raise
    
    return model_data

# Main function to run the entire prediction pipeline
def main():
    logging.info("Connecting to MongoDB...")
    client = MongoClient("mongodb://192.168.0.77:27017/")
    db = client['football_data']
    collection = db['aggregated_data']
    
    historical_data = load_historical_data("./SoccerPredictor_byRichardSzita/data/model_data_prediction_newPoisson.xlsx")
    
    future_matches = get_future_matches(collection, historical_data)
    print(future_matches)
    # future_matches = feature_engineering(future_matches, historical_data)
    
    # future_matches = future_matches.drop_duplicates()
    # # Ordering a dataframe by these columns
    # future_matches = future_matches[column_order].sort_values(by=column_order)
    # future_matches = future_matches.sort_values(by='Date')
    # future_matches = future_matches.dropna()
    
    # future_matches.to_excel('./SoccerPredictor_byRichardSzita/data/merged_data_prediction.xlsx', index=False)
    # logging.info("Exported future matches to merged_data_prediction.xlsx.")

if __name__ == "__main__":
    main()
