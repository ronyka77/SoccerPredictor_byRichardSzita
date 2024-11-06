import pandas as pd
import os
import logging
from pymongo import MongoClient
import numpy as np

# MongoDB setup
client = MongoClient('192.168.0.77', 27017)
db = client.football_data

# Collections
collection = db.aggregated_data

model_type = 'stacked_new'
# Set up logging
log_file_path = './SoccerPredictor_byRichardSzita/score_prediction/test/log/stacked_' + model_type + '_model_training.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Set directories and files
model_dir = "./models/"
os.makedirs(model_dir, exist_ok=True)

logging.info(f"Starting model training for {model_type}")
column_order = ['running_id', 'Date', 'league', 'Home', 'Away', 'score_prediction', 'match_outcome_prediction_rounded', 'home_goals_prediction_rounded','away_goals_prediction_rounded'
                 ,'home_goals_prediction','away_goals_prediction', 'match_outcome_prediction','home_poisson_xG','away_poisson_xG']
def add_home_away_columns(existing_df):
   
    # Get the list of running_ids to filter in MongoDB
    running_ids = existing_df['running_id'].tolist()
    
    # Query MongoDB to get the matching documents
    query = {'running_id': {'$in': running_ids}}
    projection = {'_id': 0, 'running_id': 1, 'Home': 1, 'Away': 1, 'league': 1 , 'Date': 1}  # Only return these fields
    mongo_data = list(collection.find(query, projection))
    
    # Convert MongoDB result to a DataFrame
    mongo_df = pd.DataFrame(mongo_data)
    
    # Merge the MongoDB data with the existing DataFrame based on 'running_id'
    merged_df = pd.merge(existing_df, mongo_df, on='running_id', how='left')
    
    # Close the MongoDB connection
    client.close()
    
    # Return the updated DataFrame with Home, Away, and Date columns
    return merged_df

# Function to round values based on 0.5 threshold
def round_half_up(value):
    return np.floor(value + 0.5)

# Load your feature-engineered data
stacked_home_path = './SoccerPredictor_byRichardSzita/predictions_hybrid_2fit_home_goals.xlsx'
stacked_away_path = './SoccerPredictor_byRichardSzita/predictions_hybrid_2fit_away_goals.xlsx'
stacked_outcome_path = './SoccerPredictor_byRichardSzita/predictions_hybrid_2fit_match_outcome.xlsx'

stacked_home = pd.read_excel(stacked_home_path)
stacked_away = pd.read_excel(stacked_away_path)
stacked_outcome = pd.read_excel(stacked_outcome_path)

stacked_home = stacked_home[['running_id','home_goals_prediction','home_poisson_xG','away_poisson_xG']].sort_values(by='running_id')
stacked_away = stacked_away[['running_id','away_goals_prediction']].sort_values(by='running_id')
stacked_outcome = stacked_outcome[['running_id','match_outcome_prediction']].sort_values(by='running_id')

# Call the function to merge the new data
stacked_merged_df = add_home_away_columns(stacked_home)
stacked_merged_df = stacked_merged_df.merge(stacked_away,how='left',on='running_id')
stacked_merged_df = stacked_merged_df.merge(stacked_outcome,how='left',on='running_id')

# Apply the function and add new rounded columns
stacked_merged_df['home_goals_prediction_rounded'] = stacked_merged_df['home_goals_prediction'].apply(round_half_up).astype(int)
# stacked_merged_df['home_goals_prediction_with_error_rounded'] = stacked_merged_df['home_goals_prediction_with_error'].apply(round_half_up).astype(int)
stacked_merged_df['away_goals_prediction_rounded'] = stacked_merged_df['away_goals_prediction'].apply(round_half_up).astype(int)
# stacked_merged_df['away_goals_prediction_with_error_rounded'] = stacked_merged_df['away_goals_prediction_with_error'].apply(round_half_up).astype(int)
stacked_merged_df['match_outcome_prediction_rounded'] = stacked_merged_df['match_outcome_prediction'].apply(round_half_up).astype(int)
# stacked_merged_df['match_outcome_prediction_with_error_rounded'] = stacked_merged_df['match_outcome_prediction_with_error'].apply(round_half_up).astype(int)
stacked_merged_df['score_prediction'] = stacked_merged_df['home_goals_prediction_rounded'].astype(str) + '-' + stacked_merged_df['away_goals_prediction_rounded'].astype(str)
# stacked_merged_df['score_prediction_error'] = stacked_merged_df['home_goals_prediction_with_error_rounded'].astype(str) + '-' + stacked_merged_df['away_goals_prediction_with_error_rounded'].astype(str)
print(list(stacked_merged_df.columns))
stacked_merged_df = stacked_merged_df[column_order]
# Print the updated DataFrame
stacked_merged_df.to_excel('./SoccerPredictor_byRichardSzita/predictions_stacked_2fit_merged.xlsx')