import pandas as pd
import os
import numpy as np
import sys
sys.path.append('./')
import logging
from pymongo import MongoClient
from util_tools.logging_config import LoggerSetup
# from util_tools.database import MongoClient

# Set up logger with log file path
logger = LoggerSetup.setup_logger(
    name='combine_stacked_output',
    log_file=f'./log/combine_stacked_output.log',
    level=logging.INFO
)


# Connect to MongoDB server
client = MongoClient('mongodb://192.168.0.77:27017/')
db = client['football_data']  # Access football_data database
aggregated_data = db['aggregated_data']  # Collection for storing fixtures data

# Set directories and files
model_dir = "./models/"
os.makedirs(model_dir, exist_ok=True)
logger.info(f"Model directory created/checked at {model_dir}")

column_order = ['running_id', 'Date', 'league', 'Home', 'Away', 'Prediction_models', 
                'match_outcome_prediction_rounded', 'home_goals_prediction_rounded',
                'away_goals_prediction_rounded', 'home_goals_prediction',
                'away_goals_prediction', 'match_outcome_prediction',
                'Odd_Home', 'Odds_Draw', 'Odd_Away'
                ] #, ,'home_poisson_xG', 'away_poisson_xG'


def add_home_away_columns(existing_df):
    try:
        # Get the list of running_ids to filter in MongoDB
        running_ids = existing_df['running_id'].tolist()
        logger.info(f"Processing {len(running_ids)} running IDs")
        
        # Query MongoDB to get the matching documents
        query = {'running_id': {'$in': running_ids}}
        projection = {'_id': 0, 'running_id': 1, 'Home': 1, 'Away': 1, 'league': 1, 'Date': 1, 'Odd_Home': 1, 'Odds_Draw': 1, 'Odd_Away': 1}

        # Corrected: Access the database and collection using methods
        # db = db_client.get_database('football_data')
        collection = aggregated_data
        mongo_data = pd.DataFrame(list(collection.find(query, projection)))
        
        if len(mongo_data) == 0:
            logger.error("MongoDB returned an empty dataframe")
            base_data_path = './data/merged_data_prediction.csv'
            mongo_df = pd.read_csv(base_data_path)
            print(mongo_df.columns)
            # Convert 'Date' column to string format 'YYYY-MM-DD' if it exists
            if 'Date' in mongo_df.columns:
                mongo_df['Date'] = mongo_df['Date'].dt.strftime('%Y-%m-%d')
            mongo_df = mongo_df[['running_id', 'Date', 'league_encoded', 'Home', 'Away']]
            # mongo_df['Date'] = pd.to_datetime(mongo_df['year'].astype(str) + '-' + mongo_df['month'].astype(str) + '-' + mongo_df['day_of_month'].astype(str))
            logger.info(f"Retrieved {len(mongo_df)} documents from Excel")
        else:
            # Convert MongoDB result to a DataFrame
            mongo_df = pd.DataFrame(mongo_data)
            logger.info(f"Retrieved {len(mongo_data)} documents from MongoDB")
            
        # Merge the MongoDB data with the existing DataFrame based on 'running_id'
        merged_df = pd.merge(existing_df, mongo_df, on='running_id', how='left')
        logger.info(f"Merged dataframe shape: {merged_df.shape}")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error in add_home_away_columns: {str(e)}")
        pass

# Function to round values based on 0.5 threshold
def round_half_up(value):
    return np.floor(value + 0.5)

try:
    # Load feature-engineered data
    stacked_home_path = './predictions_hybrid_2fit_home_goals.xlsx'
    stacked_away_path = './predictions_hybrid_2fit_away_goals.xlsx'
    stacked_outcome_path = './predictions_hybrid_2fit_match_outcome.xlsx'

    stacked_home = pd.read_excel(stacked_home_path)
    stacked_away = pd.read_excel(stacked_away_path)
    stacked_outcome = pd.read_excel(stacked_outcome_path)
    logger.info("Successfully loaded all prediction files")

    stacked_home = stacked_home[['running_id','home_goals_prediction']].sort_values(by='running_id')
    stacked_away = stacked_away[['running_id','away_goals_prediction']].sort_values(by='running_id')
    stacked_outcome = stacked_outcome[['running_id','match_outcome_prediction']].sort_values(by='running_id')

    # Merge the data
    stacked_merged_df = add_home_away_columns(stacked_home)
    if stacked_merged_df is None or stacked_merged_df.empty:
        stacked_merged_df = stacked_home
        logger.error("add_home_away_columns returned an empty dataframe or None")
        
  
    stacked_merged_df = stacked_merged_df.merge(stacked_away, how='left', on='running_id')
    stacked_merged_df = stacked_merged_df.merge(stacked_outcome, how='left', on='running_id')
    logger.info("Successfully merged all dataframes")

    # Round predictions
    stacked_merged_df['home_goals_prediction_rounded'] = stacked_merged_df['home_goals_prediction'].apply(round_half_up).astype(int)
    stacked_merged_df['away_goals_prediction_rounded'] = stacked_merged_df['away_goals_prediction'].apply(round_half_up).astype(int)
    stacked_merged_df['match_outcome_prediction_rounded'] = stacked_merged_df['match_outcome_prediction'].apply(round_half_up).astype(int)
    stacked_merged_df['Prediction_models'] = stacked_merged_df['home_goals_prediction_rounded'].astype(str) + '-' + stacked_merged_df['away_goals_prediction_rounded'].astype(str)
    logger.info("Successfully rounded all predictions")

    # Reorder columns and save
    stacked_merged_df = stacked_merged_df[column_order]
    output_path = './made_predictions/predictions_stacked_2fit_merged.xlsx'
    stacked_merged_df.to_excel(output_path, index=False)
    logger.info(f"Successfully saved merged predictions to {output_path}")



except Exception as e:
    logger.error(f"Error in main execution: {str(e)}")
    raise 
