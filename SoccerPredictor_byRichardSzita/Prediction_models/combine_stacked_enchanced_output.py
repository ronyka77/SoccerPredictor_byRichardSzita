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
    log_file=f'./log/combine_stacked_output_enhanced.log',
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

column_order = ['running_id', 'Date', 'league', 'Home', 'Away','Score', 'real_outcome', 'Prediction_models','Outcome_models', 
                'Prediction_models_enhanced_positive', 'Outcome_models_enhanced_positive',
                'Prediction_models_enhanced_negative', 'Outcome_models_enhanced_negative',
                'Odd_Home', 'Odds_Draw', 'Odd_Away',
                'match_outcome_prediction_base', 'home_goals_prediction_base',
                'home_goals_confidence','home_goals_uncertainty',
                'away_goals_confidence','away_goals_uncertainty',
                'match_outcome_confidence','match_outcome_uncertainty'
                ] #, ,'home_poisson_xG', 'away_poisson_xG'


def add_base_columns(existing_df):
    try:
        # Get the list of running_ids to filter in MongoDB
        unique_ids = existing_df['running_id'].tolist()
        logger.info(f"Processing {len(unique_ids)} unique IDs")
        
        # Query MongoDB to get the matching documents
        query = {'running_id': {'$in': unique_ids}}
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
            
        # Merge the MongoDB data with the existing DataFrame based on 'unique_id'
        merged_df = pd.merge(existing_df, mongo_df, on='running_id', how='left')
        logger.info(f"Merged dataframe shape: {merged_df.shape}")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error in add_base_columns: {str(e)}")
        pass

def add_score(existing_df):
    """
    Add match scores from MongoDB fixtures collection to the existing DataFrame.
    
    Args:
        existing_df (pd.DataFrame): DataFrame containing unique_ids and predictions
        
    Returns:
        pd.DataFrame: DataFrame with added Score column from MongoDB
    """
    try:
        # Get list of matches to query
        logger.info(f"Querying scores for {len(existing_df)} matches")
        
        # Query MongoDB fixtures collection
        query = {
            '$and': [
                {'Home': {'$in': existing_df['Home'].tolist()}},
                {'Away': {'$in': existing_df['Away'].tolist()}},
                {'Date': {'$in': existing_df['Date'].tolist()}}
            ]
        }
        projection = {'_id': 0, 'Home': 1, 'Away': 1, 'Date': 1, 'Score': 1}
        
        # Get data from fixtures collection
        collection = db['fixtures']
        mongo_data = list(collection.find(query, projection))
        
        if len(mongo_data) == 0:
            logger.warning("No scores found in MongoDB fixtures collection")
            return existing_df
            
        # Convert to DataFrame
        scores_df = pd.DataFrame(mongo_data)
        logger.info(f"Retrieved {len(scores_df)} scores from MongoDB")
        
        # Merge scores with existing DataFrame on Home, Away and Date
        merged_df = pd.merge(existing_df, scores_df, 
                           on=['Home', 'Away', 'Date'], 
                           how='left')
        logger.info(f"Merged dataframe shape: {merged_df.shape}")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error in add_score: {str(e)}")
        return existing_df

def calculate_outcome(home_goals, away_goals):
    """
    Calculate match outcome based on score string.
    
    Args:
        score (str): Score in format 'X-Y' where X is home goals and Y is away goals
        
    Returns:
        int: 1 for home win, 2 for draw, 3 for away win
    """
    try:
        if home_goals > away_goals:
            return 1  # Home win
        elif home_goals == away_goals:
            return 2  # Draw
        elif home_goals < away_goals:
            return 3  # Away win
        else:
            return None
    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing score '{home_goals}-{away_goals}': {str(e)}")
        return None

# Function to round values based on 0.5 threshold
def round_half_up(value):
    return np.floor(value + 0.5)

try:
    # Load feature-engineered data
    stacked_home_path = './predictions_hybrid_2fit_home_goals_enhanced.xlsx'
    stacked_away_path = './predictions_hybrid_2fit_away_goals_enhanced.xlsx'
    stacked_outcome_path = './predictions_hybrid_2fit_match_outcome_enhanced.xlsx'

    stacked_home = pd.read_excel(stacked_home_path)
    stacked_away = pd.read_excel(stacked_away_path)
    stacked_outcome = pd.read_excel(stacked_outcome_path)
    logger.info("Successfully loaded all prediction files")

    stacked_home = stacked_home[['running_id','home_goals_prediction_base','home_goals_prediction_enhanced','home_goals_prediction_enhanced_negative','home_goals_confidence','home_goals_uncertainty']].sort_values(by='running_id')
    stacked_away = stacked_away[['running_id','away_goals_prediction_base','away_goals_prediction_enhanced','away_goals_prediction_enhanced_negative','away_goals_confidence','away_goals_uncertainty']].sort_values(by='running_id')
    stacked_outcome = stacked_outcome[['running_id','match_outcome_prediction_base','match_outcome_prediction_enhanced','match_outcome_prediction_enhanced_negative','match_outcome_confidence','match_outcome_uncertainty']].sort_values(by='running_id')

    # Merge the data
    stacked_merged_df = add_base_columns(stacked_home)
    if stacked_merged_df is None or stacked_merged_df.empty:
        stacked_merged_df = stacked_home
        logger.error("add_home_away_columns returned an empty dataframe or None")
        
  
    stacked_merged_df = stacked_merged_df.merge(stacked_away, how='left', on='running_id')
    stacked_merged_df = stacked_merged_df.merge(stacked_outcome, how='left', on='running_id')
    logger.info("Successfully merged all dataframes")

    # Round predictions
    stacked_merged_df['home_goals_prediction_base'] = stacked_merged_df['home_goals_prediction_base'].apply(round_half_up).astype(int)
    stacked_merged_df['away_goals_prediction_base'] = stacked_merged_df['away_goals_prediction_base'].apply(round_half_up).astype(int)
    stacked_merged_df['match_outcome_prediction_base'] = stacked_merged_df['match_outcome_prediction_base'].apply(round_half_up).astype(int)
    stacked_merged_df['home_goals_prediction_enhanced'] = stacked_merged_df['home_goals_prediction_enhanced'].apply(round_half_up).astype(int)
    stacked_merged_df['away_goals_prediction_enhanced'] = stacked_merged_df['away_goals_prediction_enhanced'].apply(round_half_up).astype(int)
    stacked_merged_df['match_outcome_prediction_enhanced'] = stacked_merged_df['match_outcome_prediction_enhanced'].apply(round_half_up).astype(int)
    stacked_merged_df['home_goals_prediction_enhanced_negative'] = stacked_merged_df['home_goals_prediction_enhanced_negative'].apply(round_half_up).astype(int)
    stacked_merged_df['away_goals_prediction_enhanced_negative'] = stacked_merged_df['away_goals_prediction_enhanced_negative'].apply(round_half_up).astype(int)
    stacked_merged_df['match_outcome_prediction_enhanced_negative'] = stacked_merged_df['match_outcome_prediction_enhanced_negative'].apply(round_half_up).astype(int)
    stacked_merged_df['Prediction_models'] = stacked_merged_df['home_goals_prediction_base'].astype(str) + '-' + stacked_merged_df['away_goals_prediction_base'].astype(str)
    stacked_merged_df['Prediction_models_enhanced_positive'] = stacked_merged_df['home_goals_prediction_enhanced'].astype(str) + '-' + stacked_merged_df['away_goals_prediction_enhanced'].astype(str)
    stacked_merged_df['Prediction_models_enhanced_negative'] = stacked_merged_df['home_goals_prediction_enhanced_negative'].astype(str) + '-' + stacked_merged_df['away_goals_prediction_enhanced_negative'].astype(str)
    # stacked_merged_df['Prediction_models_confidence'] = stacked_merged_df['home_goals_prediction_confidence'].astype(str) + '-' + stacked_merged_df['away_goals_prediction_confidence'].astype(str)
    # stacked_merged_df['Prediction_models_uncertainty'] = stacked_merged_df['home_goals_prediction_uncertainty'].astype(str) + '-' + stacked_merged_df['away_goals_prediction_uncertainty'].astype(str)
    logger.info("Successfully rounded all predictions")

    # Reorder columns and save
    stacked_merged_df['unique_id'] = stacked_merged_df['Date'].astype(str) + '_' + stacked_merged_df['Home'] + '_' + stacked_merged_df['Away']
    stacked_merged_df = add_score(stacked_merged_df)
    # Add outcome columns based on goal predictions using existing calculate_outcome function
    stacked_merged_df['Outcome_models'] = stacked_merged_df.apply(
        lambda row: calculate_outcome(row['home_goals_prediction_base'], row['away_goals_prediction_base']), axis=1
    )
    stacked_merged_df['Outcome_models_enhanced_positive'] = stacked_merged_df.apply(
        lambda row: calculate_outcome(row['home_goals_prediction_enhanced'], row['away_goals_prediction_enhanced']), axis=1
    )
    stacked_merged_df['Outcome_models_enhanced_negative'] = stacked_merged_df.apply(
        lambda row: calculate_outcome(row['home_goals_prediction_enhanced_negative'], row['away_goals_prediction_enhanced_negative']), axis=1
    )
    # Extract home and away goals from Score column and calculate real outcome
    stacked_merged_df['Score'] = stacked_merged_df['Score'].str.replace('–', '-')  # Replace en dash with regular hyphen
    # First split the scores
    score_split = stacked_merged_df['Score'].str.split('-', expand=True)
    # Replace empty strings with NaN
    score_split = score_split.replace('', np.nan)
    # Convert to float, NaN values will be preserved
    stacked_merged_df[['home_goals', 'away_goals']] = score_split.astype(float)
    stacked_merged_df['real_outcome'] = stacked_merged_df.apply(
        lambda row: calculate_outcome(row['home_goals'], row['away_goals']), axis=1
    )
 
    logger.info("Successfully added outcome columns based on goal predictions")
    stacked_merged_df = stacked_merged_df[column_order]
    output_path = './made_predictions/predictions_stacked_2fit_merged_enhanced.xlsx'
    stacked_merged_df.to_excel(output_path, index=False)
    logger.info(f"Successfully saved merged predictions to {output_path}")



except Exception as e:
    logger.error(f"Error in main execution: {str(e)}")
    raise 
