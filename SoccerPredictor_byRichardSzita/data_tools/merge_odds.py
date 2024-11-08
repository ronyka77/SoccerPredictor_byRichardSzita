import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from rapidfuzz import fuzz, process  # Use process for matching
import logging
import os
from typing import Tuple, Optional
import configparser
from util_tools.database import MongoClient
from util_tools.logging_config import LoggerSetup

logger = LoggerSetup.setup_logger(
    name='merge_odds',
    log_file='./data_tools/log/merge_odds.log',
    level=logging.INFO
)

def load_config() -> dict:
    config = configparser.ConfigParser()
    config_loaded = config.read(CONFIG_PATH)
    if not config_loaded:
        raise FileNotFoundError(f"Could not find config file at {CONFIG_PATH}")
    print(f"Config file loaded from: {CONFIG_PATH}")
    return {
        'mongo_uri': config['MongoDB']['uri'],
        'db_name': config['MongoDB']['database'],
        'threshold': config['Matching']['threshold']
    }

# Initialize MongoDB client
db_client = MongoClient()

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data from match_stats and odds_data collections in MongoDB.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Match stats and odds data frames
    
    Raises:
        ConnectionError: If MongoDB connection fails
    """
    try:
        with db_client.get_database() as db:
            match_stats_df = pd.DataFrame(list(db.fixtures.find()))
            odds_data_df = pd.DataFrame(list(db.odds_data.find()))
            return match_stats_df, odds_data_df
    except Exception as e:
        logging.error(f"Failed to load data: {str(e)}")
        raise ConnectionError("Failed to connect to MongoDB")

def fuzzy_merge(match_stats_df: pd.DataFrame, odds_data_df: pd.DataFrame, threshold: int = 90) -> Optional[pd.DataFrame]:
    """Merge match_stats and odds_data DataFrames based on fuzzy matching of unique_id."""
    # Validate input
    if match_stats_df is None or odds_data_df is None:
        logging.error("One or both DataFrames are None")
        return None
        
    if match_stats_df.empty or odds_data_df.empty:
        logging.error(f"Empty DataFrame provided - match_stats: {len(match_stats_df)}, odds_data: {len(odds_data_df)}")
        return None

    # Debug input data
    logging.info(f"Sample match_stats unique_ids: {match_stats_df['unique_id'].head().tolist()}")
    logging.info(f"Sample odds_data unique_ids: {odds_data_df['unique_id'].head().tolist()}")

    # Convert unique_ids to list and ensure they're strings
    odds_unique_ids = [str(uid) for uid in odds_data_df['unique_id'].tolist()]
    
    merged_data = []
    
    for idx, row in match_stats_df.iterrows():
        try:
            match_id = str(row['unique_id'])
            
            # Debug matching process
            logging.debug(f"Processing match_id: {match_id}")
            
            result = process.extractOne(
                match_id, 
                odds_unique_ids, 
                scorer=fuzz.ratio,
                score_cutoff=threshold
            )
            
            if result is None:
                logging.debug(f"No match found for ID: {match_id}")
                merged_data.append(row.to_dict())
                continue
                
            best_match, score, index = result
            
            if score >= threshold:
                odds_row = odds_data_df[odds_data_df['unique_id'] == best_match].iloc[0]
                merged_data.append({**row.to_dict(), **odds_row.to_dict()})
                logging.debug(f"Merged: {match_id} with {best_match} (Score: {score})")
            else:
                merged_data.append(row.to_dict())
                
        except Exception as e:
            logging.error(f"Error processing row {idx}: {str(e)}")
            continue
    
    if not merged_data:
        logging.error("No data was merged successfully")
        return None
        
    return pd.DataFrame(merged_data)

# Function to store aggregated data back in MongoDB
def store_aggregated_data(aggregated_data: pd.DataFrame, batch_size: int = 1000) -> None:
    """Store aggregated data in MongoDB using batch processing.
    
    Args:
        aggregated_data (pd.DataFrame): Data to store
        batch_size (int, optional): Size of each batch. Defaults to 1000.
    """
    try:
        with db_client.get_database() as db:
            aggregated_collection = db.fixtures
            records = aggregated_data.to_dict('records')
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                operations = [
                    {
                        'update_one': {
                            'filter': {'unique_id': record['unique_id']},
                            'update': {'$set': record},
                            'upsert': True
                        }
                    }
                    for record in batch
                ]
                try:
                    aggregated_collection.bulk_write(operations, ordered=False)
                    logging.info(f"Processed batch {i//batch_size + 1}")
                except Exception as e:
                    logging.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
    except Exception as e:
        logging.error(f"Error connecting to database: {str(e)}")

def main():
    try:
        # Add collection verification
        with db_client.get_database() as db:
            print("Available collections:", db.list_collection_names())
            print("Fixtures count:", db.fixtures.count_documents({}))
            print("Odds data count:", db.odds_data.count_documents({}))

        # Load data
        match_stats_df, odds_data_df = load_data()
        
        # Log data shapes for debugging
        logging.info(f"Loaded match_stats shape: {match_stats_df.shape}")
        logging.info(f"Loaded odds_data shape: {odds_data_df.shape}")
        
        # Perform fuzzy matching and merge
        merged_df = fuzzy_merge(match_stats_df, odds_data_df, threshold=90)
        
        if merged_df is None:
            logging.error("Merge operation failed - merged_df is None")
            return
            
        if merged_df.empty:
            logging.error("Merge operation resulted in empty DataFrame")
            return
            
        logging.info(f"Merged DataFrame shape: {merged_df.shape}")
        
        # Safe column access
        if 'Odd_Home' not in merged_df.columns:
            logging.error(f"Column 'Odd_Home' not found. Available columns: {merged_df.columns.tolist()}")
            return
            
        filtered_df = merged_df[merged_df['Odd_Home'].notna()]
        
        # Safe column dropping
        columns_to_drop = ['League', '_id']
        existing_columns = [col for col in columns_to_drop if col in filtered_df.columns]
        if existing_columns:
            filtered_df.drop(columns=existing_columns, inplace=True)
        
        if filtered_df is not None and not filtered_df.empty:
            logging.info(f"Final filtered DataFrame shape: {filtered_df.shape}")
            print("Columns in filtered DataFrame:", filtered_df.columns.tolist())
            store_aggregated_data(filtered_df)
            logging.info("Data merged and stored successfully.")
            print(filtered_df.head())
        else:
            logging.error("No data after filtering")
            
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
