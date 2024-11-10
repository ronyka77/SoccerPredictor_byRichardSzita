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
from pymongo import UpdateOne

logger = LoggerSetup.setup_logger(
    name='merge_odds',
    log_file='./data_tools/log/merge_odds.log',
    level=logging.INFO
)

# Initialize MongoDB client
db_client = MongoClient()

# Function to drop Odd_Home, Odd_Away, and Odd_Draw columns from the aggregated_data collection in MongoDB
def drop_odds_columns():
    """Drop Odd_Home, Odd_Away, and Odd_Draw columns from the aggregated_data collection in MongoDB."""
    try:
        with db_client.get_database() as db:
            result = db.aggregated_data.update_many(
                {},
                {"$unset": {"Odd_Home": "", "Odd_Away": "", "Odd_Draw": ""}}
            )
            logger.info(f"Successfully dropped columns from {result.modified_count} documents.")
    except Exception as e:
        logger.error(f"Failed to drop columns: {str(e)}")
        raise

drop_odds_columns()

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data from match_stats and odds_data collections in MongoDB.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Match stats and odds data frames
    
    Raises:
        ConnectionError: If MongoDB connection fails
    """
    try:
        with db_client.get_database() as db:
            match_stats_df = pd.DataFrame(list(db.fixtures.find({"Odd_Home": {"$exists": False}})))
            odds_data_df = pd.DataFrame(list(db.odds_data.find()))
            return match_stats_df, odds_data_df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise ConnectionError("Failed to connect to MongoDB")

def fuzzy_merge(match_stats_df: pd.DataFrame, odds_data_df: pd.DataFrame, threshold: int = 90) -> Optional[pd.DataFrame]:
    """Merge match_stats and odds_data DataFrames based on fuzzy matching of unique_id and matching date."""
    # Validate input
    if match_stats_df is None or odds_data_df is None:
        logger.error("One or both DataFrames are None")
        return None
        
    if match_stats_df.empty or odds_data_df.empty:
        logger.error(f"Empty DataFrame provided - match_stats: {len(match_stats_df)}, odds_data: {len(odds_data_df)}")
        return None

    # Debug input data
    logger.info(f"Sample match_stats unique_ids: {match_stats_df['unique_id'].head().tolist()}")
    logger.info(f"Sample odds_data unique_ids: {odds_data_df['unique_id'].head().tolist()}")

    # Convert unique_ids to list and ensure they're strings
    odds_unique_ids = [str(uid) for uid in odds_data_df['unique_id'].tolist()]
    
    # Standardize team names by removing common abbreviations and special characters
    def standardize_name(name: str) -> str:
        replacements = {
            "ã": "a",
            "Nott'ham Forest": "Nottingham", 
            "B. Monchengladbach": "Gladbach",
            "Paris S-G": "PSG",
            "Leeds United": "Leeds",
            "Athletic Club": "Athletic Bilbao",
            "Athletico Paranaense": "Athletico-PR",
            "FC Barcelona": "Barcelona",
            "FC Bayern": "Bayern Munich",
            "FC Internazionale": "Inter Milan",
            "FC Porto": "Porto"
        }
        name = str(name).strip()
        for old, new in replacements.items():
            name = name.replace(old, new)
        return name.strip()
    
    merged_data = []
    unmatched_pairs = []  # List to store unmatched ID pairs
    
    for idx, row in match_stats_df.iterrows():
        try:
            match_id = str(row['unique_id'])
            match_date = row['Date']
            # Standardize the match ID before comparison
            standardized_match_id = standardize_name(match_id)

            # Standardize all odds IDs for comparison
            standardized_odds_ids = [standardize_name(uid) for uid in odds_unique_ids]
            
            result = process.extractOne(
                standardized_match_id, 
                standardized_odds_ids, 
                scorer=fuzz.ratio,
                score_cutoff=threshold
            )
            
            if result is None:
                merged_data.append(row.to_dict())
                continue
                
            best_match, score, index = result
            
            # Use the original odds ID for the database lookup
            original_odds_id = odds_unique_ids[standardized_odds_ids.index(best_match)]
            
            if score >= threshold:
                odds_row = odds_data_df[odds_data_df['unique_id'] == original_odds_id].iloc[0]
                if odds_row['Date'] == match_date or abs((pd.to_datetime(odds_row['Date']) - pd.to_datetime(match_date)).days) == 1:
                    merged_data.append({**row.to_dict(), **odds_row.to_dict()})
                    logger.info(f"Merged: {match_id} with {original_odds_id} (Score: {score})")
                else:
                    logger.info(f"Date mismatch for ID: {match_id} and {original_odds_id}")
                    merged_data.append(row.to_dict())
            else:
                merged_data.append(row.to_dict())
                # Store unmatched pair with score
                unmatched_pairs.append({
                    'match_stats_id': match_id,
                    'closest_odds_id': original_odds_id,
                    'similarity_score': score
                })
                
        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            continue
    
    if not merged_data:
        logger.error("No data was merged successfully")
        return None
    
    # Export unmatched pairs to Excel if any exist
    if unmatched_pairs:
        unmatched_df = pd.DataFrame(unmatched_pairs)
        export_path = 'unmatched_ids.xlsx'
        unmatched_df.to_excel(export_path, index=False)
        logger.info(f"Exported {len(unmatched_pairs)} unmatched ID pairs to {export_path}")
    
    merged_df = pd.DataFrame(merged_data)
    merged_df.to_excel('merge_odds_data.xlsx', index=False)
    return merged_df

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
                # Fixed: Correct format for bulk write operations
                operations = [
                    UpdateOne(
                        {'unique_id': record['unique_id']},
                        {'$set': record},
                        upsert=True
                    )
                    for record in batch
                ]
                try:
                    aggregated_collection.bulk_write(operations, ordered=False)
                    logger.info(f"Processed batch {i//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")

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
        logger.info(f"Loaded match_stats shape: {match_stats_df.shape}")
        logger.info(f"Loaded odds_data shape: {odds_data_df.shape}")
        
        # Perform fuzzy matching and merge
        merged_df = fuzzy_merge(match_stats_df, odds_data_df, threshold=90)
        
        if merged_df is None:
            logger.error("Merge operation failed - merged_df is None")
            return
            
        if merged_df.empty:
            logger.error("Merge operation resulted in empty DataFrame")
            return
            
        logger.info(f"Merged DataFrame shape: {merged_df.shape}")
        
        # Safe column access
        if 'Odd_Home' not in merged_df.columns:
            logger.error(f"Column 'Odd_Home' not found. Available columns: {merged_df.columns.tolist()}")
            return
            
        filtered_df = merged_df[merged_df['Odd_Home'].notna()]
        # Export rows where Odd_Home is NA to Excel for analysis
        na_odds_df = merged_df[merged_df['Odd_Home'].isna()]
        if not na_odds_df.empty:
            na_odds_df.to_excel('missing_odds_data.xlsx', index=False)
            logger.info(f"Exported {len(na_odds_df)} rows with missing odds data to missing_odds_data.xlsx")
        # Safe column dropping
        columns_to_drop = ['League', '_id']
        existing_columns = [col for col in columns_to_drop if col in filtered_df.columns]
        if existing_columns:
            filtered_df.drop(columns=existing_columns, inplace=True)
        
        if filtered_df is not None and not filtered_df.empty:
            logger.info(f"Final filtered DataFrame shape: {filtered_df.shape}")
            # print("Columns in filtered DataFrame:", filtered_df.columns.tolist())
            store_aggregated_data(filtered_df)
            logger.info("Data merged and stored successfully.")
            # print(filtered_df.head())
        else:
            logger.error("No data after filtering")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
