import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from rapidfuzz import fuzz, process  # Use process for matching
import logging
from typing import Tuple, Optional
import configparser
from util_tools.database import MongoClient
from util_tools.logging_config import LoggerSetup
from pymongo import UpdateOne
from util_tools.delete_duplicates import DuplicateHandler

logger = LoggerSetup.setup_logger(
    name='merge_odds',
    log_file='./data_tools/log/merge_odds.log',
    level=logging.INFO
)

# Initialize MongoDB client
db_client = MongoClient()

# Default collection names
MATCH_STATS_COLLECTION = 'fixtures'

# Drop 'League' and '_id' columns if they exist in the merged_row
columns_to_drop = ['League', '_id']

# Function to drop Odd_Home, Odd_Away, and Odd_Draw columns from the aggregated_data collection in MongoDB
def drop_odds_columns(collection_name: str = MATCH_STATS_COLLECTION) -> None:
    """Drop Odd_Home, Odd_Away, and Odd_Draw columns from the specified collection in MongoDB."""
    try:
        with db_client.get_database() as db:
            result = db[collection_name].update_many(
                {},
                {"$unset": {"Odd_Home": "", "Odd_Away": "", "Odds_Draw": ""}}
            )
            logger.info(f"Successfully dropped columns from {result.modified_count} documents.")
    except Exception as e:
        logger.error(f"Failed to drop columns: {str(e)}")
        raise

# drop_odds_columns() #Only use if you need to drop the odds columns from the collection

def load_data(match_stats_collection: str = MATCH_STATS_COLLECTION) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data from match_stats and odds_data collections in MongoDB."""
    try:
        with db_client.get_database() as db:
            odds_data_df = pd.DataFrame(list(db.odds_data.find()))
            match_stats_df = pd.DataFrame(list(db[match_stats_collection].find(
                {"Odds_Draw": {"$exists": 0}}
            )))
            
            # Drop rows where Odds_Draw is a number
            # match_stats_df = match_stats_df[
            #     ~match_stats_df['Odds_Draw'].apply(lambda x: isinstance(x, (int, float)) if 'Odds_Draw' in match_stats_df.columns and x is not None else False)
            # ]
            print(f"Rows after dropping numeric Odds_Draw: {len(match_stats_df)}")
            match_stats_df['Date'] = pd.to_datetime(match_stats_df['Date'])
            odds_data_df['Date'] = pd.to_datetime(odds_data_df['Date'])
            match_stats_df = match_stats_df[match_stats_df['Date'] < odds_data_df['Date'].max()]
            odds_data_df.to_excel('odds_data.xlsx', index=False)
            return match_stats_df, odds_data_df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise ConnectionError("Failed to connect to MongoDB")

# Import standardize_name from DuplicateHandler
standardize_name = DuplicateHandler.standardize_name

def fuzzy_merge_row(row: pd.Series, odds_data_df: pd.DataFrame, threshold: int = 90) -> Tuple[dict, Optional[dict]]:
    """Merge a single row from match_stats with odds_data based on fuzzy matching of unique_id and matching date."""
    try:
        match_id = str(row['unique_id'])
        match_date = row['Date']
        # Standardize the match ID before comparison
        standardized_match_id = match_id

        # Standardize all odds IDs for comparison
        odds_unique_ids = [str(uid) for uid in odds_data_df['unique_id'].tolist()]
        # standardized_odds_ids = [standardize_name(uid) for uid in odds_unique_ids]
        standardized_odds_ids = odds_unique_ids
        
        result = process.extractOne(
            standardized_match_id, 
            standardized_odds_ids, 
            scorer=fuzz.ratio,
            score_cutoff=threshold
        )
        
        if result is None:
            return row.to_dict(), None
            
        best_match, score, index = result
        
        # Use the original odds ID for the database lookup
        original_odds_id = odds_unique_ids[standardized_odds_ids.index(best_match)]
        
        if score >= threshold:
            odds_row = odds_data_df[odds_data_df['unique_id'] == original_odds_id].iloc[0]
            if odds_row['Date'] == match_date or abs((pd.to_datetime(odds_row['Date']) - pd.to_datetime(match_date)).days) <= 1:
                return {**row.to_dict(), **odds_row.to_dict()}, None
            else:
                logger.info(f"Date mismatch for ID: {match_id} and {original_odds_id}")
                return row.to_dict(), None
        else:
            return row.to_dict(), {
                'match_stats_id': match_id,
                'closest_odds_id': original_odds_id,
                'similarity_score': score
            }
    except Exception as e:
        logger.error(f"Error processing row: {str(e)}")
        return row.to_dict(), None

def store_aggregated_data_row_by_row(df: pd.DataFrame, collection_name: str = MATCH_STATS_COLLECTION) -> None:
    """Store each row of the DataFrame into the specified collection in MongoDB."""
    if df.empty:
        logger.error("The DataFrame to store is empty.")
        raise ValueError("The DataFrame to store is empty.")
    
    try:
        with db_client.get_database() as db:
            collection = db[collection_name]
            operations = []
            for _, row in df.iterrows():
                # Convert the row to a dictionary and prepare the update operation
                row_dict = row.to_dict()
                unique_id = row_dict.get('unique_id')
                if unique_id is not None:
                    # Only update odds-related fields
                    odds_update = {
                        k: v for k, v in row_dict.items() 
                        if k in ['Odd_Home', 'Odd_Away', 'Odds_Draw']
                    }
                    if odds_update:
                        operations.append(
                            UpdateOne(
                                {'unique_id': unique_id},
                                {'$set': odds_update},
                                upsert=True
                            )
                        )
            
            # if operations:
            #     result = collection.bulk_write(operations)
            #     logger.info(f"Bulk write result: {result.bulk_api_result}")
    
    except Exception as e:
        logger.error(f"Failed to store data row by row: {str(e)}")
        raise
    
def main(match_stats_collection: str = MATCH_STATS_COLLECTION) -> None:
    try:
        # Add collection verification
        with db_client.get_database() as db:
            logger.info(f"Available collections: {db.list_collection_names()}")
            logger.info(f"Fixtures count: {db[match_stats_collection].count_documents({})}")
            logger.info(f"Odds data count: {db.odds_data.count_documents({})}")

        # Load data
        match_stats_df, odds_data_df = load_data(match_stats_collection)
        
        # Log data shapes for debugging
        logger.info(f"Loaded match_stats shape: {match_stats_df.shape}")
        logger.info(f"Loaded odds_data shape: {odds_data_df.shape}")
        
        merged_data = []
        unmatched_pairs = []  # List to store unmatched ID pairs
        
        # Perform fuzzy matching and merge row by row
        for idx, row in match_stats_df.iterrows():
            merged_row, unmatched_pair = fuzzy_merge_row(row, odds_data_df, threshold=90)
            merged_data.append(merged_row)
            if merged_row and not pd.isna(merged_row.get('Odd_Home')):
                merged_row = {k: v for k, v in merged_row.items() if k not in columns_to_drop}
                store_aggregated_data_row_by_row(pd.DataFrame([merged_row]), match_stats_collection)
                logger.info(f"Successfully stored row: {merged_row['unique_id']}")
            if unmatched_pair:
                unmatched_pairs.append(unmatched_pair)
        
        if not merged_data:
            logger.error("No data was merged successfully")
            return
            
        # Export unmatched pairs to Excel if any exist
        if unmatched_pairs:
            unmatched_df = pd.DataFrame(unmatched_pairs)
            export_path = 'unmatched_ids.xlsx'
            unmatched_df.to_excel(export_path, index=False)
            logger.info(f"Exported {len(unmatched_pairs)} unmatched ID pairs to {export_path}")
        
        merged_df = pd.DataFrame(merged_data)
        merged_df.to_excel('merge_odds_data.xlsx', index=False)
        
        if merged_df.empty:
            logger.error("Merge operation resulted in empty DataFrame")
            return
            
        logger.info(f"Merged DataFrame shape: {merged_df.shape}")
        
        # Safe column access
        if 'Odd_Home' not in merged_df.columns:
            logger.error(f"Column 'Odd_Home' not found. Available columns: {merged_df.columns.tolist()}")
            return
            
        # Filter out rows where 'Odd_Home' is not null
        filtered_df = merged_df.dropna(subset=['Odd_Home'])

        # Export rows with missing 'Odd_Home' values to an Excel file for further analysis
        na_odds_df = merged_df[merged_df['Odd_Home'].isna()]
        if not na_odds_df.empty:
            na_odds_df.to_excel('missing_odds_data.xlsx', index=False)
            logger.info(f"Exported {len(na_odds_df)} rows with missing odds data to missing_odds_data.xlsx")

        # Drop specified columns if they exist in the DataFrame
        filtered_df.drop(columns=[col for col in columns_to_drop if col in filtered_df.columns], inplace=True)

        # Check if the filtered DataFrame is not empty before proceeding
        if not filtered_df.empty:
            logger.info(f"Final filtered DataFrame shape: {filtered_df.shape}")
        else:
            logger.error("No data after filtering")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
