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

logger = LoggerSetup.setup_logger(
    name='merge_odds',
    log_file='./data_tools/log/merge_odds.log',
    level=logging.INFO
)

# Initialize MongoDB client
db_client = MongoClient()

# Function to drop Odd_Home, Odd_Away, and Odd_Draw columns from the aggregated_data collection in MongoDB
def drop_odds_columns() -> None:
    """Drop Odd_Home, Odd_Away, and Odd_Draw columns from the aggregated_data collection in MongoDB."""
    try:
        with db_client.get_database() as db:
            result = db.fixtures.update_many(
                {},
                {"$unset": {"Odd_Home": "", "Odd_Away": "", "Odds_Draw": ""}}
            )
            logger.info(f"Successfully dropped columns from {result.modified_count} documents.")
    except Exception as e:
        logger.error(f"Failed to drop columns: {str(e)}")
        raise

# drop_odds_columns()

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data from match_stats and odds_data collections in MongoDB.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Match stats and odds data frames
    
    Raises:
        ConnectionError: If MongoDB connection fails
    """
    try:
        with db_client.get_database() as db:
           
            odds_data_df = pd.DataFrame(list(db.odds_data.find()))
            match_stats_df = pd.DataFrame(list(db.fixtures.find({"Odd_Home": {"$exists": False}})))
            match_stats_df['Date'] = pd.to_datetime(match_stats_df['Date'])
            odds_data_df['Date'] = pd.to_datetime(odds_data_df['Date'])
            match_stats_df = match_stats_df[match_stats_df['Date'] < odds_data_df['Date'].max()]
            odds_data_df.to_excel('odds_data.xlsx', index=False)
            return match_stats_df, odds_data_df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise ConnectionError("Failed to connect to MongoDB")

def standardize_name(name: str) -> str:
    """Standardize team names by removing common abbreviations and special characters."""
    replacements = {
        "ã": "a",
        "ñ": "n",
        "é": "e",
        "ò": "o",
        "á": "a",
        "ó": "o",
        "Arminia": "Arminia Bielefeld",
        "Athletic Club": "Ath Bilbao",
        "Atletico Madrid": "Atl. Madrid",
        "Athletic Club": "Athletic Bilbao",
        "Athletico Paranaense": "Athletico-PR",
        "B. Monchengladbach": "Gladbach",
        "Cádiz": "Cadiz CF  ",
        "Nott'ham Forest": "Nottingham", 
        "Paris S-G": "PSG",
        "Leeds United": "Leeds",
        "FC Barcelona": "Barcelona",
        "FC Bayern": "Bayern Munich",
        "FC Internazionale": "Inter Milan",
        "FC Porto": "Porto",
        "Köln": "FC Koln",
        "Leverkursen": "Bayer Leverkusen",
        "Hellas Verona": "Verona",
        "Hertha BSC":"Hertha Berlin",
        "Luton Town": "Luton",
        "Saint-Étienne": "St Etienne",
        "Newcastle Utd": "Newcastle",
        "Norwich City": "Norwich",
        "Mainz 05": "Mainz"
    }
    name = str(name).strip()
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.strip()

def fuzzy_merge_row(row: pd.Series, odds_data_df: pd.DataFrame, threshold: int = 90) -> Tuple[dict, Optional[dict]]:
    """Merge a single row from match_stats with odds_data based on fuzzy matching of unique_id and matching date."""
    try:
        match_id = str(row['unique_id'])
        match_date = row['Date']
        # Standardize the match ID before comparison
        standardized_match_id = standardize_name(match_id)

        # Standardize all odds IDs for comparison
        odds_unique_ids = [str(uid) for uid in odds_data_df['unique_id'].tolist()]
        standardized_odds_ids = [standardize_name(uid) for uid in odds_unique_ids]
        
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

def store_aggregated_data_row_by_row(df: pd.DataFrame) -> None:
    """Store each row of the DataFrame into the aggregated_data collection in MongoDB.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to be stored.
    
    Raises:
        ValueError: If the DataFrame is empty.
    """
    if df.empty:
        logger.error("The DataFrame to store is empty.")
        raise ValueError("The DataFrame to store is empty.")
    
    try:
        with db_client.get_database() as db:
            collection = db.aggregated_data
            operations = []
            for _, row in df.iterrows():
                # Convert the row to a dictionary and prepare the update operation
                row_dict = row.to_dict()
                running_id = row_dict.get('running_id')
                if running_id is not None:
                    operations.append(
                        UpdateOne(
                            {'running_id': running_id},
                            {'$set': row_dict},
                            upsert=True
                        )
                    )
            
            for operation in operations:
                result = collection.update_one(operation.filter, operation.update, upsert=True)
                if result.modified_count > 0 or result.upserted_id is not None:
                    logger.info(f"Inserted/Updated document with filter: {operation.filter}")
                else:
                    logger.warning(f"No changes made for document with filter: {operation.filter}")
    
    except Exception as e:
        logger.error(f"Failed to store data row by row: {str(e)}")
        raise
    
# Drop 'League' and '_id' columns if they exist in the merged_row
columns_to_drop = ['League', '_id']

def main() -> None:
    try:
        # Add collection verification
        with db_client.get_database() as db:
            logger.info(f"Available collections: {db.list_collection_names()}")
            logger.info(f"Fixtures count: {db.fixtures.count_documents({})}")
            logger.info(f"Odds data count: {db.odds_data.count_documents({})}")

        # Load data
        match_stats_df, odds_data_df = load_data()
        
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
                store_aggregated_data_row_by_row(pd.DataFrame([merged_row]))
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
