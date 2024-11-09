from pymongo import MongoClient, UpdateOne
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
import logging
from util_tools.mongo_add_id import mongo_add_running_id

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB setup
client = MongoClient('192.168.0.77', 27017)
db = client.football_data

# Collections
matches_collection = db.fixtures
stats_collection = db.match_stats
aggregated_collection = db.aggregated_data

# Create indexes for better query performance
def ensure_indexes():
    """Create necessary indexes if they don't exist"""
    try:
        matches_collection.create_index("unique_id")
        stats_collection.create_index("unique_id") 
        aggregated_collection.create_index("unique_id", unique=True)
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")

def aggregate_data():
    """Aggregate data with optimized DataFrame operations"""
    try:
        # Fetch only necessary fields from MongoDB
        projection = {"_id": 0}  # Exclude _id field, add other needed fields if necessary
        
        # Use generator expressions instead of list to reduce memory usage
        matches_df = pd.DataFrame(matches_collection.find({}, projection))
        stats_df = pd.DataFrame(stats_collection.find({}, projection))
        
        if matches_df.empty or stats_df.empty:
            logger.warning("No data found in one or both collections")
            return pd.DataFrame()

        # Convert to string type only for unique_id column
        matches_df['unique_id'] = matches_df['unique_id'].astype(str)
        stats_df['unique_id'] = stats_df['unique_id'].astype(str)

        # Optimize merge operation
        aggregated_df = pd.merge(
            matches_df, 
            stats_df, 
            how='left', 
            on='unique_id', 
            suffixes=('_match', '_stats')
        )
        
        aggregated_df.drop_duplicates(subset=['unique_id'], inplace=True)

        return aggregated_df
    except Exception as e:
        logger.error(f"Error in data aggregation: {e}")
        return pd.DataFrame()

def calculate_outcome(score: str) -> int:
    """Calculate match outcome with error handling"""
    if not isinstance(score, str):
        return None
        
    try:
        home_goals, away_goals = map(int, score.split('–'))
        if home_goals > away_goals:
            return 1
        elif home_goals < away_goals:
            return -1
        return 0
    except Exception as e:
        # logger.error(f"Error calculating outcome for score {score}: {e}")
        return None

def store_aggregated_data(aggregated_data: pd.DataFrame, batch_size: int = 1000):
    """Store aggregated data using bulk operations"""
    if aggregated_data.empty:
        logger.warning("No data to store")
        return

    try:
        # Pre-calculate outcomes for all records
        aggregated_data['match_outcome'] = aggregated_data['Score'].apply(calculate_outcome)
        
        # Convert to records and prepare bulk operations
        records = aggregated_data.to_dict('records')
        bulk_operations = []
        
        for record in records:
            if 'unique_id' not in record:
                logger.warning(f"Record missing unique_id, skipping: {record}")
                continue
                
            bulk_operations.append(
                UpdateOne(
                    {'unique_id': record['unique_id']},
                    {'$set': record},
                    upsert=True
                )
            )
            
            # Execute bulk operations in batches
            if len(bulk_operations) >= batch_size:
                try:
                    aggregated_collection.bulk_write(bulk_operations, ordered=False)
                    bulk_operations = []
                except Exception as e:
                    logger.error(f"Error in bulk write operation: {e}")

        # Execute remaining operations
        if bulk_operations:
            try:
                aggregated_collection.bulk_write(bulk_operations, ordered=False)
            except Exception as e:
                logger.error(f"Error in final bulk write operation: {e}")

        logger.info("Aggregated data stored/updated in MongoDB")
    except Exception as e:
        logger.error(f"Error storing aggregated data: {e}")

if __name__ == '__main__':
    try:
        # Ensure indexes exist
        ensure_indexes()
        
        # Aggregate and clean the data
        logger.info("Starting data aggregation...")
        aggregated_data = aggregate_data()
        
        if not aggregated_data.empty:
            logger.info("Data aggregation complete, starting database merge...")
            # Store the aggregated data back into MongoDB
            store_aggregated_data(aggregated_data)
            
            logger.info("Running ID addition...")
            mongo_add_running_id()
            
            logger.info("Process completed successfully")
        else:
            logger.error("No data was aggregated, stopping process")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        client.close()
