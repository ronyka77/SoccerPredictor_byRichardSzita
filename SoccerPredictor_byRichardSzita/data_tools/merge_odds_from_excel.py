import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pymongo
import logging
from util_tools.database import MongoClient

# Set up logger
logger = logging.getLogger('merge_odds_from_excel')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('./log/merge_odds_from_excel.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize MongoDB client and get database connection
db_client = MongoClient()
db = db_client.get_collection('fixtures')

def load_excel_data(file_path: str) -> pd.DataFrame:
    """
    Load data from an Excel file.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    try:
        data = pd.read_excel(file_path)
        if '_id' in data.columns:
            data.drop(columns=['_id'], inplace=True)
        logger.info(f"Data loaded from {file_path} with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def upsert_to_mongodb(data: pd.DataFrame, collection_name: str):
    """
    Upsert data to MongoDB collection.

    Args:
        data (pd.DataFrame): The data to upsert.
        collection_name (str): The name of the MongoDB collection.
    """
    try:
        collection = db
        operations = [
            pymongo.UpdateOne(
                {'unique_id': record['unique_id']},
                {'$set': record},
                upsert=True
            )
            for record in data.to_dict('records')
        ]
        collection.bulk_write(operations, ordered=False)
        logger.info(f"Upserted {len(operations)} records")
    except Exception as e:
        logger.error(f"Error upserting data to {collection_name} collection: {str(e)}")
        raise
def main():
    try:
        # Define file path and collection name
        file_path = './merge_odds_data.xlsx'
        collection_name = 'fixtures'

        # Load data from Excel
        data = load_excel_data(file_path)

        # Upsert data to MongoDB using unique_id for matching
        upsert_to_mongodb(data, collection_name)
        
        # logger.info("Data imported and upserted to MongoDB successfully.")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

