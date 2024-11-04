from pymongo import MongoClient
import pandas as pd
from mongo_add_id import mongo_add_running_id

# MongoDB setup
client = MongoClient('192.168.0.77', 27017)
db = client.football_data

# Collections
matches_collection = db.fixtures
stats_collection = db.match_stats

# Function to aggregate data
def aggregate_data():
    # Fetch data from MongoDB collections
    matches = list(matches_collection.find({}))
    stats = list(stats_collection.find({}))

    # Convert to DataFrame for easier manipulation
    matches_df = pd.DataFrame(matches)
    stats_df = pd.DataFrame(stats)

    # Ensure the 'id' columns in both DataFrames are of the same type
    matches_df['unique_id'] = matches_df['unique_id'].astype(str)
    stats_df['unique_id'] = stats_df['unique_id'].astype(str)

    # Merge matches with stats data
    aggregated_df = pd.merge(matches_df, stats_df, how='left', on='unique_id', suffixes=('_match', '_stats'))

    # Perform data cleaning (e.g., handling missing values)
    aggregated_df.fillna(value={'stats_key': 'N/A'}, inplace=True)

    # Remove duplicates (if any)
    aggregated_df.drop_duplicates(subset=['unique_id'], inplace=True)

    return aggregated_df
# Function to calculate match outcome based on the score
def calculate_outcome(score):
    try:
        # Split the score (e.g., "2–1" to ["2", "1"])
        home_goals, away_goals = map(int, score.split('–'))
        # Determine the outcome based on the goals
        if home_goals > away_goals:
            return 1  # Home win
        elif home_goals < away_goals:
            return -1  # Away win
        else:
            return 0  # Draw
        
    except Exception:
        return None
    
# Function to store aggregated data back in MongoDB
def store_aggregated_data(aggregated_data):
    aggregated_collection = db.aggregated_data
    # Convert DataFrame back to dictionary
    aggregated_data_dict = aggregated_data.to_dict('records')
    # Insert or update the aggregated data in MongoDB
    for record in aggregated_data_dict:
        # Find the documents with a non-empty Score field
        score = record['Score']
        record['match_outcome'] = calculate_outcome(score)

        match_id = record['unique_id']
        if aggregated_collection.find_one({'unique_id': match_id}) is None:
            aggregated_collection.insert_one(record)
        else:
            aggregated_collection.update_one(
                {'unique_id': match_id},
                {'$set': record}
            )
    print(f"Aggregated data stored/updated in MongoDB.")

if __name__ == '__main__':
    # Aggregate and clean the data
    aggregated_data = aggregate_data()
    print("Data aggregation done, start merge into database...")
    # Store the aggregated data back into MongoDB
    store_aggregated_data(aggregated_data)

    print("Data aggregation and cleaning complete.")
    mongo_add_running_id()
