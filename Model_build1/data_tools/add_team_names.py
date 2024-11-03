import pandas as pd
from pymongo import MongoClient
# MongoDB setup
client = MongoClient('192.168.0.77', 27017)
db = client.football_data

# Collections
collection = db.aggregated_data


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

# Path to the second Excel file (replace with actual file path)
base_path = './Model_build1/data/model_data_prediction.xlsx'
prediction_path = './Model_build1/hybrid_predictions.xlsx'

prediction_df = pd.read_excel(prediction_path)
base_df = pd.read_excel(base_path)

# Call the function to merge the new data
updated_df = add_home_away_columns(prediction_df)

# Print the updated DataFrame
updated_df.to_excel('./Model_build1/hybrid_predictions_with_names.xlsx')
