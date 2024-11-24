from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://192.168.0.77:27017/')  # Change the URI to your MongoDB connection
db = client['football_data']  # Replace with your database name
collection = db['aggregated_data']  # Replace with your collection name

def mongo_add_running_id():
    # Step 1: Find the current maximum running_id in the collection
    max_running_id_doc = collection.find_one(
        {'running_id': {'$exists': True}},  # Only consider documents that already have a running_id
        sort=[('running_id', -1)]  # Sort by running_id in descending order to get the max value
    )

    # Get the current max running_id, if none exists, start from 1
    if max_running_id_doc:
        counter = max_running_id_doc['running_id'] + 1
    else:
        counter = 1

    # Step 2: Iterate over documents that don't have a running_id and assign a new one
    for document in collection.find({'running_id': {'$exists': False}}):  # Only documents without running_id
        # Set a running ID for each document that doesn't have one
        collection.update_one(
            {'_id': document['_id']},  # Find the document by its unique _id
            {'$set': {'running_id': counter}}  # Add the running_id field with the current counter value
        )
        counter += 1  # Increment the counter

    print(f"Added running IDs starting from {counter-1}.")

def update_fixtures_unique_id():
    """Update unique_id format in fixtures collection to Date_Home_Away."""
    fixtures_collection = db['fixtures']
    
    # Find all documents in fixtures collection
    for doc in fixtures_collection.find():
        try:
            # Extract required fields
            date = doc.get('Date')
            home = doc.get('Home')
            away = doc.get('Away')
            
            # Skip if any required field is missing
            if not all([date, home, away]):
                print(f"Skipping document {doc['_id']} - Missing required fields")
                continue
                
            # Create new unique_id in required format
            new_unique_id = f"{date}_{home}_{away}"
            
            # Update document with new unique_id
            fixtures_collection.update_one(
                {'_id': doc['_id']},
                {'$set': {'unique_id': new_unique_id}}
            )
            
        except Exception as e:
            print(f"Error processing document {doc.get('_id')}: {str(e)}")
            continue
            
    print("Finished updating unique_ids in fixtures collection")

# update_fixtures_unique_id()
def update_match_stats_unique_id():
    """Update unique_id in match_stats collection by joining with fixtures on Match Report URL."""
    match_stats_collection = db['match_stats']
    fixtures_collection = db['fixtures']

    # Create index on Match Report field for faster lookup
    fixtures_collection.create_index([("Match Report", 1)])

    # Find all documents in match_stats collection
    for match_stat in match_stats_collection.find():
        try:
            # Get the Match Report URL from match_stats
            match_report_url = match_stat.get('url')
            
            if not match_report_url:
                print(f"Skipping match_stat {match_stat['_id']} - Missing Match Report URL")
                continue

            # Find corresponding fixture document
            fixture = fixtures_collection.find_one({"Match Report": match_report_url})
            
            if not fixture:
                print(f"No matching fixture found for URL: {match_report_url}")
                continue

            # Get unique_id from fixture
            new_unique_id = fixture.get('unique_id')
            
            if not new_unique_id:
                print(f"Fixture {fixture['_id']} missing unique_id")
                continue

            # Update match_stats document with new unique_id
            match_stats_collection.update_one(
                {'_id': match_stat['_id']},
                {'$set': {'unique_id': new_unique_id}}
            )

        except Exception as e:
            print(f"Error processing match_stat {match_stat.get('_id')}: {str(e)}")
            continue

    print("Finished updating unique_ids in match_stats collection")

# update_match_stats_unique_id()

def drop_team_names():
    """Drop documents from fixtures collection where Home column does not exist."""
    try:
        fixtures_collection = db['fixtures']
        
        # Delete documents where Home field does not exist
        result = fixtures_collection.delete_many({
            'Home': {'$exists': 0}
        })
        
        print(f"Dropped {result.deleted_count} documents missing Home column from fixtures collection")
        
    except Exception as e:
        print(f"Error dropping documents: {str(e)}")
        raise

drop_team_names()
