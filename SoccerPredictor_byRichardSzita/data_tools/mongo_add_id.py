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

