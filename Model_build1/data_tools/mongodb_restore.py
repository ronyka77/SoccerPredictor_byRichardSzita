import subprocess
from pymongo import MongoClient
import os

# MongoDB connection details
mongodb_uri = "mongodb://localhost:27017"  # Replace with your MongoDB URI
database_name = "football_data"               # Replace with your database name
# Backup directory
backup_dir = "./backup"  # Specify your backup folder path

# Ensure backup directory exists
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)
    
def restore_mongodb(backup_path, uri):
    try:
        # Construct the mongorestore command
        mongorestore_cmd = [
            "mongorestore", 
            f"--uri={uri}",        # MongoDB URI
            f"--dir={backup_path}" # Backup directory to restore from
        ]

        # Run the mongorestore command
        result = subprocess.run(mongorestore_cmd, capture_output=True)

        # Check for errors
        if result.returncode == 0:
            print(f"Restore successful! Data restored from: {backup_path}")
        else:
            print(f"Restore failed! Error: {result.stderr.decode()}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Call the Restore function
restore_mongodb(mongodb_uri, database_name, backup_dir)