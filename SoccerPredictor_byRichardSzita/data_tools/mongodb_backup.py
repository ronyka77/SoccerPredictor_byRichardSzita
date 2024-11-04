import subprocess
from pymongo import MongoClient
import os

# MongoDB connection details
mongodb_uri = "mongodb://192.168.0.77:27017"  # Replace with your MongoDB URI
database_name = "football_data"               # Replace with your database name

# Backup directory
backup_dir = "./SoccerPredictor_byRichardSzita/data/mongo_backup"  # Specify your backup folder path

# Ensure backup directory exists
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)

# Function to run the mongodump command from Python
def backup_mongodb(uri, db_name, backup_path):
    try:
        # Full path to mongodump
        mongodump_path = r"C:\Program Files\MongoDB\Tools\100\bin\mongodump.exe"
        # Construct the mongodump command
        mongodump_cmd = [
            mongodump_path, 
            f"--uri={uri}",        # MongoDB URI
            f"--db={db_name}",     # Database name to back up
            f"--out={backup_path}" # Output directory for backup
        ]

        # Run the mongodump command using subprocess
        result = subprocess.run(mongodump_cmd, capture_output=True)

        # Check for any errors in the backup process
        if result.returncode == 0:
            print(f"Backup successful! Backup stored in: {backup_path}")
        else:
            print(f"Backup failed! Error: {result.stderr.decode()}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Call the backup function
backup_mongodb(mongodb_uri, database_name, backup_dir)
