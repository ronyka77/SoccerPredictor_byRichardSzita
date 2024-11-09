import subprocess
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util_tools.database import MongoClient

def ensure_backup_directory_exists(backup_path: str) -> None:
    """Ensure the backup directory exists, create if not."""
    if not os.path.exists(backup_path):
        os.makedirs(backup_path)

def backup_mongodb(backup_path: str) -> None:
    """
    Run the mongodump command to back up a MongoDB database.

    Args:
        backup_path (str): Directory path to store the backup.
    """
    try:
        # Initialize MongoDB client to get URI and database name
        client = MongoClient()
        with client.get_database() as db:
            uri = client.config.get('MongoDB', 'uri')
            db_name = db.name

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
        result = subprocess.run(mongodump_cmd, capture_output=True, text=True)

        # Check for any errors in the backup process
        if result.returncode == 0:
            print(f"Backup successful! Backup stored in: {backup_path}")
        else:
            print(f"Backup failed! Error: {result.stderr}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Backup directory
backup_dir = "./data/mongo_backup"  # Specify your backup folder path

# Ensure backup directory exists
ensure_backup_directory_exists(backup_dir)

# Call the backup function
backup_mongodb(backup_dir)
