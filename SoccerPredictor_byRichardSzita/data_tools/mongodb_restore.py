import subprocess
import os
import sys
import platform

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util_tools.database import MongoClient  # Use the custom MongoClient
# Backup directory
backup_dir = "./data/mongo_backup/"  # Specify your backup folder path

def get_mongorestore_path():
    """
    Determine the path to mongorestore executable based on the operating system.
    
    Returns:
        str: Path to mongorestore executable
    """
    system = platform.system().lower()
    
    if system == "windows":
        # Common MongoDB installation paths on Windows
        possible_paths = [
            r"C:\Program Files\MongoDB\Tools\100\bin\mongorestore.exe",
            r"C:\Program Files\MongoDB\Server\6.0\bin\mongorestore.exe",
            r"C:\Program Files\MongoDB\Server\5.0\bin\mongorestore.exe",
            r"C:\Program Files\MongoDB\Server\4.4\bin\mongorestore.exe"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
    else:
        # On Unix-like systems, try common locations
        possible_paths = [
            "/usr/bin/mongorestore",
            "/usr/local/bin/mongorestore",
            "/opt/homebrew/bin/mongorestore"  # Common on macOS with Homebrew
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
    
    # If mongorestore is in PATH, return just the command name
    try:
        subprocess.run(["mongorestore", "--version"], 
                      capture_output=True, 
                      check=False)
        return "mongorestore"
    except FileNotFoundError:
        raise FileNotFoundError("mongorestore executable not found. Please ensure MongoDB tools are installed and in your PATH")

# Ensure backup directory exists
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)
    
def restore_mongodb(backup_path):
    """
    Restore MongoDB database from backup.
    
    Args:
        backup_path (str): Path to the backup directory
    
    Raises:
        FileNotFoundError: If mongorestore executable is not found
        subprocess.CalledProcessError: If restore operation fails
    """
    try:
        mongorestore_path = get_mongorestore_path()
        
        # Initialize MongoDB client to get URI
        client = MongoClient()
        uri = client.config.get('MongoDB', 'uri')
        
        # Construct the mongorestore command
        mongorestore_cmd = [
            mongorestore_path, 
            f"--uri={uri}",        # MongoDB URI
            f"--dir={backup_path}" # Backup directory to restore from
        ]

        # Run the mongorestore command
        result = subprocess.run(mongorestore_cmd, 
                              capture_output=True, 
                              text=True,
                              check=True)

        print(f"Restore successful! Data restored from: {backup_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Restore failed! Error: {e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Call the Restore function
if __name__ == "__main__":
    restore_mongodb(backup_dir)