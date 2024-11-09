from contextlib import contextmanager
from typing import Generator, Optional, Any
from pymongo import MongoClient as PyMongoClient
from pymongo.database import Database
from pymongo.collection import Collection
import logging
from configparser import ConfigParser
import os

class MongoClient:
    """A wrapper class for MongoDB client with configuration management and connection handling"""
    
    def __init__(self, config_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'database.ini')):
        """Initialize MongoDB client with configuration
        
        Args:
            config_path: Path to the database configuration file
        """
        logging.debug(f"Attempting to load config from: {config_path}")
        self.config = self._load_config(config_path)
        self._client: Optional[PyMongoClient] = None
    
    def _load_config(self, config_path: str) -> ConfigParser:
        """Load and validate database configuration
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            ConfigParser object with loaded configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        config = ConfigParser()
        config.read(config_path)
        
        # Validate required config sections and keys
        if not config.has_section('MongoDB'):
            raise ValueError("MongoDB section missing in config")
        if not config.has_option('MongoDB', 'uri'):
            raise ValueError("MongoDB URI missing in config")
        if not config.has_option('MongoDB', 'database'):
            raise ValueError("Database name missing in config")
            
        return config
    
    def connect(self):
        """Establish a connection to the MongoDB client."""
        if not self._client:
            self._client = PyMongoClient(self.config.get('MongoDB', 'uri'))
            logging.debug("MongoDB client connected.")

    def close(self):
        """Close the MongoDB client connection."""
        if self._client:
            self._client.close()
            self._client = None
            logging.debug("MongoDB client connection closed.")
    
    @contextmanager
    def get_database(self) -> Generator[Database, None, None]:
        """Context manager for database connections
        
        Yields:
            MongoDB database instance
            
        Raises:
            ConnectionError: If connection to database fails
        """
        try:
            self.connect()
            yield self._client[self.config.get('MongoDB', 'database')]
        except Exception as e:
            logging.error(f"Database connection error: {str(e)}")
            raise ConnectionError(f"Failed to connect to database: {str(e)}")

    def get_collection(self, collection_name: str) -> Collection:
        """Get a specific collection with error handling
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            MongoDB collection instance
            
        Raises:
            ValueError: If collection name is invalid
            ConnectionError: If database connection fails
        """
        if not collection_name or not isinstance(collection_name, str):
            raise ValueError("Invalid collection name")
            
        try:
            with self.get_database() as db:
                return db[collection_name]
        except Exception as e:
            logging.error(f"Error accessing collection {collection_name}: {str(e)}")
            raise ConnectionError(f"Failed to access collection {collection_name}: {str(e)}")