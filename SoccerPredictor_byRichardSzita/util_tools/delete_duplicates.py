import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import List, Dict, Any
from util_tools.database import MongoClient
from util_tools.logging_config import LoggerSetup
import pandas as pd


class DuplicateHandler:
    """Class to handle finding and removing duplicate entries in MongoDB collections."""

    def __init__(self, collection_name: str = 'fixtures'):
        """Initialize DuplicateHandler with collection name and setup logging.
        
        Args:
            collection_name: Name of MongoDB collection to process. Defaults to 'fixtures'.
        """
        self.collection_name = collection_name
        # Initialize logger before anything else
        self.logger = LoggerSetup.setup_logger(
            name='delete_duplicates',
            log_file='./util_tools/log/delete_duplicates.log',
            level=logging.INFO
        )
        try:
            self.client = MongoClient()
            # Test database connection immediately
            with self.client.get_database() as db:
                collection = db[self.collection_name]
                doc_count = collection.count_documents({})
                self.logger.info(f"Successfully connected to database. Found {doc_count} documents.")
        except Exception as e:
            self.logger.error(f"Failed to initialize database connection: {str(e)}")
            raise

    @staticmethod 
    def standardize_name(name: str) -> str:
        """Standardize team names by removing common abbreviations and special characters."""
        if not name:
            return ""
            
        replacements = {
            "â": "a",
            "ã": "a",
            "ß": "ss",
            "ç": "c",      
            "é": "e",
            "è": "e",
            "ê": "e",
            "ï": "i",
            "ò": "o",
            "á": "a",
            "ô": "o",
            "ő": "o",
            "ó": "o",
            "ű": "u",
            "ü": "u",
            "ñ": "n", 
            "Arminia": "Arminia Bielefeld",
            "AC Ajaccio": "Ajaccio",
            "Athletic Club": "Ath Bilbao",
            "Atletico Madrid": "Atl. Madrid", 
            "Athletic Club": "Athletic Bilbao",
            "Athletico Paranaense": "Athletico-PR",
            "B. Monchengladbach": "Gladbach",
            "Cádiz": "Cadiz CF  ",
            "Nott'ham Forest": "Nottingham",
            "Paris S-G": "PSG",
            "Leeds United": "Leeds",
            "FC Barcelona": "Barcelona",
            "FC Bayern": "Bayern Munich", 
            "FC Internazionale": "Inter Milan",
            "FC Porto": "Porto",
            "Köln": "FC Koln",
            "Leverkursen": "Bayer Leverkusen",
            "Hellas Verona": "Verona",
            "Hertha BSC":"Hertha Berlin",
            "Luton Town": "Luton",
            "Saint-Étienne": "St Etienne",
            "Newcastle Utd": "Newcastle",
            "Norwich City": "Norwich",
            "Mainz 05": "Mainz",
            "Sparta R'dam": "Sparta Rotterdam"
        }
        name = str(name).strip()
        for old, new in replacements.items():
            name = name.replace(old, new)
        return name.strip()

    def find_duplicates(self) -> List[Dict[str, Any]]:
        """Find duplicate entries in the collection based on standardized team names and dates.
        
        Returns:
            List of documents that are duplicates
        """
        try:
            duplicates = []
            
            with self.client.get_database() as db:
                collection = db[self.collection_name]
                
                # Create index on Odd_Home for faster sorting
                collection.create_index([("Odd_Home", -1)])
                
                # Get all documents and verify we have data
                # Only fetch fields we need
                documents = list(collection.find({}, {
                    'unique_id': 1,
                    '_id': 1, 
                    'Home': 1,
                    'Away': 1,
                    'Date': 1,
                    'Odd_Home': 1,
                    'Odd_Away': 1,
                    'Odds_Draw': 1
                }))
                
                if not documents:
                    self.logger.warning("No documents found in collection")
                    return []
                    
                self.logger.info(f"Processing {len(documents)} documents. This may take a few minutes...")
                print(f"Processing {len(documents)} documents. This may take a few minutes...")
                
                # Pre-standardize all team names to avoid repeated processing
                for doc in documents:
                    doc['std_home'] = self.standardize_name(doc.get('Home', ''))
                    doc['std_away'] = self.standardize_name(doc.get('Away', ''))
                    try:
                        doc['date_obj'] = pd.to_datetime(doc.get('Date', ''))
                    except:
                        self.logger.error(f"Invalid date format for document {doc.get('unique_id')}: {doc.get('Date')}")
                        doc['date_obj'] = None
                
                # Filter out documents with invalid dates
                documents = [doc for doc in documents if doc['date_obj'] is not None]
                
                # Sort documents by Odd_Home in descending order
                documents.sort(key=lambda x: float(x.get('Odd_Home', '0') if x.get('Odd_Home') not in [None, '', '-'] else '0'), reverse=True)
                self.logger.info(f"Sorted {len(documents)} documents by Odd_Home")
            
                # Create a dictionary to track unique combinations
                seen_matches = {}
                
                # Process in batches for better memory management
                batch_size = 1000
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    self.logger.info(f"Processing batch {i // batch_size + 1} of {len(documents) // batch_size + 1}")
                    print(f"Processing batch {i // batch_size + 1} of {len(documents) // batch_size + 1}")  
                    for doc in batch:
                        base_id = doc.get('unique_id', '')
                        
                        # Check for matches within ±2 days
                        match_found = False
                        for key, existing in seen_matches.items():
                            existing_date = existing['date']
                            date_diff = abs((doc['date_obj'] - existing_date).days)
                            
                            if (date_diff <= 2 and 
                                doc['std_home'] == existing['home'] and 
                                doc['std_away'] == existing['away']):
                                # This is a duplicate
                                duplicates.append(doc)
                                match_found = True
                                
                                # Merge odds data from duplicate into original document
                                original_id = existing['_id']
                                odds_update = {k: doc[k] for k in ['Odd_Home', 'Odd_Away', 'Odds_Draw'] 
                                             if k in doc and doc[k]}
                                
                                if odds_update:
                                    collection.update_one(
                                        {'_id': original_id},
                                        {'$set': odds_update}
                                    )
                                break
                                
                        if not match_found:
                            # No duplicate found, add to seen matches
                            match_key = f"{doc['date_obj']}_{doc['std_home']}_{doc['std_away']}"
                            seen_matches[match_key] = {
                                '_id': doc['_id'],
                                'date': doc['date_obj'],
                                'home': doc['std_home'],
                                'away': doc['std_away']
                            }
            
            self.logger.info(f"Found {len(duplicates)} duplicate entries in {self.collection_name}")
            print(f"Found {len(duplicates)} duplicate entries in {self.collection_name}")
            return duplicates
            
        except Exception as e:
            self.logger.error(f"Error finding duplicates in {self.collection_name}: {str(e)}")
            raise

    def delete_duplicates(self) -> None:
        """Delete duplicate entries from the collection."""
        try:
            self.logger.info("Starting duplicate deletion process")
            duplicates = self.find_duplicates()
            
            if not duplicates:
                self.logger.info(f"No duplicates found in {self.collection_name}")
                return
                
            with self.client.get_database() as db:
                collection = db[self.collection_name]
                
                # Delete each duplicate document
                duplicate_ids = [doc['_id'] for doc in duplicates]
                
                result = collection.delete_many({'_id': {'$in': duplicate_ids}})
                
                self.logger.info(f"Successfully deleted {result.deleted_count} duplicate documents from {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"Error deleting duplicates from {self.collection_name}: {str(e)}")
            raise

if __name__ == "__main__":
    # This code block runs when the script is executed directly (not imported as a module)
    # Create a DuplicateHandler instance to manage the 'fixtures' collection
    duplicate_handler = DuplicateHandler(collection_name='fixtures')
    
    # Call delete_duplicates() to find and remove any duplicate entries
    # This will:
    # 1. Connect to MongoDB
    # 2. Find duplicate entries based on date and team names
    # 3. Delete the duplicate documents while preserving the original entries
    duplicate_handler.delete_duplicates()
