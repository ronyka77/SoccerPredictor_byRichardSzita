from contextlib import contextmanager
from typing import Generator
from pymongo import MongoClient
from pymongo.database import Database

class MongoDBClient:
    def __init__(self, uri: str, database: str):
        self.uri = uri
        self.database = database
        self._client = None
    
    @contextmanager
    def get_database(self) -> Generator[Database, None, None]:
        try:
            if not self._client:
                self._client = MongoClient(self.uri)
            yield self._client[self.database]
        finally:
            if self._client:
                self._client.close()
                self._client = None 