from configparser import ConfigParser
import os

class Config:
    def __init__(self):
        self.config = ConfigParser()
        self.config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
        self.load_config()
    
    def load_config(self):
        if not self.config.read(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
    
    @property
    def mongodb_uri(self):
        return self.config.get('MongoDB', 'uri', fallback='mongodb://192.168.0.77:27017/')
    
    @property
    def database_name(self):
        return self.config.get('MongoDB', 'database', fallback='football_data') 