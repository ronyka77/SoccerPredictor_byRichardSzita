from utils.database import MongoDBClient
from utils.logging_config import LoggerSetup
from utils.data_processing import DataProcessor

# Initialize logger
logger = LoggerSetup.setup_logger(
    name=__name__,
    log_file='./logs/aggregation.log'
)

# Use MongoDB client
db_client = MongoDBClient()
with db_client.get_database() as db:
    matches_collection = db['fixtures']
    stats_collection = db['match_stats']
    # Continue with your aggregation logic 