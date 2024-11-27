import requests  # For making HTTP requests (not currently used)
from bs4 import BeautifulSoup  # For parsing HTML content (not currently used)
import pandas as pd  # For data manipulation
from pymongo import MongoClient  # For interacting with MongoDB
from collections import Counter  # For counting elements (not currently used)
from fake_useragent import UserAgent  # For generating random user agents
from selenium import webdriver  # For automating web interaction
from selenium.webdriver.chrome.service import Service  # For setting up ChromeDriver as a service
from selenium.webdriver.common.by import By  # For locating elements by selector
from selenium.webdriver.chrome.options import Options  # For configuring Chrome options
from selenium.webdriver.support.ui import WebDriverWait  # For adding explicit wait conditions
from selenium.webdriver.support import expected_conditions as EC  # For defining expected conditions in waits
from webdriver_manager.chrome import ChromeDriverManager  # For automatic ChromeDriver installation
import logging  # For logging information
import random  # For random selections, used here for user-agent rotation
import time  # For adding delays between actions
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util_tools.delete_duplicates import DuplicateHandler
# Suppress pandas warnings about duplicate columns
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
# Add this specific warning filter
warnings.filterwarnings('ignore', message='DataFrame columns are not unique')

# Set up logging
log_file_path = './data_tools/log/fbref_get_data.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of user agents for rotating to mimic different browsers and avoid bot detection
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36"
]

def delete_duplicate_fixtures():
    """Delete duplicate entries from the fixtures collection in MongoDB.
    
    Raises:
        Exception: If there is an error during the duplicate deletion process.
    """
    try:
        logging.info("Starting duplicate deletion process")
        print('Starting duplicate deletion process')
        # Create duplicate handler instance
        duplicate_handler = DuplicateHandler('fixtures')
        
        # Run duplicate deletion
        duplicate_handler.delete_duplicates()
        logging.info("Duplicate deletion completed successfully")
        
    except Exception as e:
        logging.error(f"Error during duplicate deletion: {str(e)}")
        raise

# Function to initialize and configure the Chrome WebDriver with specific options
def initialize_driver():
    """Initialize the Chrome WebDriver with custom options, including user agent and headless mode."""
    chrome_options = Options()
    chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")  # Choose a random user agent for anonymity
    chrome_options.add_argument("--headless")  # Run in headless mode (no GUI) for faster execution
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Instantiate the WebDriver for use throughout the script
driver = initialize_driver()

# Function to create a list of URLs to scrape for each league and season combination
def get_all_urls():
    """Generate URLs for specific leagues and seasons to scrape data from."""
    leagues = [
        ('Champions-League', '8'),  # Europe
        ('Premier League', '9'),  # England
        ('Championship','10'),  # England
        ('League One', '15'),  # England
        ('La Liga', '12'),  # Spain
        ('Segunda-Division', '17'),  # Spain
        ('Serie A', '11'),  # Italy
        ('Serie B', '18'),  # Italy
        ('Ligue 1', '13'),  # France
        ('Ligue 2', '60'),  # France
        ('Bundesliga', '20'),  # Germany
        ('2-Bundesliga','33'),  # Germany
        ('3-Liga','59'),  # Germany
        ('Liga-Profesional-Argentina', '21'),  # Argentina
        ('Serie A', '24'),  # Brazil Serie A
        ('Serie-B','38'),  # Brazil Serie B
        ('Eredivisie','23'),  # Netherlands
        ('Eerste-Divisie','51'),  # Netherlands
        ('J1-League','25'),  # Japan
        ('Super-Lig','26'),  # Turkey
        ('Allsvenskan','29'),  # Sweden
        ('Superettan','48'),  # Sweden
        ('Russian-Premier-League','30'),  # Russia       
        ('Primeira-Liga','32'),  # Portugal
        ('Ekstraklasa','36'),  # Poland
        ('Scottish-Premiership','40'),  # Scotland
        ('Primera-A','41'), #Colombia
        ('Liga-I','47'), #Romania
        ('Danish-Superliga','50') #Denmark
    ]
    
    # leagues = [
    #     ('Super-Lig','26'),  # Turkey
    #     ('Scottish-Premiership','40'),  # Scotland
    #     ('Primera-A','41'), #Colombia
    #     ('Liga-I','47'), #Romania
    #     ('Danish-Superliga','50') #Denmark
    # ]

    # Specific seasons to scrape data for; add additional seasons as needed
    seasons = ["2020-2021","2021-2022"] 
    urls = []  # List to store generated URLs
    
    # Generate URLs for each league and season, appending to the urls list
    for league, league_id in leagues:
        for season in seasons:
            league_name = league.replace(" ", "-")  # Replace spaces with hyphens for URL formatting
            url = f'https://fbref.com/en/comps/{league_id}/{season}/schedule/{season}-{league_name}-Scores-and-Fixtures'
            urls.append((url, league, season))  # Append URL, league, and season as a tuple
    
    return urls  # Return the complete list of URLs

# Function to fetch HTML content for a specific URL and insert data into MongoDB
def get_html_data(url, league, season, collection):
    """Scrape match data from a URL, process it, and store it in MongoDB."""
    driver.get(url)  # Open the URL in WebDriver
    time.sleep(3)  # Wait for the page to load
    
    table = driver.find_element(By.CSS_SELECTOR, "table")  # Locate the main table containing match data

    # Extract table headers
    headers = []
    header_rows = table.find_elements(By.CSS_SELECTOR, "thead tr")[0]  # Get the first row of the table header
    header_columns = header_rows.find_elements(By.CSS_SELECTOR, "th")  # Get each header cell
    for col in header_columns:
        headers.append(col.get_attribute('aria-label'))  # Get header name (e.g., 'Date', 'Home', etc.)

    headers.remove('Notes')  # Remove 'Notes' column, not needed for analysis

    # Rename headers for 'xG' fields to differentiate between Home and Away expected goals
    for i, header in enumerate(headers):
        if 'xG' in header:
            headers[i] = 'Home_xG' if i < 6 else 'Away_xG'  # First 'xG' is for Home team, second for Away team
    if 'Matchweek Number' in headers:
        headers.remove('Matchweek Number')  # Remove 'Matchweek Number' if present
    
    print('HEADERS: ' + str(headers))  # Print headers for debugging

    # Retrieve each row from the table body and process the data
    rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")

    for row in rows:
        try:
            data = []
            columns = row.find_elements(By.CSS_SELECTOR, "td")  # Get all columns in the row
                
            for column in columns:
                if column.text == "Match Report":
                    report_link = column.find_element(By.CSS_SELECTOR, "a")  # Extract link if column has "Match Report"
                    data.append([report_link.get_attribute("href")])  # Append link URL
                else:
                    data.append([column.text])  # Append regular cell text

            # Flatten nested lists into a single list
            flat_data = [item[0] for item in data]
            
            # Ensure the data length matches headers by adjusting as needed
            if len(flat_data) > len(headers):
                flat_data = flat_data[:-1]  # Remove extra items if present
                # print(flat_data)
                if len(flat_data) > len(headers):
                    headers.append('Matchweek Number')  # Re-add 'Matchweek Number' if data requires it
            
            # Create a DataFrame row for the scraped data with headers
            df = pd.DataFrame([flat_data], columns=headers)
            fixtures = df  # Assign DataFrame to fixtures
            fixtures['season'] = season  # Add season as a column
            fixtures['league'] = league  # Add league as a column

            fixtures['Home'] = DuplicateHandler.standardize_name(fixtures['Home'].iloc[0])
            fixtures['Away'] = DuplicateHandler.standardize_name(fixtures['Away'].iloc[0])
            # print(fixtures['Home'])
            # print(fixtures['Away'])
            
            # Generate a unique identifier based on date, home, and away team names
            fixtures['unique_id'] = fixtures['Date'] + "_" + fixtures['Home'] + "_" + fixtures['Away']
            
            # Insert each record into MongoDB, avoiding duplicates by using 'unique_id'
            for record in fixtures.to_dict("records"):
                if collection.find_one({'unique_id': record['unique_id']}) is None:
                    collection.insert_one(record)  # Insert new record
                    print(f"Match inserted: {record['unique_id']}")
                else:
                    # Update if a record with the same unique_id exists
                    collection.update_one(
                        {"unique_id": record["unique_id"]},
                        {"$set": record},
                        upsert=True  # Upsert ensures insert if not found
                    )
                    print(f"Duplicate found: {record['unique_id']}")

        except Exception as e:
            print(data)  # Print data if there's an error for debugging
            logging.error(f"Error processing {url}: {e}")  # Log any processing errors
            continue

    print(f"Data successfully inserted for {league} {season}.")  # Print completion message for league/season

def delete_invalid_matches(collection):
    """Delete matches with null dates, empty scores, or missing match reports from the collection.
    
    Args:
        collection: MongoDB collection containing match data
        
    Returns:
        None
        
    Raises:
        Exception: If deletion operation fails
    """
    try:
        result = collection.delete_many({
            "$or": [
                {"Date": None},
                {"Score": ""},
                {"Match Report": ""},
                {"Home": {"$exists": False}}
            ]
        })
        logging.info(f"Successfully deleted {result.deleted_count} matches that had null dates or empty scores")
    except Exception as e:
        logging.error(f"Error deleting future/null date matches: {e}")
        raise

# Main function to initialize MongoDB, generate URLs, and run scraping
def main():
    """Main execution function to scrape data and store it in MongoDB."""
    client = MongoClient('192.168.0.77', 27017)  # Connect to MongoDB server
    db = client['football_data']  # Database for storing football data
    collection = db['fixtures']  # Collection for fixtures
    
    all_urls = get_all_urls()  # Generate URLs for scraping
    # Only run delete_invalid_matches if 2024-2025 season exists in URLs
    if any(season == "2024-2025" for _, _, season in all_urls):
        delete_invalid_matches(collection)

    # Loop through each URL and scrape data
    for url, league, season in all_urls:
        logging.info(f"Processing: {league} {season}")
        try:    
            get_html_data(url, league, season, collection)  # Scrape data and store it
        except Exception as e:
            print('Error:' + str(e))  # Print error if scraping fails
            continue
        time.sleep(10)  # Add delay between requests for rate limiting

    driver.quit()  # Close the WebDriver after scraping
    logging.info("All data scraped and stored in MongoDB successfully.")  # Log final completion
    delete_duplicate_fixtures()

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()




