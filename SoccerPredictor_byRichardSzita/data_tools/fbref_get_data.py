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
        ('Champions-League', '8'),
        ('Premier League', '9'),
        ('Championship','10'),
        ('League One', '15'),
        ('La Liga', '12'),
        ('Segunda-Division', '17'),
        ('Serie A', '11'),
        ('Serie B', '18'),
        ('Ligue 1', '13'),
        ('Ligue 2', '60'),
        ('Bundesliga', '20'),
        ('2-Bundesliga','33'),
        ('3-Liga','59'),
        ('Serie A', '24'),  # Brazil Serie A
        ('Serie-B','38'),    # Brazil Serie B
        ('Liga-Profesional-Argentina', '21'),
        ('Eredivisie','23'),
        ('J1-League','25'),
        ('Allsvenskan','29'),
        ('Russian-Premier-League','30'),
        ('Primeira-Liga','32'),
        ('Ekstraklasa','36'),
        ('Superettan','48'),
        ('Eerste-Divisie','51')
    ]

    # Specific seasons to scrape data for; add additional seasons as needed
    seasons = ['2024-2025'] 
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
                print(flat_data)
                if len(flat_data) > len(headers):
                    headers.append('Matchweek Number')  # Re-add 'Matchweek Number' if data requires it
            
            # Create a DataFrame row for the scraped data with headers
            df = pd.DataFrame([flat_data], columns=headers)
            fixtures = df  # Assign DataFrame to fixtures
            fixtures['season'] = season  # Add season as a column
            fixtures['league'] = league  # Add league as a column
            
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

    print(f"Data successfully inserted for {league} {season}.")  # Print completion message for league/season

# Main function to initialize MongoDB, generate URLs, and run scraping
def main():
    """Main execution function to scrape data and store it in MongoDB."""
    client = MongoClient('192.168.0.77', 27017)  # Connect to MongoDB server
    db = client['football_data']  # Database for storing football data
    collection = db['fixtures']  # Collection for fixtures

    all_urls = get_all_urls()  # Generate URLs for scraping

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

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()




