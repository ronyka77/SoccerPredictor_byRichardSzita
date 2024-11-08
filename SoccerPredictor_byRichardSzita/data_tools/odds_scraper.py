from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from pymongo import MongoClient
import logging
import pandas as pd
from datetime import datetime, timedelta
import random
import time

# Configure logging to save logs in a file named 'scraper.log' with timestamps and log levels
log_file_path = './scraper.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36"]

def initialize_mongodb():
    """Initialize MongoDB client and return a collection."""
    client = MongoClient("mongodb://192.168.0.77:27017/")  # Connect to MongoDB server
    db = client["football_data"]  # Use football_data database
    return db["odds_data"]  # Return odds_data collection for storing scraped data

# URL for the main site to scrape and constant league identifiers for URL generation
MAIN_URL = "https://www.oddsportal.com/"

# Dictionary defining leagues and their country identifiers for URL structure
LEAGUES = {
    "champions-league": "europe",
    "laliga": "spain",
    "laliga2": "spain",
    "bundesliga": "germany",
    "2-bundesliga": "germany",
    "3-liga": "germany",
    "serie-a": "italy",
    "serie-b": "italy",
    "ligue-1": "france",
    "ligue-2": "france",
    "premier-league": "england",
    "championship": "england",
    "league-one": "england"
}

# Define years for historical data scraping
YEARS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024"]

# Define column names for the DataFrame where scraped data will be stored
columns = ['Date', 'Time', 'Home', 'Away', 'Odd_Home', 'Odds_Draw', 'Odd_Away']

def generate_urls(leagues, years):
    """Generate URLs for both current and historical seasons based on league structure."""
    urls = []
    for league, country in leagues.items():
        # URL for the current active season (without a specific year)
        urls.append({
            "url": f"https://www.oddsportal.com/soccer/{country}/{league}/results/",
            "league": league
        })
        
        # URLs for past seasons with specified years
        for year in years:
            urls.append({
                "url": f"https://www.oddsportal.com/soccer/{country}/{league}-{year}/results/",
                "league": league
            })
    
    # Define a dictionary to organize special URLs by league
    special_urls_by_league = {
        "brazil": [
            "https://www.oddsportal.com/football/brazil/serie-b/results/",
            "https://www.oddsportal.com/football/brazil/serie-b-2023/results/",
            "https://www.oddsportal.com/football/brazil/serie-b-2022/results/",
            "https://www.oddsportal.com/football/brazil/serie-b-2021/results/",
            "https://www.oddsportal.com/football/brazil/serie-b-2020/results/",
            "https://www.oddsportal.com/football/brazil/serie-a-betano/results/",
            "https://www.oddsportal.com/football/brazil/serie-a-2023/results/",
            "https://www.oddsportal.com/football/brazil/serie-a-2022/results/",
            "https://www.oddsportal.com/football/brazil/serie-a-2021/results/",
            "https://www.oddsportal.com/football/brazil/serie-a-2020/results/"
        ],
        "argentina": [
            "https://www.oddsportal.com/football/argentina/torneo-betano/results/",
            "https://www.oddsportal.com/football/argentina/liga-profesional-2023/results/",
            "https://www.oddsportal.com/football/argentina/liga-profesional-2022/results/",
            "https://www.oddsportal.com/football/argentina/liga-profesional-2021/results/",
            "https://www.oddsportal.com/football/argentina/superliga-2019-2020/results/"
        ],
        "japan": [
            "https://www.oddsportal.com/football/japan/j1-league/results/",
            "https://www.oddsportal.com/football/japan/j1-league-2023/results/",
            "https://www.oddsportal.com/football/japan/j1-league-2022/results/",
            "https://www.oddsportal.com/football/japan/j1-league-2021/results/",
            "https://www.oddsportal.com/football/japan/j1-league-2020/results/"
        ]
    }

    # Loop through the dictionary and append each URL with its league to `urls`
    for league, urls_list in special_urls_by_league.items():
        for url in urls_list:
            urls.append({"url": url, "league": league})
    
    logging.info(f"urls: {urls}")
    return urls  # Return all generated URLs for scraping

urls = generate_urls(LEAGUES, YEARS)  # Generate URLs using LEAGUES and YEARS

def initialize_driver():
    """Initialize the Chrome WebDriver with custom options, including a random user agent and headless mode."""
    chrome_options = Options()
    chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")  # Random user agent
    chrome_options.add_argument("--headless")  # Run in headless mode for faster scraping without a visible browser
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Instantiate the WebDriver for use in scraping
driver = initialize_driver()

def approve_cookie():
    """Handle cookie consent on the main website."""
    driver.get(MAIN_URL)  # Open the main URL
    try:
        # Wait until the cookie consent button is clickable, then click to approve
        cookie_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, 'onetrust-accept-btn-handler'))
        )
        cookie_button.click()  # Click the button to approve cookies
    except Exception as e:
        logging.warning(f"Cookie approval failed: {e}")  # Log any issue with cookie handling

def get_pagination(url):
    """
    Retrieve the number of pages available for a given URL.
    This function assumes the presence of pagination links with data-number attributes.
    """
    driver.get(url)  # Load the page
    soup = BeautifulSoup(driver.page_source, "lxml")  # Parse the page source with BeautifulSoup
    pagination_links = soup.select('a[data-number]')  # Select pagination links based on data-number attribute
    # Return the last pagination number, or default to 1 if no pagination exists
    return int(pagination_links[-1].text) if pagination_links else 1

def load_page_and_scroll(url, page):
    """Load the page and scroll to the bottom to load all elements."""
    driver.get(f"{url}#/page/{page}")
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'eventRow')))
    time.sleep(2)  # Small delay for page stability
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)  # Wait for additional content to load
    return driver.find_elements(By.CLASS_NAME, 'eventRow')

def parse_date(event):
    """Extract and format date from event row."""
    try:
        dates = [dv.text for dv in event.find_all('div', class_='truncate')]
        if dates:
            date_text = dates[1] 
            date_part = date_text.split(' - ')[0].strip() if date_text.split(' - ')[0].strip() != '1' else dates[0].split(' - ')[0].strip()
            for date_format in ("%d %B %Y", "%d %b %Y", "%d %m %Y"):
                if "Yesterday" in date_part:  
                    datum = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                    logging.info(f"Date found: {datum}")
                    return datum
                elif "Today" in date_part:
                    datum = datetime.now().strftime("%Y-%m-%d")
                    logging.info(f"Date found: {datum}")
                    return datum
                else:
                    try:
                        datum = datetime.strptime(date_part, date_format).strftime("%Y-%m-%d")
                        logging.info(f"Date found: {datum}")
                        return datum
                    except ValueError:
                        logging.info(f"Wrong date format: {date_part}")
    except Exception as e:
        logging.error(f"Failed to parse date: {e}")
    # logging.warning(f"Date format not recognized for '{date_text}'. Skipping event.")
    return None  # Return None if date parsing fails

def parse_odds(event):
    """Extract odds information from event row."""
    try:
        odds = [p.text for p in event.find_all('p')]
        if len(odds) == 7:  # Adjust if necessary
            odds = odds[1:]
        return odds
    except Exception as e:
        logging.warning(f"Odds data is incomplete: {e}")
        return None

def process_event(event, league, actual_date):
    """Process a single event and return structured data if parsing is successful."""
    event_html = event.get_attribute('outerHTML')
    event = BeautifulSoup(event_html, "lxml")
    
    # Parse odds
    odds = parse_odds(event)
    if not odds or len(odds) < 6:
        return None  # Skip this event if odds data is incomplete

    # Extract home and away teams, generate unique_id
    home_team, away_team = odds[1], odds[2]
    unique_id = f"{actual_date}_{home_team}_{away_team}"
    
    # Create event data dictionary
    return {
        "unique_id": unique_id,
        "League": league,
        'Date': actual_date,
        'Time': odds[0],
        'Home': home_team,
        'Away': away_team,
        'Odd_Home': odds[3],
        'Odds_Draw': odds[4],
        'Odd_Away': odds[5]
    }
    
def scrape_page(url, page, league, retries=3, delay=5):
    """Scrapes event odds data from a specific page of a URL, with retries if data is empty."""
    for attempt in range(retries):
        try:
            # Use Selenium to locate event rows directly without parsing the entire page HTML
            event_elements = load_page_and_scroll(url, page)
            data = []  # Initialize list to store parsed data
            actual_date = None  # Variable to store parsed date
            event_number = 1  # Track event number for logging
            
            # Process each event row to extract date and odds information
            for event_element in event_elements:
                # logging.info(f"Event number: {event_number}")
                event_html = event_element.get_attribute('outerHTML')  # Get HTML of each event row
                event = BeautifulSoup(event_html, "lxml")  # Parse only this row’s HTML with BeautifulSoup
                
                # Parse date
                date = parse_date(event)
                if date:
                    actual_date = date  # Update actual date if found
                if actual_date is None:
                    logging.warning(f"Date format not recognized for '{date}'. Skipping event.")
                    continue  # Skip this event if date parsing fails 
                
                event_data = process_event(event_element, league, actual_date)
                if event_data:
                    data.append(event_data)
                    # logging.info(f"Processed event: {event_data['unique_id']}")
                else:
                    logging.info(f"Skipped event {event_number} due to parsing issues.")

                # Increment event number after processing each event
                event_number += 1
            
            # Return collected data if available
            if data:
                logging.info(f"Data found for {str(len(data))} matches")
                return data
            else:
                logging.warning(f"No data found on attempt {attempt + 1} for {url} page {page}")
                time.sleep(delay)  # Wait before retrying if data is missing
            
        except Exception as e:
            logging.error(f"Error on {url} page {page} attempt {attempt + 1}: {e}")
            print(f"Error on {url} page {page} attempt {attempt + 1}: {e}")
            time.sleep(delay)  # Wait before retrying if an error occurs
    
    # Log if all retries failed and return empty data
    logging.error(f"Failed to scrape data from {url} page {page} after {retries} attempts")
    return []

def validate_data(data):
    """Basic validation to ensure data has expected fields."""
    required_fields = {"Date", "Odd_Home"}
    return all(required_fields.issubset(event.keys()) for event in data)

def insert_to_mongodb(data):
    """Insert or update records in MongoDB based on unique Date, Home, and Away fields."""
    collection = initialize_mongodb()  # Access MongoDB collection
    if data:
        for record in data:
            # Define the filter to check if a matching record exists
            filter_query = {
                'Date': record['Date'],
                'Home': record['Home'],
                'Away': record['Away']
            }
            
            # Use the upsert option to insert if not found, or update if it exists
            update_query = {"$set": record}  # Updates fields with new data if record exists
            collection.update_one(filter_query, update_query, upsert=True)
        
        logging.info(f"Processed {len(data)} records with upsert (insert/update) into MongoDB")
    else:
        logging.warning("No data to insert.")

def scrape_data():
    """Main function to iterate through URLs and scrape data from each available page, with validation and retries."""
    data = []
    for entry in urls:
        url = entry["url"]
        league = entry["league"]
        pagination = get_pagination(url)
        for page in range(1, pagination + 1):
            event_data = scrape_page(url, page, league)
            if event_data:
                # Validate and insert to MongoDB if data is present
                if validate_data(event_data):
                    insert_to_mongodb(event_data)
                else:
                    logging.warning(f"Data validation failed. No data inserted into MongoDB. {event_data}")
                logging.info(f"Scraped data from {url} page {page}")
            else:
                logging.warning(f"Skipping empty data for {url} page {page}")

# Run cookie approval and start data scraping process
approve_cookie()
scrape_data()
driver.quit()  # Close the WebDriver once scraping is complete
