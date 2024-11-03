import requests
from bs4 import BeautifulSoup
import pandas as pd
from pymongo import MongoClient
import time
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import logging
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize WebDriver once
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

def get_all_urls():
    leagues = [
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
        ('Serie A', '24'), #Brazil Serie A
        ('Serie-B','38'), #Brazil Serie B
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

    # leagues = [
    #     ('Eerste-Divisie','51'),
    #     ('Ekstraklasa','36'),
    #     ('Superettan','48'),
    #     ('Primeira-Liga','32'),
    #     ('Russian-Premier-League','30'),
    #     ('3-Liga','59')
       
    # ]   


    # seasons = ['2020-2021','2021-2022','2022-2023','2023-2024','2024-2025'] 
    seasons = ['2024-2025'] 
    urls = []
    
    for league, league_id in leagues:
        for season in seasons:
            league_name = league.replace(" ", "-")
            url = f'https://fbref.com/en/comps/{league_id}/{season}/schedule/{season}-{league_name}-Scores-and-Fixtures'
            urls.append((url, league, season))
    
    return urls

def get_html_data(url, league, season, collection):
    # driver = webdriver.Chrome()
    driver.get(url)
    # content = driver.page_source

    # Allow some time for the page to load
    time.sleep(3)

    table = driver.find_element(By.CSS_SELECTOR, "table")

    # if league=='Ligue 1' and season=='2020-2021' or league=='Bundesliga' and season=='2020-2021':
    #     headers = ['Round','Day', 'Date', 'Time', 'Home', 'Home_xG', 'Score', 'Away_xG', 'Away', 'Attendance', 'Venue', 'Referee', 'Match Report','Comment']
    
    # if league == 'League One':
    #     headers = ['Round', 'Day', 'Date', 'Time', 'Home', 'Score', 'Away', 'Attendance', 'Venue', 'Referee', 'Match Report','Comment']
    
    # else:
    #     headers = ['Day', 'Date', 'Time', 'Home', 'Home_xG', 'Score', 'Away_xG', 'Away', 'Attendance', 'Venue', 'Referee', 'Match Report','Comment']

    headers = []
    header_rows = table.find_elements(By.CSS_SELECTOR, "thead tr")[0]
    header_columns = header_rows.find_elements(By.CSS_SELECTOR, "th")
    for col in header_columns:
        headers.append(col.get_attribute('aria-label'))
    
    headers.remove('Notes')
    # Loop through the HEADERS list and rename 'xG: Expected Goals'
    for i, header in enumerate(headers):
        if 'xG' in header:  # Check if 'xG' is in the string
            if i < 6: # First occurrence
                headers[i] = 'Home_xG'
            else:  # Second occurrence
                headers[i] = 'Away_xG'
    # Remove 'Matchweek Number'
    if 'Matchweek Number' in headers:
        headers.remove('Matchweek Number')
    print('HEADERS: ' + str(headers))
    
    rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")

    for row in rows:
        try:
            data = []
            columns = row.find_elements(By.CSS_SELECTOR, "td")
              
                # print('Data length:' + str(len(columns)))
                # print('Header length:' + str(len(headers)))

                # except Exception as e:
                #     print('Exception found' + str(e))
                
            for column in columns:
                if column.text == "Match Report":
                    report_link = column.find_element(By.CSS_SELECTOR, "a")
                    data.append([report_link.get_attribute("href")])
                else:
                    data.append([column.text])

            # Reshape the data into a flat list
            flat_data = [item[0] for item in data]
            
            if len(flat_data) > len(headers):
                flat_data = flat_data[:-1]
                print(flat_data)
                if len(flat_data) > len(headers):
                    headers.append('Matchweek Number')
            
            df = pd.DataFrame([flat_data], columns=headers)
            fixtures = df #.drop(columns=['Matchweek Number'], errors='ignore')
            # fixtures = df[['Day', 'Date', 'Time', 'Home', 'Home_xG', 'Attendance', 'Score', 'Away_xG', 'Away', 'Venue', 'Referee', 'Match Report']]
            fixtures['season'] = season
            fixtures['league'] = league
            # Generate a unique identifier for the game
            fixtures['unique_id'] = fixtures['Date'] + "_" + fixtures['Home'] + "_" + fixtures['Away']
            # print(fixtures['unique_id'])
            for record in fixtures.to_dict("records"):
                # Check if the record already exists in the collection
                if collection.find_one({'unique_id': record['unique_id']}) is None:
                    collection.insert_one(record)
                    print(f"Match inserted: {record['unique_id']}")
                else:
                    # Check if a document with the same game_id exists
                    collection.update_one(
                        {"unique_id": record["unique_id"]},
                        {"$set": record},
                        upsert=True
                    )
                    print(f"Duplicate found: {record['unique_id']}")

        except Exception as e:
            print(data)
            logging.error(f"Error processing {url}: {e}")

    print(f"Data successfully inserted for {league} {season}.")
    # driver.quit()


def main():
    client = MongoClient('192.168.0.77', 27017)
    db = client['football_data']
    collection = db['fixtures']

    all_urls = get_all_urls()

    for url, league, season in all_urls:
        logging.info(f"Processing: {league} {season}")
        try:    
            get_html_data( url, league, season, collection)
        except Exception as e:
            print('Error:' + str(e))
            continue
        time.sleep(10)  # Rate limiting

    driver.quit()
    logging.info("All data scraped and stored in MongoDB successfully.")

if __name__ == "__main__":
    main()




