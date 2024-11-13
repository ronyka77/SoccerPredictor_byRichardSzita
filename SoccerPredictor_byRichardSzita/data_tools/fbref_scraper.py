import requests  # For making HTTP requests
from bs4 import BeautifulSoup  # For parsing HTML content
import json  # For handling JSON data
from pymongo import MongoClient  # For MongoDB interactions
import pandas as pd  # For data manipulation with DataFrames
import re  # For regex operations (used in text extraction)
import time  # For adding delays between requests

# Connect to MongoDB server
client = MongoClient('mongodb://192.168.0.77:27017/')
db = client['football_data']  # Access football_data database
fixtures_collection = db['fixtures']  # Collection for storing fixtures data
match_stats_collection = db['match_stats']  # Collection for match stats data
retry_after = 0  # Default retry time

def fetch_match_data(url, unique_id):
    """Fetch and structure match data from a given URL."""
    print("Getting match stats...")

    # Send a GET request to the match URL
    response = requests.get(url)
    print(response.status_code)  # Print the response status code
    if response.status_code == 429:  # Too many requests error
        retry_after = response.headers.get("Retry-After")  # Get retry time from headers
        print(f"429 Too Many Requests - Retrying in {retry_after} seconds...")
        time.sleep(int(retry_after))  # Wait specified time before retrying

    else:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the match title to identify teams
        match_title = soup.find('h1').text.strip()
        print(match_title)
        striped_title = match_title.split(" Match Report")
        teams = striped_title[0].split(" vs. ")
        
        # Extract home and away team names from title
        home_team, away_team = teams[0].strip(), teams[1].strip()

        # Initialize data structure for match data
        match_data = {
            "unique_id": unique_id,  # Set unique_id passed to the function
            "match": striped_title,  # Store match title
            "url": url,  # Store URL of match report
            "team_stats": {},  # Dictionary to store team stats
            "team_stats_extra": {}  # Dictionary to store additional stats
        }
        
        # Extract 'team_stats' table data
        team_stats_table = soup.find('div', {'id': 'team_stats'})  # Locate team stats section
        if not team_stats_table:  # Raise error if table not found
            raise ValueError("Team stats table not found")

        # Loop through rows in the table to extract individual stats
        rows = team_stats_table.find_all('tr')[1:]  # Skip the first header row
        for i in range(0, len(rows), 2):  # Iterate over rows in pairs
            stat_name = rows[i].find('th').text.strip()  # Stat name in the first row
            home_stat = rows[i+1].find_all('td')[0].text.replace("\u00a0\u2014\u00a0", " ").strip()  # Home stat value
            away_stat = rows[i+1].find_all('td')[1].text.replace("\u00a0\u2014\u00a0", " ").strip()  # Away stat value

            # Add the stats to match_data under 'team_stats'
            match_data['team_stats'][stat_name] = {
                "home_value": home_stat,
                "away_value": away_stat
            }

    # Extract additional team stats if available
        team_stats_extra_div = soup.find('div', id='team_stats_extra')
    
        if team_stats_extra_div:
            divs2 = team_stats_extra_div.text.replace(" ", "")  # Remove extra spaces from text

            # Split the data into lines and remove empty lines
            lines = [line for line in divs2.split('\n') if line]

            # Team names are always the first line
            teams = lines[0].split()

            # Process each line in 'team_stats_extra' to extract stats
            for line in lines[1:]:
                # Split the line into numbers and letters using regex
                split_data = re.findall(r'(\d+|[A-Za-z]+)', line)
                
                # Separate values into home, stat name, and away value
                home_value = split_data[0]  # Home team stat value
                try:
                    away_value = split_data[2]  # Away team stat value
                except Exception:
                    away_value = ''.join(filter(str.isdigit, line.split()[-1]))
                    continue
                stat_name = ''.join(filter(str.isalpha, line))  # Extract stat name
                if home_value == "":  # Skip empty stats
                    continue
                else:
                    # Add the extracted extra stats to match_data
                    match_data['team_stats_extra'][stat_name] = {
                        "home_value": home_value,
                        "away_value": away_value
                    }
        return match_data  # Return the complete match data dictionary

def insert_into_mongo(data):
    """Insert match data into MongoDB."""
    print("Inserting match...")
    collection = db['match_stats']  # Select match_stats collection
    
    # Upsert data (update if it exists, insert if it doesn't)
    collection.update_one(
        {'unique_id': data['unique_id']},  # Use unique_id to check for existing record
        {'$set': data},  # Update document with new data
        upsert=True  # Insert if no existing document is found
    )

def get_match_data():
    """Retrieve matches from fixtures that don't yet have stats in match_stats."""
    print("Get matches without stats...")
    
    # Get the list of unique IDs already in match_stats
    match_stats_ids = match_stats_collection.distinct('unique_id')

    # Query fixtures collection for matches not in match_stats and have a score
    unmatched_fixtures = fixtures_collection.find({
        'unique_id': {'$nin': match_stats_ids},  # Exclude IDs already in match_stats
        'Score': {'$ne': ''},  # Include only matches with a score
        'Match Report': {'$ne': ''}  # Include only matches with a non-blank URL
    })
    return unmatched_fixtures  # Return the cursor to the matching documents

def main():
    """Main function to fetch and insert match data."""
    matches = get_match_data()  # Retrieve matches needing stats
    print("Match selection successful...")
    for match in matches:
        url = match['Match Report']  # Get URL for match report
        unique_id = match['unique_id']  # Get match unique ID
        
        try:
            # Fetch and structure the match data
            # print(url)
            if url == "" or url == "nan" or url == "None" or url == None:
                print("No URL found")
                continue
            else:
                match_data = fetch_match_data(url, unique_id)
                # Insert the structured data into MongoDB
                insert_into_mongo(match_data)
                time.sleep(5)  # Delay between each match fetch
        except Exception as e:
            # Handle request overload error
            print("Too many requests, sleeping for 10 seconds..." + str(e))
            print(str(unique_id) + str(url))  # Convert both to strings
            time.sleep(5)  # Wait before retrying
            continue

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
