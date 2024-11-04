import requests
from bs4 import BeautifulSoup
import json
from pymongo import MongoClient
import pandas as pd
import re
import time

# Connect to MongoDB
client = MongoClient('mongodb://192.168.0.77:27017/')
db = client['football_data']
fixtures_collection = db['fixtures']
match_stats_collection = db['match_stats']
retry_after = 0

def fetch_match_data(url, unique_id):
    print("Getting match stats...")

    # Send a GET request to the URL
    response = requests.get(url)
    print(response.status_code)
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        print(f"429 Too Many Requests - Retrying in {retry_after} seconds...")
        time.sleep(int(retry_after))
        
        # response = requests.get(url)
    # response.raise_for_status()  # Raise an error if the request was unsuccessful
    else:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract team names from the match title
        match_title = soup.find('h1').text.strip()
        print(match_title)
        striped_title = match_title.split(" Match Report")
        teams = striped_title[0].split(" vs. ")
        # if len(teams) != 2:
        #     raise ValueError("Unable to extract team names from the match title")
        home_team, away_team = teams[0].strip(), teams[1].strip()

        # Generate a unique ID for the match (you can customize this)
        unique_id = unique_id  # Example: using the last part of the URL as unique ID

        # Initialize the data structure
        match_data = {
            "unique_id": unique_id,
            "match": striped_title,
            "url": url,
            "team_stats": {},
            "team_stats_extra": {}
        }
        
        # Extract the 'team_stats' table data
        team_stats_table = soup.find('div', {'id': 'team_stats'})
        if not team_stats_table:
            raise ValueError("Team stats table not found")

        # Loop through the rows in the table to extract stats
        rows = team_stats_table.find_all('tr')[1:]  # Skip the first header row
        for i in range(0, len(rows), 2):
            stat_name = rows[i].find('th').text.strip()
            home_stat = rows[i+1].find_all('td')[0].text.replace("\u00a0\u2014\u00a0"," ").strip()
            away_stat = rows[i+1].find_all('td')[1].text.replace("\u00a0\u2014\u00a0"," ").strip()

            match_data['team_stats'][stat_name] = {
                "home_value": home_stat,
                "away_value": away_stat
            }
    # Extract Team Stats Extra
        team_stats_extra_div = soup.find('div', id='team_stats_extra')
    
        if team_stats_extra_div:
    
            divs2 = team_stats_extra_div.text.replace(" ","")
            
            # Split the data into lines and remove empty lines
            lines = [line for line in divs2.split('\n') if line]

            # Team names are always the first line
            teams = lines[0].split()

            # Initialize a list to store rows for the DataFrame
            rows = []

            # Process every stat line starting from the second line
            for line in lines[1:]:
                # Regular expression to split into numbers and letters
                split_data = re.findall(r'(\d+|[A-Za-z]+)', line)
                # Split the line into the home value, stat name, and away value
                home_value = split_data[0]#''.join(filter(str.isdigit, line.split()[0]))
                try:
                    away_value = split_data[2] #''.join(filter(str.isdigit, line.split()[-1]))
                except Exception:
                    away_value = ''.join(filter(str.isdigit, line.split()[-1]))
                    continue
                stat_name = ''.join(filter(str.isalpha, line))
                if home_value == "":
                    continue
                else:
                    match_data['team_stats_extra'][stat_name] = {
                        "home_value": home_value,
                        "away_value": away_value
                    }
        return match_data

def insert_into_mongo(data):
    print("Inserting match...")
    collection = db['match_stats']
    # Upsert the data to avoid duplicates (update if exists, insert if not)
    collection.update_one(
        {'unique_id': data['unique_id']},  # Match based on the 'unique_id' field
        {'$set': data},  # Update the document with the new data
        upsert=True  # Insert the document if it does not exist
    )

def get_match_data():
    print("Get matches without stats...")
    # Get the list of unique IDs from match_stats
    match_stats_ids = match_stats_collection.distinct('unique_id')

    # Query the fixtures collection for documents where unique_id is not in match_stats
    unmatched_fixtures = fixtures_collection.find({
        'unique_id': {'$nin': match_stats_ids},
        'Score': {'$ne': ''} 
    })
    return unmatched_fixtures

def main():
    # Example match URL
    # url = 'https://fbref.com/en/matches/e4331e7c/Arminia-Freiburg-August-14-2021-Bundesliga'
    
    matches = get_match_data()
    print("Match selection successfull...")
    for match in matches:
      
        url = match['Match Report']
        unique_id = match['unique_id']
        
        try:
            # Fetch and structure the match data
            
            match_data = fetch_match_data(url, unique_id)

            # Insert the structured data into MongoDB
            insert_into_mongo(match_data)
            time.sleep(5)
        except Exception as e:
            print("Too much request, sleeping for 60 sec..." + str(e))
            print(unique_id + url)
            time.sleep(10)
            continue

if __name__ == "__main__":
    main()
