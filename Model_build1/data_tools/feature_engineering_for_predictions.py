import re
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
import dask.dataframe as dd


client = MongoClient('192.168.0.77', 27017)
db = client['football_data']
collection = db['aggregated_data']

le = LabelEncoder()


# # Function to apply multiprocessing
# def parallel_apply(df, cumsum_home_df, cumsum_away_df, column, num_cores=4):
#     # Prepare the data as needed
#     df_split = np.array_split(df, num_cores)
    
#     # Create a pool of workers
#     pool = mp.Pool(num_cores)
    
#     # Pass the partial function with the home and away cumulative dataframes
#     results = pool.starmap(apply_cumulative_sum_chunk, [(chunk, cumsum_home_df, cumsum_away_df, column) for chunk in df_split])
    
#     # Close the pool
#     pool.close()
#     pool.join()
    
#     # Concatenate the results back
#     return pd.concat(results, axis=0)

# # Wrapper function to handle chunks of the DataFrame
# def apply_cumulative_sum_chunk(df_chunk, cumsum_home_df, cumsum_away_df, column):
#     df_chunk[column] = df_chunk.apply(lambda row: calculate_cumulative_sum_parallel(row, cumsum_home_df, cumsum_away_df, column), axis=1)
#     return df_chunk

# # Function to be applied in parallel
# def calculate_cumulative_sum_parallel(row, cumsum_home_df, cumsum_away_df, column):
#     if column == 'home_match_cumcount':
#         return calculate_cumulative_sum(
#             cumsum_home_df, cumsum_away_df, row['running_id'], row['home_encoded'], row['Date'], 'home_count', 'away_count', 1
#         )
#     if column == 'away_match_cumcount':
#         return calculate_cumulative_sum(
#             cumsum_home_df, cumsum_away_df, row['running_id'], row['away_encoded'], row['Date'], 'home_count', 'away_count', 2
#         )
#     if column == 'home_win_cumsum':
#         return calculate_cumulative_sum(
#             cumsum_home_df, cumsum_away_df, row['running_id'], row['home_encoded'], row['Date'], 'home_win_cumsum', 'away_win_cumsum', 1
#         )
#     if column == 'away_win_cumsum':
#         return calculate_cumulative_sum(
#             cumsum_home_df, cumsum_away_df, row['running_id'], row['away_encoded'], row['Date'], 'home_win_cumsum', 'away_win_cumsum', 2
#         )

# Helper function to convert string values to float (removing % if necessary)
def convert_to_float(value):
    try:
        if isinstance(value, str):
            return float(value.replace('%', '').strip())
        return float(value)
    except (ValueError, TypeError):
        return np.nan  # Return NaN for easier detection of missing or invalid data

def extract_percentage_and_ratio(value,key,home_away):
    if value: 
        try:
            if ('Passing Accuracy' in key or 'Shots on Target' in key or 'Saves' in key) and home_away==2:
                # Regular expression to extract values like "61% 181 of 297"
                match = re.search(r'([\d\.]+%)\s*(\d+)\s*of\s*(\d+)', value)
                if match:
                    numerator = int(match.group(2)) if match.group(2) else 0
                    denominator = int(match.group(3)) if match.group(3) else 0
                    percentage = float(match.group(1).replace('%', '')) / 100 if match.group(1) else None
                    return numerator, denominator, percentage
            else:
                # Regular expression to extract values like "181 of 297 61%"
                match = re.search(r'(\d+)\s*of\s*(\d+)\s*([\d\.]+%)?', value)
                if match:
                    numerator = int(match.group(1))
                    denominator = int(match.group(2))
                    percentage = float(match.group(3).replace('%', '')) / 100 if match.group(3) else None
                    return numerator, denominator, percentage
        except Exception as e:
            print('Error with extracting percentage and ratio' + str(e))
            print(value)
            print(match)
    return 0, 0, 0

# Flatten the team_stats and team_stats_extra fields
def flatten_stats(row):
    flat_row = {}
    if isinstance(row, dict):
        for key, val in row.items():
            if isinstance(val, dict):
                # Handle special cases for complex strings (ratios and percentages)
                if 'Passing Accuracy' in key or 'Shots on Target' in key or 'Saves' in key:
                    numerator, denominator, percentage = extract_percentage_and_ratio(val.get('home_value', ''),key,1)
                    flat_row[f"home_{key}_number"] = numerator if numerator is not None else 0
                    flat_row[f"home_{key}_percentage"] = percentage if percentage is not None else 0

                    numerator, denominator, percentage = extract_percentage_and_ratio(val.get('away_value', ''),key,2)
                    flat_row[f"away_{key}_number"] = numerator if numerator is not None else 0
                    flat_row[f"away_{key}_percentage"] = percentage if percentage is not None else 0
                else:
                    # Simple numeric values
                    flat_row[f"home_{key}"] = convert_to_float(val.get('home_value', 0))
                    flat_row[f"away_{key}"] = convert_to_float(val.get('away_value', 0))
            else:
                flat_row[key] = val
    return flat_row

# Function to calculate rolling team form
def calculate_team_form(team, df, window=5):
    team_home_results = df[df['home_encoded'] == team]['home_points'].rolling(window=window, min_periods=1).apply(lambda x: sum(x == 1)).sum()
    team_away_results = df[df['away_encoded'] == team]['away_points'].rolling(window=window, min_periods=1).apply(lambda x: sum(x == -1)).sum()
    # print('team_home_results: ' + str(team_home_results))
    # print('team_away_results: ' + str(team_away_results))
    form = (team_home_results + team_away_results) / 10
    return form

# Function to calculate rolling team form
def calculate_rolling_values(team, home_df, away_df, league, season, year, week, home_column, away_column):
    try:
       
        # Filter based on team, league, and season from original DataFrame (not the rolled result)
        team_home_results_new = home_df[
            (home_df['home_encoded'] == team) & 
            (home_df['league_encoded'] == league) & 
            (home_df['season_encoded'] == season) &
            (home_df['week_of_year'] == week) &
            (home_df['year'] == year)
        ][home_column].sum()

        team_away_results_new = away_df[
            (away_df['away_encoded'] == team) & 
            (away_df['league_encoded'] == league) & 
            (away_df['season_encoded'] == season) &
            (away_df['week_of_year'] == week) &
            (away_df['year'] == year)
        ][away_column].sum()

        if team_home_results_new != 0 and team_away_results_new != 0:
            # Calculate rolling value
            rolling_value = (team_home_results_new + team_away_results_new) / 2
        else:
            rolling_value = team_home_results_new + team_away_results_new
            
        return rolling_value

    except Exception as e:
        print(f"Error calculating rolling values(probably column not in rolling_colums list): {e}")
        return 0

# Function to calculate rolling team form
def calculate_cumulative_sum( cumsum_home_df, cumsum_away_df, running_id, team, datum, home_column, away_column, home_away):
    try:
        cumsum_home_df['Date'] = pd.to_datetime(cumsum_home_df['Date'], errors='coerce')
        cumsum_away_df['Date'] = pd.to_datetime(cumsum_away_df['Date'], errors='coerce')
        datum = pd.to_datetime(datum, errors='coerce')
        if home_away == 1:
            
            # Filter based on team, league, and season from original DataFrame (not the rolled result)
            team_home_results_new = cumsum_home_df[
                (cumsum_home_df['running_id'] == running_id) 
            ][home_column].tail(1)

            team_away_results_new = cumsum_away_df[
                (cumsum_away_df['away_encoded'] == team) &
                (cumsum_away_df['Date'] <= datum)
            ][away_column].tail(1)

            team_home_count = cumsum_home_df[
                (cumsum_home_df['running_id'] == running_id) 
            ]['home_count'].tail(1)
            
            team_away_count = cumsum_away_df[
                    (cumsum_away_df['away_encoded'] == team) &
                (cumsum_away_df['Date'] <= datum)
            ]['away_count'].tail(1)
        elif home_away == 2:
                
            # Filter based on team, league, and season from original DataFrame (not the rolled result)
            team_home_results_new = cumsum_home_df[
                (cumsum_home_df['home_encoded'] == team) &
                (cumsum_home_df['Date'] <= datum)
            ][home_column].tail(1)

            team_away_results_new = cumsum_away_df[
                (cumsum_away_df['running_id'] == running_id) 
            ][away_column].tail(1)

            team_home_count = cumsum_home_df[
                (cumsum_home_df['home_encoded'] == team) &
                (cumsum_home_df['Date'] <= datum) 
            ]['home_count'].tail(1)
            
            team_away_count = cumsum_away_df[
                   (cumsum_away_df['running_id'] == running_id) 
            ]['away_count'].tail(1)
        
        # Check if the result is empty before accessing the value
        if not team_home_results_new.empty:
            team_home_results_new = team_home_results_new.values[0]
        else:
            team_home_results_new = 0  # or another default value
        if not team_away_results_new.empty:
            team_away_results_new = team_away_results_new.values[0]
        else:
            team_away_results_new = 0  # or another default value
        
        if not team_home_count.empty:
            team_home_count = team_home_count.values[0]
        else:
            team_home_count = 0  # or another default value
        
        if not team_away_count.empty:
            team_away_count = team_away_count.values[0]
        else:
            team_away_count = 0  # or another default value
                
        # if team_home_results_new != 0 and team_away_results_new != 0:
        #     # Calculate rolling value
        #     cumulative_value = (team_home_results_new + team_away_results_new) / (team_away_count + team_home_count)
        # else:
        cumulative_value = team_home_results_new + team_away_results_new
            
        return cumulative_value

    except Exception as e:
        print(f"running id: {str(running_id)} team:   {team} Date: {str(datum),home_column, away_column, str(home_away)}")
        print(f"Error calculating cumulative values(probably column not in cumulative_columns list): {e}")
        # print(cumsum_home_df.head())
        # print(cumsum_away_df.head())
        return 0

# Function to extract home and away goals from 'Score' column
def extract_goals(df):
    goals_split = df['Score'].str.split('–', expand=True)
    df['home_goals'] = pd.to_numeric(goals_split[0], errors='coerce')
    df['away_goals'] = pd.to_numeric(goals_split[1], errors='coerce')
    return df

# Function to drop and rename columns
def dataframe_drop_rename_columns(data, mode):
    rename_columns = {
        'home_Possession': 'home_possession_accuracy',
        'away_Possession': 'away_possession_accuracy',
        'home_Passing Accuracy_number': 'home_passes',
        'home_Passing Accuracy_percentage': 'home_passing_accuracy',
        'away_Passing Accuracy_number': 'away_passes',
        'away_Passing Accuracy_percentage': 'away_passing_accuracy',
        'home_Shots on Target_number': 'home_shots_on_target',
        'home_Shots on Target_percentage': 'home_shoting_accuracy',
        'away_Shots on Target_number': 'away_shots_on_target',
        'away_Shots on Target_percentage': 'away_shoting_accuracy',
        'home_Saves_number': 'home_saves',
        'home_Saves_percentage': 'home_saves_accuracy',
        'away_Saves_number': 'away_saves',
        'away_Saves_percentage': 'away_saves_accuracy',
        'home_Cards': 'home_cards',
        'away_Cards': 'away_cards'
    }
    columns_to_keep = [
        'home_Fouls', 'away_Fouls', 'home_Corners', 'away_Corners', 'home_Crosses', 'away_Crosses',
        'home_Touches', 'away_Touches', 'home_Tackles', 'away_Tackles', 'home_Interceptions', 'away_Interceptions',
        'home_AerialsWon', 'away_AerialsWon', 'home_Clearances', 'away_Clearances', 'home_Offsides', 'away_Offsides',
        'home_GoalKicks', 'away_GoalKicks', 'home_ThrowIns', 'away_ThrowIns', 'home_LongBalls', 'away_LongBalls'
    ]

    if mode == 1:
        data.rename(columns=rename_columns, inplace=True)
    if mode == 2:
        data = data[columns_to_keep]
        data.columns = data.columns.str.lower()

    return data

# Function to calculate points based on match outcome
def calculate_points(match_outcome):
    if match_outcome == 1:
        return 3
    elif match_outcome == 0:
        return 1
    else:
        return 0

# Parallelized apply across multiple workers
def calculate_sum_dask(row, cumsum_home_df, cumsum_away_df, column):
    # Your cumulative sum logic here
    if column == 'home_match_cumcount':
        return calculate_cumulative_sum(
            cumsum_home_df, cumsum_away_df, row['running_id'], row['home_encoded'], row['Date'], 'home_count', 'away_count', 1
        )
    if column == 'away_match_cumcount':
        return calculate_cumulative_sum(
            cumsum_home_df, cumsum_away_df, row['running_id'], row['away_encoded'], row['Date'], 'home_count', 'away_count', 2
        )
    if column == 'home_win_cumsum':
        return calculate_cumulative_sum(
            cumsum_home_df, cumsum_away_df, row['running_id'], row['home_encoded'], row['Date'], 'home_win_cumsum', 'away_win_cumsum', 1
        )
    if column == 'away_win_cumsum':
        return calculate_cumulative_sum(
            cumsum_home_df, cumsum_away_df, row['running_id'], row['away_encoded'], row['Date'], 'home_win_cumsum', 'away_win_cumsum', 2
        )
    pass

# Main function for data loading and cleaning
def load_and_prepare_data():
    print("Collecting data...")
   
    data = pd.DataFrame(list(collection.find()))
    
    data = data.dropna(subset=['match_outcome'])
    data = data.dropna(subset=['match'])
    print("Data Collected, Start Cleaning and Feature Engineering...")

    # Drop irrelevant columns
    data = data.drop(columns=['prediction_outcome', 'model_prediction'], errors='ignore')

    # Type conversions and extracting date components
    data['home_advantage'] = 1
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['year'] = data['Date'].dt.year
    data['month'] = data['Date'].dt.month
    data['day_of_month'] = data['Date'].dt.day
    data['day_of_week'] = data['Date'].dt.dayofweek
    data['week_of_year'] = data['Date'].dt.isocalendar().week
    
    # Label encoding for categorical variables
    data['venue_encoded'] = le.fit_transform(data['Venue'])
    data['season_encoded'] = le.fit_transform(data['season'])
    data['league_encoded'] = le.fit_transform(data['league'])
    data['home_encoded'] = le.fit_transform(data['Home'])
    data['away_encoded'] = le.fit_transform(data['Away'])

    # Flatten nested dictionaries
    team_stats_flat = data['team_stats'].apply(flatten_stats).apply(pd.Series)
    team_stats_extra_flat = data['team_stats_extra'].apply(flatten_stats).apply(pd.Series)
    
    team_stats_flat = dataframe_drop_rename_columns(team_stats_flat, 1)
    team_stats_extra_flat = dataframe_drop_rename_columns(team_stats_extra_flat, 2)
    
    data = pd.concat([data.drop(columns=['team_stats', 'team_stats_extra']), team_stats_flat, team_stats_extra_flat], axis=1)
    print('Data flatten...')
    
    # First, sort the data based on the year, month, and day_of_month for each group.
    data = data.sort_values(by=['running_id'])

    # Numeric conversion and NaN handling
    data['Home_xG'] = pd.to_numeric(data['Home_xG'], errors='coerce').fillna(0)
    data['Away_xG'] = pd.to_numeric(data['Away_xG'], errors='coerce').fillna(0)
    data['home_cards'] = pd.to_numeric(data['home_cards'], errors='coerce').fillna(0)
    data['away_cards'] = pd.to_numeric(data['away_cards'], errors='coerce').fillna(0)

    # Extract goals from 'Score'
    data = extract_goals(data)

    print('Start possession and shooting...')
    # Feature engineering for possession and shooting
    data['total_fouls'] = data['home_fouls'] + data['away_fouls']
    data['home_possession_shooting'] = data['home_possession_accuracy'] * data['home_shoting_accuracy']
    data['away_possession_shooting'] = data['away_possession_accuracy'] * data['away_shoting_accuracy']

    # Referee Stats
    if 'Referee' in data.columns:
        data['referee_encoded'] = le.fit_transform(data['Referee'])
    else:
        raise KeyError("Error: 'Referee' column not found in DataFrame.")

    referee_stats = data.groupby('referee_encoded')['total_fouls'].mean()
    data['referee_foul_rate'] = data['referee_encoded'].map(referee_stats)

    # Additional feature engineering
    print('Base feature engineering done, start additional features...')
    data['home_possession_diff'] = data['home_possession_accuracy'] - data['away_possession_accuracy']
    data['pass_accuracy_diff'] = data['home_passing_accuracy'] - data['away_passing_accuracy']
    data['home_shots_on_target_ratio'] = data['home_shots_on_target'] / data['away_shots_on_target']
    data['away_shots_on_target_ratio'] = data['away_shots_on_target'] / data['home_shots_on_target']
    data['home_save_efficiency'] = data['home_saves'] / data['away_shots_on_target']
    data['away_save_efficiency'] = data['away_saves'] / data['home_shots_on_target']

    # Defensive activity
    data['home_defensive_activity'] = data['home_tackles'] + data['home_interceptions'] + data['home_clearances']
    data['away_defensive_activity'] = data['away_tackles'] + data['away_interceptions'] + data['away_clearances']

    # Set-piece threat and foul impact
    data['home_set_piece_threat'] = data['home_corners'] + data['home_crosses']
    data['away_set_piece_threat'] = data['away_corners'] + data['away_crosses']

    # Outcome calculation
    print('Outcome calculation')
    data['home_win'] = data['match_outcome'].apply(lambda x: 1 if x == 1 else 0)
    data['away_win'] = data['match_outcome'].apply(lambda x: 1 if x == -1 else 0)
    data['draw'] = data['match_outcome'].apply(lambda x: 1 if x == 0 else 0)
    
    # Points 
    print('Points and Form calculation')
    data['home_points'] = data['home_win'].apply(lambda x: 3 if x == 1 else 0) + data['draw']
    data['away_points'] = data['away_win'].apply(lambda x: 3 if x == 1 else 0) + data['draw']
    
    # data['home_form'] = data.apply(lambda row: calculate_team_form(row['home_encoded'], data), axis=1)
    # data['away_form'] = data.apply(lambda row: calculate_team_form(row['away_encoded'], data), axis=1)
    
    data['home_goal_difference'] = data['home_goals'] - data['away_goals']
    data['away_goal_difference'] = data['away_goals'] - data['home_goals']

    # Rolling averages and cumulative sums
    print('Start calculating Cumulative values...')
    data['home_points_cumulative'] = data.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_points'].cumsum()
    data['away_points_cumulative'] = data.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_points'].cumsum()
    data['home_goal_diff_cumulative'] = data.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_goal_difference'].cumsum()
    data['away_goal_diff_cumulative'] = data.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_goal_difference'].cumsum()
    return data 
    
def add_cumulative_sums(dataframe):
    
    cumulative_columns = ['running_id','Date','home_encoded','away_encoded', 'season_encoded', 'league_encoded','home_saves','away_saves',
                          'home_shots_on_target','away_shots_on_target','home_possession_accuracy','away_possession_accuracy','home_fouls','away_fouls'
                          ,'home_points','away_points','home_goal_difference','away_goal_difference','home_win','away_win','draw']
    
    dataframe = dataframe.replace([np.inf, -np.inf], 0)
    
    cumsum_home_df = dataframe[cumulative_columns]
    # Ensure you are working with a copy of the DataFrame, not a slice
    cumsum_home_df = cumsum_home_df.copy()
    
  
    cumsum_home_df.loc[:,'home_points_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_points'].cumsum()
    cumsum_home_df.loc[:,'home_saves_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_saves'].cumsum()
    cumsum_home_df.loc[:,'home_shots_on_target_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_shots_on_target'].cumsum()
    cumsum_home_df.loc[:,'home_possession_accuracy_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_possession_accuracy'].cumsum()
    cumsum_home_df.loc[:,'home_fouls_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_fouls'].cumsum()
    cumsum_home_df.loc[:,'home_goal_difference_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_goal_difference'].cumsum()
    cumsum_home_df.loc[:,'home_win_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_win'].cumsum()
    cumsum_home_df.loc[:,'draw_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['draw'].cumsum()
    cumsum_home_df.loc[:,'home_count'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['running_id'].cumcount()+1
    
    cumsum_away_df = dataframe[cumulative_columns]
    # Ensure you are working with a copy of the DataFrame, not a slice
    cumsum_away_df = cumsum_away_df.copy()
    
    cumsum_away_df.loc[:, 'away_points_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_points'].cumsum()
    cumsum_away_df.loc[:, 'away_fouls_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_fouls'].cumsum()
    cumsum_away_df.loc[:, 'away_saves_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_saves'].cumsum()
    cumsum_away_df.loc[:, 'away_shots_on_target_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_shots_on_target'].cumsum()
    cumsum_away_df.loc[:, 'away_possession_accuracy_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_possession_accuracy'].cumsum()
    cumsum_away_df.loc[:, 'away_goal_difference_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_goal_difference'].cumsum()
    cumsum_away_df.loc[:, 'away_win_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_win'].cumsum()
    cumsum_away_df.loc[:, 'draw_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['draw'].cumsum()
    cumsum_away_df.loc[:, 'away_count'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['running_id'].cumcount()+1
    
    cumsum_home_df = cumsum_home_df.sort_values(by=['running_id'])
    cumsum_away_df = cumsum_away_df.sort_values(by=['running_id'])
    cumsum_home_df.to_excel('cumsum_home_df.xlsx')
    cumsum_away_df.to_excel('cumsum_away_df.xlsx')
    
    print("CUMSUM Dataframes ready, start calculating values...")
    dataframe['home_cumcount'] = dataframe.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['running_id'].cumcount() + 1
    dataframe['away_cumcount'] = dataframe.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['running_id'].cumcount() + 1
    
    return dataframe

def add_rolling_columns(dataframe):
      # Create dataframeframes for rolling values
    print('Start calculating Rolling values...')
    rolling_columns = ['running_id','year','week_of_year','home_encoded','away_encoded', 'season_encoded', 'league_encoded','home_goals', 'away_goals', 'Home_xG', 'Away_xG', 'home_goal_difference', 'away_goal_difference', 
                    'home_saves_accuracy', 'away_saves_accuracy', 'home_shoting_accuracy', 'away_shoting_accuracy', 
                    'home_shots_on_target', 'away_shots_on_target','home_points','away_points']

    rolling_home_df = dataframe[rolling_columns].sort_values(by=['home_encoded', 'season_encoded', 'league_encoded', 'year', 'week_of_year'])
    rolling_away_df = dataframe[rolling_columns].sort_values(by=['away_encoded', 'season_encoded', 'league_encoded', 'year', 'week_of_year'])
    
    # Rolling sum for home dataframe
    home_df = rolling_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded','year','week_of_year']) \
                .rolling(window=4, min_periods=1).sum().reset_index()

    # Rolling sum for away dataframe 
    away_df = rolling_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded','year','week_of_year']) \
                .rolling(window=4, min_periods=1).sum().reset_index()

    print('Rolling dataframeframes ready, calculating values...')
    # home_df.to_excel('./Model_build1/home_rolling_df.xlsx')
    # away_df.to_excel('./Model_build1/away_rolling_df.xlsx')      
       
    # Calculate rolling values 
    print('Rolling values: home_goals_rolling_avg')
    dataframe['home_goals_rolling_avg'] = dataframe.apply(lambda row: 
        calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                 row['league_encoded'],row['season_encoded'],
                                 row['year'],row['week_of_year'],
                                 'home_goals','away_goals'), axis=1).ffill()
    
    print('Rolling values: away_goals_rolling_avg')
    dataframe['away_goals_rolling_avg'] = dataframe.apply(lambda row: 
        calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                row['league_encoded'],row['season_encoded'],
                                 row['year'],row['week_of_year'],
                                 'home_goals','away_goals'), axis=1).ffill()
     
    print('Rolling values: home_xG_rolling_avg')   
    dataframe['home_xG_rolling_avg'] = dataframe.apply(lambda row: 
        calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                row['league_encoded'],row['season_encoded'],
                                row['year'],row['week_of_year'],
                                'Home_xG','Away_xG'), axis=1).ffill()
    
    print('Rolling values: away_xG_rolling_avg')
    dataframe['away_xG_rolling_avg'] = dataframe.apply(lambda row: 
        calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                row['league_encoded'],row['season_encoded'],
                                row['year'],row['week_of_year'],
                                'Home_xG','Away_xG'), axis=1).ffill()
    
    print('Rolling values: home_goal_difference_rolling_avg')   
    dataframe['home_goal_difference_rolling_avg'] = dataframe.apply(lambda row: 
        calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                row['league_encoded'],row['season_encoded'],
                                row['year'],row['week_of_year'],
                                'home_goal_difference','away_goal_difference'), axis=1).ffill()
    
    print('Rolling values: away_goal_difference_rolling_avg')
    dataframe['away_goal_difference_rolling_avg'] = dataframe.apply(lambda row: 
        calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                row['league_encoded'],row['season_encoded'],
                                row['year'],row['week_of_year'],
                                'home_goal_difference','away_goal_difference'), axis=1).ffill()
    
    print('Rolling values: home_saves_accuracy_rolling_avg')
    dataframe['home_saves_accuracy_rolling_avg'] = dataframe.apply(lambda row: 
        calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                row['league_encoded'],row['season_encoded'],
                                row['year'],row['week_of_year'],
                                'home_saves_accuracy','away_saves_accuracy'), axis=1).ffill()
    
    print('Rolling values: away_saves_accuracy_rolling_avg')
    dataframe['away_saves_accuracy_rolling_avg'] = dataframe.apply(lambda row: 
        calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                row['league_encoded'],row['season_encoded'],
                                row['year'],row['week_of_year'],
                                'home_saves_accuracy','away_saves_accuracy'), axis=1).ffill()
    
    print('Rolling values: home_shots_on_target_rolling_avg')
    dataframe['home_shots_on_target_rolling_avg'] = dataframe.apply(lambda row: 
        calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                row['league_encoded'],row['season_encoded'],
                                row['year'],row['week_of_year'],
                                'home_shots_on_target','away_shots_on_target'), axis=1).ffill()
    
    print('Rolling values: away_shots_on_target_rolling_avg')
    dataframe['away_shots_on_target_rolling_avg'] = dataframe.apply(lambda row: 
        calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                row['league_encoded'],row['season_encoded'],
                                row['year'],row['week_of_year'],
                                'home_shots_on_target','away_shots_on_target'), axis=1).ffill()
    
    print('Rolling values: home_shots_on_target_accuracy_rolling_avg')
    dataframe['home_shots_on_target_accuracy_rolling_avg'] = dataframe.apply(lambda row: 
        calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                row['league_encoded'],row['season_encoded'],
                                row['year'],row['week_of_year'],
                                'home_shoting_accuracy','away_shoting_accuracy'), axis=1).ffill()
    
    print('Rolling values: away_shots_on_target_accuracy_rolling_avg')
    dataframe['away_shots_on_target_accuracy_rolling_avg'] = dataframe.apply(lambda row: 
        calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                row['league_encoded'],row['season_encoded'],
                                row['year'],row['week_of_year'],
                                'home_shoting_accuracy','away_shoting_accuracy'), axis=1).ffill()
    
    print('Rolling values: home_points_rolling_avg')
    dataframe['home_points_rolling_avg'] = dataframe.apply(lambda row: 
        calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                row['league_encoded'],row['season_encoded'],
                                row['year'],row['week_of_year'],
                                'home_points','away_points'), axis=1).ffill()
    
    print('Rolling values: away_points_rolling_avg')
    dataframe['away_points_rolling_avg'] = dataframe.apply(lambda row: 
        calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                row['league_encoded'],row['season_encoded'],
                                row['year'],row['week_of_year'],
                                'home_points','away_points'), axis=1).ffill()
  
    
    # League position and strength index
    dataframe['home_league_position'] = dataframe.groupby(['season_encoded', 'league_encoded'])['home_points_cumulative'] \
        .transform(lambda x: x.rank(method='dense', ascending=False))
    dataframe['away_league_position'] = dataframe.groupby(['season_encoded', 'league_encoded'])['away_points_cumulative'] \
        .transform(lambda x: x.rank(method='dense', ascending=False))

    print("New features added and dataframeset saved!")

    # Drop unnecessary columns
    dataframe = dataframe.drop(columns=['_id_match','_id_odds','season', 'Venue', 'league',
                              'Date', 'Day', 'Referee', 'match', 'Attendance', 'Match Report', 'Score', 'url', 'Time'], errors='ignore')
    dataframe = dataframe.replace([np.inf, -np.inf], 0)

    print("dataframe Cleaning successful...")
    return dataframe
    
# Execute the function
base_dataframe = load_and_prepare_data()

cumsummed_dataframe = add_cumulative_sums(base_dataframe)

rolling_dataframe = add_rolling_columns(cumsummed_dataframe)
rolling_dataframe.to_excel('./Model_build1/data/model_data_prediction.xlsx')

# Define the new order of columns
# new_column_order = [
#     # 1. Identifiers & Metadata
#     'running_id',
    
#     # 2. Match Date & Time
#     'year', 'month', 'day_of_month', 'day_of_week', 'week_of_year',
    
#      # 3. Team info
#     'Home', 'Away','home_encoded', 'away_encoded',
#     'venue_encoded', 'season_encoded', 'league_encoded', 
#     # 'home_form', 'away_form',
#     'home_team_average_points', 'away_team_average_points',
    
#     # 4. Match Outcome
#     'match_outcome',
#     'home_points','away_points','home_points_cumulative', 'away_points_cumulative', 'home_points_rolling_avg', 'away_points_rolling_avg',
    
#     # 5. Goals
#     'home_goals', 'away_goals', 'home_goals_rolling_avg', 'away_goals_rolling_avg',
#     'home_goal_difference', 'away_goal_difference', 'home_goal_diff_cumulative', 'away_goal_diff_cumulative','home_goal_difference_rolling_avg', 'away_goal_difference_rolling_avg',
#     'Home_xG', 'Away_xG', 'home_xG_rolling_avg', 'away_xG_rolling_avg',
    
#     # 6. Win/Draw Rates
#     'home_win_rate', 'away_win_rate', 'home_draw_rate', 'away_draw_rate', 'draw', 'home_win', 'away_win',
    
#     # CUMSUM  VALUES
#     'home_cumcount','away_cumcount','home_match_cumcount','away_match_cumcount','home_win_cumsum','away_win_cumsum','home_draw_cumsum','away_draw_cumsum',
    
#     # 7. Team Strength & Position
#     'home_strength_index', 'away_strength_index', 'home_league_position', 'away_league_position',
    
#     # 8. Passing & Possession
#     'home_possession_accuracy', 'away_possession_accuracy', 'home_passing_accuracy', 'away_passing_accuracy',
#     'home_passes', 'away_passes', 
#     # 'home_possession_diff', 'away_possession_diff',
#     'home_possession_shooting', 'away_possession_shooting',
    
#     # 9. Shots, Saves, and Set Pieces
#     'home_shots_on_target', 'away_shots_on_target', 'home_shots_on_target_rolling_avg','away_shots_on_target_rolling_avg',
#     'home_shoting_accuracy', 'away_shoting_accuracy', 'home_shots_on_target_accuracy_rolling_avg','away_shots_on_target_accuracy_rolling_avg',
#     'home_shots_on_target_ratio', 'away_shots_on_target_ratio',
#     'home_saves', 'away_saves', 'home_saves_accuracy', 'away_saves_accuracy', 'home_saves_accuracy_rolling_avg', 'away_saves_accuracy_rolling_avg',
#     'home_set_piece_threat', 'away_set_piece_threat',
    
#     # 10. Defensive Stats
#     'home_clearances', 'away_clearances', 'home_interceptions', 'away_interceptions',
#     'home_defensive_activity', 'away_defensive_activity',
    
#     # 11. Fouls, Cards, and Discipline
#     'home_fouls', 'away_fouls', 'home_cards', 'away_cards', 'total_fouls',
#     # 'home_foul_impact', 'away_foul_impact',
#     'home_offsides', 'away_offsides',
    
#     # 12. Referee
#     'referee_encoded', 'referee_foul_rate',
       
# ]

# # Reorder the dataframe columns
# final_dataframe = rolling_dataframe[new_column_order]

# print('New column order applied...')
# # Display the first few rows to verify the reordering
# final_dataframe.to_excel('./Model_build1/data/model_data_prediction.xlsx')

print('data exported to xlsx...')


