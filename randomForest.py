# Import necessary libraries
import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



# -----------------------------------------------
# Section 2: Load and Combine User Data
# -----------------------------------------------

# Define the data directory (replace with your actual directory path)
data_dir = 'data/processed/'

# Get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

# Initialize a list to hold data from each user
user_data_list = []

# Loop through each CSV file
for file in csv_files:
    # Extract the filename and username
    filename = os.path.basename(file)
    name_part = os.path.splitext(filename)[0]
    name_parts = name_part.split('_')
    username = '_'.join(name_parts[:-1])  # Assumes date is the last part

    # Read the user's CSV file
    df_user = pd.read_csv(file)

    # Add the username column
    df_user['username'] = username

    # Debugging: Print the filename and extracted username
    print(f'Processing file: {filename}')
    print(f'Extracted username: {username}')

    # Append the DataFrame to the list
    user_data_list.append(df_user)

# Combine all user data into a single DataFrame
df = pd.concat(user_data_list, ignore_index=True)

# Ensure 'username' column is present
assert 'username' in df.columns, "'username' column is missing in df"

# -----------------------------------------------
# Section 3: Data Cleaning
# -----------------------------------------------

# Check for missing values
print(df.isnull().sum())

# Decide on a strategy: drop or impute missing values
# Uncomment and replace 'essential_column1', 'essential_column2' with actual column names if needed
# df = df.dropna(subset=['essential_column1', 'essential_column2'])

# Drop duplicate rows
df = df.drop_duplicates()

# List of irrelevant columns to drop
irrelevant_columns = [
    'ts', 'platform', 'conn_country', 'ip_addr_decrypted', 'user_agent_decrypted',
    'spotify_track_uri', 'episode_name', 'episode_show_name', 'spotify_episode_uri',
    'reason_start', 'reason_end', 'offline_timestamp', 'track_href', 'analysis_url',
    'uri', 'id', 'track_name', 'album_name', 'album_release_date', 'duration_ms_y',
    'type', 'master_metadata_track_name', 'master_metadata_album_artist_name',
    'master_metadata_album_album_name'
]

# Drop irrelevant columns (ignore errors if columns are missing)
df = df.drop(columns=irrelevant_columns, errors='ignore')

# -----------------------------------------------
# Section 4: Genre Processing
# -----------------------------------------------

# 1. Handle missing values in 'artist_genres'
df['artist_genres'] = df['artist_genres'].fillna('')

# 2. Split genres into lists
df['artist_genres_list'] = df['artist_genres'].apply(
    lambda x: x.split(', ') if isinstance(x, str) and x else []
)

# 3. Verify that all entries in 'artist_genres_list' are lists
assert df['artist_genres_list'].apply(lambda x: isinstance(x, list)).all(), "Not all entries are lists."

# 4. Aggregate genres per user
user_genres = df.groupby('username')['artist_genres_list'].apply(
    lambda lists: [genre for sublist in lists for genre in sublist]
)

# 5. Apply MultiLabelBinarizer to create genre dummy variables
mlb = MultiLabelBinarizer()
genre_dummies = pd.DataFrame(
    mlb.fit_transform(user_genres),
    columns=mlb.classes_,
    index=user_genres.index
).reset_index()

# Debugging: Check the columns in genre_dummies
print("'username' in genre_dummies columns:", 'username' in genre_dummies.columns)
print("Columns in genre_dummies:", genre_dummies.columns.tolist())

# -----------------------------------------------
# Section 5: Key Variable Processing
# -----------------------------------------------

# a. Fill missing 'key' values and convert to integers
df['key'] = df['key'].fillna(-1).astype(int)

# b. Create dummy variables for 'key'
key_dummies = pd.get_dummies(df['key'], prefix='key')

# c. Concatenate 'username' and key_dummies
df_keys = pd.concat([df[['username']], key_dummies], axis=1)

# d. Aggregate 'key' dummies per user
user_keys = df_keys.groupby('username').mean().reset_index()

# Debugging: Check if 'username' is in user_keys columns
print("'username' in user_keys columns:", 'username' in user_keys.columns)

# -----------------------------------------------
# Section 6: Aggregate Numerical Features
# -----------------------------------------------

# Define the aggregation functions for numerical features
agg_functions = {
    'ms_played': 'sum',
    'skipped': 'mean',
    'shuffle': 'mean',
    # Add other features as needed
}

# Aggregate numerical features per user
user_numerical_agg = df.groupby('username').agg(agg_functions).reset_index()

# -----------------------------------------------
# Section 7: Prepare DataFrames for Merging
# -----------------------------------------------

# Function to prepare DataFrames
def prepare_dataframe(df, name):
    if 'username' not in df.columns:
        df.reset_index(inplace=True)
    df['username'] = df['username'].astype(str)
    df.columns = df.columns.str.strip()
    print(f"DataFrame '{name}' columns:", df.columns.tolist())
    print(f"Number of rows in '{name}':", len(df))
    return df

# Prepare each DataFrame
user_numerical_agg = prepare_dataframe(user_numerical_agg, 'user_numerical_agg')
genre_dummies = prepare_dataframe(genre_dummies, 'genre_dummies')
user_keys = prepare_dataframe(user_keys, 'user_keys')

# -----------------------------------------------
# Section 8: Data Consistency Checks
# -----------------------------------------------

# Check for NaN and duplicates in 'username' across DataFrames
for name, df_temp in [('user_numerical_agg', user_numerical_agg), 
                      ('genre_dummies', genre_dummies), 
                      ('user_keys', user_keys)]:
    print(f"NaN in 'username' of {name}:", df_temp['username'].isna().sum())
    print(f"Duplicates in 'username' of {name}:", df_temp['username'].duplicated().sum())

# -----------------------------------------------
# Section 9: Merge User-Level Data
# -----------------------------------------------

# Start with numerical aggregations
user_data = user_numerical_agg.copy()

# Merge genre dummy variables
user_data = pd.merge(user_data, genre_dummies, on='username', how='left')

# Merge key dummy variables
user_data = pd.merge(user_data, user_keys, on='username', how='left')

# Fill missing values and infer data types
user_data.fillna(0, inplace=True)
user_data = user_data.infer_objects()

# -----------------------------------------------
# Section 10: Merge with Personality Data
# -----------------------------------------------

# Load personality data
user_personality = pd.read_csv('data/results.csv')

# Debugging: Check the columns in user_personality
print("Columns in user_personality:", user_personality.columns.tolist())

# Rename the column if 'username' is not present
if 'username' not in user_personality.columns:
    # Assuming the username is stored in the 'name' column
    user_personality.rename(columns={'name': 'username'}, inplace=True)
    print("Renamed 'name' column to 'username'.")

# Verify that 'username' is now in the columns
print("'username' in user_personality columns:", 'username' in user_personality.columns)
print("Columns in user_personality after renaming:", user_personality.columns.tolist())

# Convert 'username' to string
user_personality['username'] = user_personality['username'].astype(str)

# Merge user data with personality data
user_data = pd.merge(user_data, user_personality, on='username', how='inner')

# -----------------------------------------------
# Section 11: Further Aggregations (Optional)
# -----------------------------------------------

# Define additional aggregation functions if needed
additional_agg_functions = {
    'ms_played': 'sum',
    'skipped': 'mean',
    'shuffle': 'mean',
    'offline': 'mean',
    'incognito_mode': 'mean',
    'track_popularity': ['mean', 'std'],
    'explicit': 'mean',
    'danceability': ['mean', 'std'],
    'energy': ['mean', 'std'],
    'loudness': ['mean', 'std'],
    'mode': 'mean',
    'speechiness': ['mean', 'std'],
    'acousticness': ['mean', 'std'],
    'instrumentalness': ['mean', 'std'],
    'liveness': ['mean', 'std'],
    'valence': ['mean', 'std'],
    'tempo': ['mean', 'std'],
    'time_signature': ['mean', 'std']
}

# Aggregate additional features per user
user_agg = df.groupby('username').agg(additional_agg_functions).reset_index()

# Flatten MultiIndex columns in user_agg
def flatten_columns(columns):
    flattened_columns = []
    for col in columns:
        if isinstance(col, tuple):
            # Handle the 'username' column
            if col[1] == '':
                flattened_columns.append(col[0])
            else:
                flattened_columns.append('_'.join(col).strip())
        else:
            flattened_columns.append(col)
    return flattened_columns

user_agg.columns = flatten_columns(user_agg.columns.values)

# Ensure 'username' is present in all DataFrames
assert 'username' in user_agg.columns, "'username' missing in user_agg"
assert 'username' in genre_dummies.columns, "'username' missing in genre_dummies"
assert 'username' in user_keys.columns, "'username' missing in user_keys"

# Convert 'username' to string in all DataFrames
user_agg['username'] = user_agg['username'].astype(str)
genre_dummies['username'] = genre_dummies['username'].astype(str)
user_keys['username'] = user_keys['username'].astype(str)

# Merge all DataFrames on 'username'
user_data = pd.merge(user_agg, genre_dummies, on='username', how='left')
user_data = pd.merge(user_data, user_keys, on='username', how='left')

# print(f"Number of users in dataset: {user_data['username'].nunique()}")
print(f"Total records: {len(user_data)}")
# print(user_data.head())

# Load the personality traits data
user_personality = pd.read_csv('data/results.csv')

# Ensure 'username' is in the columns
if 'username' not in user_personality.columns:
    user_personality.rename(columns={'name': 'username'}, inplace=True)

user_personality['username'] = user_personality['username'].astype(str)

# Merge user data with personality traits
user_data = pd.merge(user_data, user_personality, on='username', how='inner')

# Define target columns
target_columns = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

# Ensure these columns exist in user_data
missing_traits = [col for col in target_columns if col not in user_data.columns]
if missing_traits:
    print(f"Warning: The following personality trait columns are missing: {missing_traits}")

# Features are all columns except 'username' and target columns
feature_columns = [col for col in user_data.columns if col not in ['username'] + target_columns]

# Prepare feature matrix X and target matrix y
X = user_data[feature_columns]
y = user_data[target_columns]

# Handle any remaining missing values
X.fillna(0, inplace=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Initialize cross-validator
kf = KFold(n_splits=3, shuffle=True, random_state=42)
print(f"Number of folds: {kf.get_n_splits()}")

# Iterate over all personality traits
for trait in target_columns:
    print(f"\nTraining model to predict '{trait}'")
    y_trait = y[trait]
    
    # Initialize the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Use cross_val_predict to get predictions for all data points
    y_pred = cross_val_predict(rf_model, X, y_trait, cv=kf)
    
    # Evaluate the model
    mse = mean_squared_error(y_trait, y_pred)
    r2 = r2_score(y_trait, y_pred)
    
    print(f"Mean Squared Error for '{trait}': {mse}")
    print(f"R² Score for '{trait}': {r2}")
    
    # Save the predictions
    user_data[f'Predicted_{trait}'] = y_pred
    
    # Visualize Actual vs. Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_trait, y_pred, color='blue')
    plt.plot([y_trait.min(), y_trait.max()], [y_trait.min(), y_trait.max()], 'r--')
    plt.xlabel(f'Actual {trait}')
    plt.ylabel(f'Predicted {trait}')
    plt.title(f'Actual vs. Predicted {trait}')
    plt.grid(True)
    plt.show()
    
    # Check for missing predictions
    num_missing = pd.isnull(y_pred).sum()
    print(f"Number of missing predictions for '{trait}': {num_missing}")
    
    # Verify predictions
    print(f"Predictions for '{trait}': {y_pred}")
    print(f"Actual values for '{trait}': {y_trait.values}")