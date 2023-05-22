import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Define the parameters
SYMBOL = "BTCUSDT"
INTERVAL = "1h"

# Define the folder path and file name
folder_path = "../data/raw"
file_name = SYMBOL+INTERVAL+".csv"

# Construct the file path
file_path = os.path.join(folder_path, file_name)

# Load the preprocessed data from the CSV file
df = pd.read_csv(file_path)

# Rename the columns for better clarity
df.columns = [
    'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
    'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
    'Taker Buy Quote Asset Volume', 'Unused'
]

# Convert timestamp columns to datetime format
df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')

# Convert other numeric columns to appropriate data types
numeric_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume',
    'Number of Trades', 'Taker Buy Base Asset Volume',
    'Taker Buy Quote Asset Volume'
]
df[numeric_columns] = df[numeric_columns].astype('float64')

# Set the 'Open Time' column as the DataFrame index
df.set_index('Open Time', inplace=True)

# Remove the last column (Unused)
df = df.drop('Unused', axis=1)

# Handling missing values ( maybe handle them really ? but binance pretty fiable source. One crypto concept is gap filling, idk if it fits here)
# Ways to handle are : deleting the row or calculate a median and replace the missing value
if (df.isna().sum().sum() == 0):
    print("No missing values detected")
else:
    print("Dataset corrupted, missing values found")

# --------------------------------
# --- FEATURE ENGINEERING STEPS --
# --------------------------------
# for the moment, we only use EMAs and RSI but it's a part to optimize to get better results
# need try and see process

# Calculate moving averages
window_sizes = [10, 20, 50]  # try and adjust

for window_size in window_sizes:
    column_name = f"SMA_{window_size}"
    df[column_name] = df['Close'].rolling(window=window_size).mean()

# Calculate RSI
window_size_rsi = 14  # try and adjust

df['Price Change'] = df['Close'].diff()
df['Positive Change'] = df['Price Change'].apply(lambda x: x if x > 0 else 0)
df['Negative Change'] = df['Price Change'].apply(
    lambda x: abs(x) if x < 0 else 0)
df['Avg Gain'] = df['Positive Change'].rolling(window=window_size_rsi).mean()
df['Avg Loss'] = df['Negative Change'].rolling(window=window_size_rsi).mean()
df['RS'] = df['Avg Gain'] / df['Avg Loss']
df['RSI'] = 100 - (100 / (1 + df['RS']))


# Define the destination folder path
folder_path = "../data/processed/full"

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Save the preprocessed data to a CSV file in the folder with custom name
file_path = os.path.join(folder_path, file_name)
df.to_csv(file_path)

print("Preprocessed file "+file_name+" saved successfully!")

# Splitting the Dataset into the Training set and Test set
# We'll take 75% of the set for training so we get a full month and the rest is reserved for testing

# Split the data into training and testing sets
df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)

# Save the training set
folder_path = "../data/processed/train"

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

file_path = os.path.join(folder_path, file_name)
df_train.to_csv(file_path)

print("Training preprocessed file saved successfully!")

# Save the testing set
folder_path = "../data/processed/test"

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

file_path = os.path.join(folder_path, file_name)
df_test.to_csv(file_path)

print("Testing preprocessed file saved successfully!")

# Normalizing data
# We normalize data to improve performance during learning process
# Check an article about "Euclidean distance" for further explainations

# Testing to see if StandardScaler can perform better ?
# MinMaxScaler supposed to be a better choice for data not normally distributed ( Gaussian distribution )

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Exclude 'Close Time' column from training set
df_train_features = df_train.drop(columns=['Close Time'])

# Fit the scaler to the training set
scaler.fit(df_train_features)

# Transform the training set
df_train_norm = pd.DataFrame(scaler.transform(
    df_train_features), columns=df_train_features.columns)

# Save the normalized training set
folder_path = "../data/processed/train"

file_path = os.path.join(folder_path, "normalized_" + file_name)
df_train_norm.to_csv(file_path)

print("Normalized training preprocessed file saved successfully!")

# Exclude 'Close Time' column from testing set
df_test_features = df_test.drop(columns=['Close Time'])

# Fit the scaler to the testing set
scaler.fit(df_test_features)

# Transform the testing set
df_test_norm = pd.DataFrame(scaler.transform(
    df_test_features), columns=df_test_features.columns)

# Save the normalized testing set
folder_path = "../data/processed/test"

file_path = os.path.join(folder_path, "normalized_" + file_name)
df_test_norm.to_csv(file_path)

print("Normalized testing preprocessed file saved successfully!")

# Since the dataset consists of candlestick data with a fixed time interval of one hour,
# the open and close times are no longer explicitly included in the normalized data.
