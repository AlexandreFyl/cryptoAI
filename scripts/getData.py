import os
import requests
import pandas as pd

# Define the API endpoint
API_ENDPOINT = "https://api.binance.com/api/v3/klines"

# Define the parameters
SYMBOL = "BTCUSDT"
INTERVAL = "1h"

# Make the request
response = requests.get(API_ENDPOINT, params={
    "symbol": SYMBOL,
    "interval": INTERVAL,
    "limit": 1000  # 1000/24 = 42 days max (api limitation)
})

# Check the response status code
if response.status_code == 200:
    # Convert the response to a Pandas DataFrame
    df = pd.DataFrame(response.json())

    # Print the DataFrame
    # print(df)

    # Define the folder path and file name
    folder_path = "../data/raw"
    file_name = SYMBOL+INTERVAL+".csv"

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the DataFrame to a CSV file in the folder (overwriting if it already exists)
    file_path = os.path.join(folder_path, file_name)
    df.to_csv(file_path, index=False)
    print("CSV file "+file_name+" saved successfully!")

else:
    print("Error fetching data:", response.status_code)
