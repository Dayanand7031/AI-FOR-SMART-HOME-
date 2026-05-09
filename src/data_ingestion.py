import requests
import pandas as pd
import json
import os
from config import NASA_BASE_URL, NASA_PARAMS, RAW_DATA_DIR, DEFAULT_LAT, DEFAULT_LON, START_DATE, END_DATE

def fetch_nasa_data(lat=DEFAULT_LAT, lon=DEFAULT_LON, start=START_DATE, end=END_DATE, resolution="daily"):
    """
    Fetches energy and climate data from NASA POWER API.

    Args:
        lat (float): Latitude
        lon (float): Longitude
        start (str): Start date in YYYYMMDD format
        end (str): End date in YYYYMMDD format
        resolution (str): Temporal resolution ('daily', 'monthly', 'yearly')

    Returns:
        pd.DataFrame: Processed dataframe of fetched data
    """
    # Construct API URL
    # Format: /api/temporal/{resolution}/point?parameters=...&community=AG&longitude=...&latitude=...&start=...&end=...
    url = f"{NASA_BASE_URL}/{resolution}/point"

    params = {
        "parameters": ",".join(NASA_PARAMS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end
    }

    print(f"Fetching data from NASA POWER API for Lat: {lat}, Lon: {lon}...")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"NASA API request failed with status {response.status_code}: {response.text}")

    data = response.json()

    # Extract the parameters data from the JSON response
    # The JSON structure contains a 'properties' key which has 'parameter' mapping date -> value
    param_data = data['properties']['parameter']

    # Convert to DataFrame
    df = pd.DataFrame(param_data)

    # The index of the returned dataframe is the date string.
    # We reset it to have it as a column.
    df.index.name = 'date'
    df = df.reset_index()

    # Save raw JSON for audit/reproduction
    raw_json_path = RAW_DATA_DIR / f"nasa_{resolution}_{lat}_{lon}_{start}_{end}.json"
    with open(raw_json_path, 'w') as f:
        json.dump(data, f)

    # Save as CSV
    csv_path = RAW_DATA_DIR / f"nasa_{resolution}_{lat}_{lon}_{start}_{end}.csv"
    df.to_csv(csv_path, index=False)

    print(f"Data successfully saved to {csv_path}")
    return df

if __name__ == "__main__":
    # Test fetch for default location
    try:
        df_daily = fetch_nasa_data()
        print("\nSample Data:\n", df_daily.head())
    except Exception as e:
        print(f"Error: {e}")
