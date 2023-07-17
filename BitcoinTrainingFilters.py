
import pandas as pd
import numpy as np


def triple_exponential_moving_average(data, span):
    ema1 = data.ewm(span=span, adjust=False).mean()
    ema2 = ema1.ewm(span=span, adjust=False).mean()
    ema3 = ema2.ewm(span=span, adjust=False).mean()
    tema = 3 * ema1 - 3 * ema2 + ema3
    return tema


def calculate_rocs(data, intervals):
    for interval in intervals:
        data[f'ROC_{interval}'] = data['TEMA'].pct_change(
            periods=-1*interval) * 100.0
        data[f'ROC_{interval}'] = data[f'ROC_{interval}'].fillna(
            0)  # Fill NaN values with zero
    return data


def preprocess_bitcoin_ohlc_data(file_path, output_file_path):

    # Step 1: Read the data from the CSV file into a DataFrame with correct date parsing format
    # Read the CSV file without specifying parse_dates and index_col
    df = pd.read_csv(file_path, dayfirst=True)
    df = df.rename(columns={'Price': 'Close',
                   'Vol.': 'Volume', 'Change %': 'Close_Change_Pct'})
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%M/%d/%Y')

    # Set 'Date' column as the index for the DataFrame
    df.set_index('Date', inplace=True)

    # Step 2: Ensure appropriate data types and handle missing values (if any)
    df['Volume'] = df['Volume'].str.rstrip("K").astype(float)*1000
    df['Volume'] = df['Volume'].fillna(0)
    df['Volume'] = df['Volume'].astype('int64')
    df['Volume'] = df['Volume'] * df['Close']
    df['Close_Change_Pct'] = df['Close_Change_Pct'].str.rstrip(
        '%').astype(float)
    df['Close'] = df['Close'].str.replace(',', '').astype(float)
    df['Open'] = df['Open'].str.replace(',', '').astype(float)
    df['High'] = df['High'].str.replace(',', '').astype(float)
    df['Low'] = df['Low'].str.replace(',', '').astype(float)

    # Step 3: Smooth the raw price data using a Triple Exponential Moving Average (TEMA)
    # You can adjust the span as needed.
    df['TEMA'] = triple_exponential_moving_average(df['Close'], span=5)
    # Assuming you want to fill missing 'TEMA' values with zeros
    df['TEMA'] = df['TEMA'].fillna(0)

    # Step 4: Calculate ROCs for different time intervals
    roc_intervals = [2, 4, 8, 16, 32, 64]
    df = calculate_rocs(df, roc_intervals)

    # Step 5: Return the DataFrame with the additional ROC columns
    df.to_csv(output_file_path)

    return df


# Example usage:
data_file_path = 'data\Coinbase_Pro_BTC-USD_2019-04-22_2023-04-14.csv'
output_file_path = 'data\preprocessed_data.csv'  # Specify the output file path
preprocessed_data = preprocess_bitcoin_ohlc_data(
    data_file_path, output_file_path)
print(preprocessed_data.head())


# TODO: define filters that the environment can use to smooth raw data

class BitcoinTrainingFilters:
    def __init__(self) -> None:
        pass
