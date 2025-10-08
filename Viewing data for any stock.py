import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta
import pandas as pd

ticker = "IXHL"

# Use current date instead of hardcoded future dates
end_date = datetime.now()
start_date = end_date - timedelta(days=1)  # Show last 7 days

print(f"Requesting data from {start_date.date()} to {end_date.date()}")

# 1. Fetch data for the available date range
try:
    # Try to get hourly data first
    data = yf.download(ticker, start=start_date, end=end_date, interval="15m")
    
    # # If no hourly data, try 30-minute intervals
    # if data.empty:
    #     print("No hourly data available, trying 30-minute intervals...")
    #     data = yf.download(ticker, start=start_date, end=end_date, interval="30m")
    
    # # If still no data, try 15-minute intervals
    # if data.empty:
    #     print("No 30-minute data available, trying 15-minute intervals...")
    #     data = yf.download(ticker, start=start_date, end=end_date, interval="15m")
    
    # # If still no intraday data, fall back to daily data
    # if data.empty:
    #     print("No intraday data available, trying daily data...")
    #     data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        
except Exception as e:
    print(f"Error downloading data: {e}")
    # Final fallback - get whatever data is available
    data = yf.download(ticker, period="7d")

# 2. Check if we have MultiIndex columns and flatten them if needed
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join(col).strip() for col in data.columns]
    
    # 3. Try to rename columns to OHLCV standard
    column_mapping = {}
    for col in data.columns:
        if 'Open' in col:
            column_mapping[col] = 'Open'
        elif 'High' in col:
            column_mapping[col] = 'High'
        elif 'Low' in col:
            column_mapping[col] = 'Low'
        elif 'Close' in col:
            column_mapping[col] = 'Close'
        elif 'Volume' in col:
            column_mapping[col] = 'Volume'
    
    data.rename(columns=column_mapping, inplace=True)
else:
    # If not MultiIndex, just ensure we have standard column names
    standard_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [col for col in standard_cols if col in data.columns]
    data = data[available_cols]

# 4. Get start and end dates for title
if not data.empty:
    start_date_str = data.index[0].strftime('%Y-%m-%d %H:%M')
    end_date_str = data.index[-1].strftime('%Y-%m-%d %H:%M')
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f"Data range: {start_date_str} to {end_date_str}")
    print(f"Current time: {current_time}")
    print(f"Number of data points: {len(data)}")
    print(f"Data frequency: {pd.infer_freq(data.index)}")

    # 5. Plot candlestick chart WITHOUT volume
    mpf.plot(
        data,
        type='candle',
        style='charles',
        title=f'{ticker} Price\n{start_date_str} to {end_date_str}\n(Current time: {current_time})',
        volume=False,
        figratio=(12, 6),
        ylabel='Price (USD)'
    )
else:
    print("No data available for the specified ticker and date range.")