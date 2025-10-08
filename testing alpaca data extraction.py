from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime, timedelta, timezone

client = StockHistoricalDataClient("PKOGMRZV5WHDQY829VUB", "o2OgCIG79VNOx6Jp5I8pPGFEaneBpqnCGcXl3jqJ")

request = StockBarsRequest(
    symbol_or_symbols="IXHl",
    timeframe=TimeFrame.Minute,
    start=datetime.now(timezone.utc) - timedelta(days=1),
    end=datetime.now(timezone.utc)
)

bars = client.get_stock_bars(request).df
data = bars.xs("IXHL") if isinstance(bars.index, pd.MultiIndex) else bars

print("Latest timestamp:", data.index[-1])
print(data.tail())