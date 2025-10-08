import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from mplfinance.original_flavor import candlestick_ohlc
from itertools import product
import warnings
from matplotlib.dates import date2num
from datetime import datetime, timedelta
import pytz
# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ticker = "IXHL"

def get_intraday_data():
    try:
        # MANUALLY SET THE DATE RANGE FOR AUGUST 22, 2025
        start_date = datetime(2025, 9, 10)  # Start from previous day to ensure coverage
        end_date = datetime(2025, 10, 1)    # End at next day to capture all of Aug 22
        
        print(f"Requesting data from {start_date.date()} to {end_date.date()}")
        
        data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval="15m",
            progress=False,
            auto_adjust=True,
            prepost=True
        )
        
        if data.empty:
            print("No 5m data available, trying 15m intervals...")
            data = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval="15m",
                progress=False,
                auto_adjust=True
            )
        
        if data.empty:
            print("No 15m data available, trying 60m intervals...")
            data = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval="60m",
                progress=False,
                auto_adjust=True
            )
        
        # REMOVED THE FILTERING SECTION - KEEP ALL DATA
        
        # Debug info
        if not data.empty:
            print(f"Data retrieved: {len(data)} rows")
            print(f"Date range: {data.index[0]} to {data.index[-1]}")
            print(f"Unique dates: {sorted(set(data.index.date))}")
            
            # Check specifically for Aug 22 data
            aug_22_data = data[data.index.date == pd.Timestamp('2025-08-22').date()]
            print(f"August 22 data points: {len(aug_22_data)}")
        else:
            print("No data retrieved - empty DataFrame")
            
        return data
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Get data with error handling
data = get_intraday_data()

if data is None or data.empty:
    print("Failed to fetch intraday data. Please check your internet connection or try different dates.")
    exit()

# Print available date range
print(f"\nData available from {data.index[0]} to {data.index[-1]}")
print(f"Data frequency: {pd.infer_freq(data.index)}")
print(f"Number of data points: {len(data)}")

# Clean and prepare data
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Parameter ranges optimized for intraday trading
s_lb = 5   # Short MA lower bound (25 minutes)
s_ub = 20   # Short MA upper bound (105 minutes)
l_lb = 21   # Long MA lower bound (100 minutes)  
l_ub = 50   # Long MA upper bound (255 minutes)

short_mas = range(s_lb, s_ub)
long_mas = range(l_lb, l_ub)
holding_period = 1  # Reduced holding period for intraday
initial_balance = 200

def calculate_success_rate(data, signals):
    trades = []
    buy_price = None
    buy_date = None
    
    for i in range(len(data)):
        if signals.iloc[i] == 1 and buy_price is None:  # Buy signal
            buy_price = data['Close'].iloc[i]
            buy_date = data.index[i]
        elif signals.iloc[i] == -1 and buy_price is not None:  # Sell signal
            sell_price = data['Close'].iloc[i]
            sell_date = data.index[i]
            profit_pct = (sell_price - buy_price) / buy_price * 100
            trades.append({
                'Buy Date': buy_date,
                'Sell Date': sell_date,
                'Buy Price': buy_price,
                'Sell Price': sell_price,
                'Profit %': profit_pct,
                'Success': profit_pct > 0
            })
            buy_price = None
            buy_date = None
    
    if buy_price is not None:  # Handle open trades
        sell_price = data['Close'].iloc[-1]
        profit_pct = (sell_price - buy_price) / buy_price * 100
        trades.append({
            'Buy Date': buy_date,
            'Sell Date': data.index[-1],
            'Buy Price': buy_price,
            'Sell Price': sell_price,
            'Profit %': profit_pct,
            'Success': profit_pct > 0
        })
    
    if not trades:
        return 0.0, []
    
    success_rate = sum(trade['Success'] for trade in trades) / len(trades) * 100
    return success_rate, trades

def backtest_strategy(data, sm, lg):
    # Calculate Moving Averages
    ma_sm = data['Close'].rolling(sm).mean()
    ma_lg = data['Close'].rolling(lg).mean()
    
    # Generate Signals
    signals = pd.Series(0, index=data.index)
    signals[(ma_sm > ma_lg) & (ma_sm.shift(1) <= ma_lg.shift(1))] = 1    # Buy
    signals[(ma_sm < ma_lg) & (ma_sm.shift(1) >= ma_lg.shift(1))] = -1   # Sell
    
    # Trade simulation
    balance = initial_balance
    position = 0
    equity_curve = []
    
    for i in range(len(data)):
        current_value = balance + (position * data['Close'].iloc[i] if position > 0 else 0)
        equity_curve.append(current_value)
        
        if i >= len(data) - holding_period:
            continue
            
        if signals.iloc[i] == 1 and position <= 0:  # Buy signal
            position = balance / data['Close'].iloc[i]
            balance = 0
        elif signals.iloc[i] == -1 and position > 0:  # Sell signal
            balance = position * data['Close'].iloc[i]
            position = 0
    
    # Final portfolio value
    if position > 0:
        balance = position * data['Close'].iloc[-1]
        equity_curve[-1] = balance
    
    # Calculate success rate
    success_rate, trades = calculate_success_rate(data, signals)
    
    return {
        'return_pct': (balance/initial_balance-1)*100,
        'signal_count': len(signals[signals != 0]),
        'success_rate': success_rate,
        'trades': trades,
        'signals': signals,
        'ma_sm': ma_sm,
        'ma_lg': ma_lg,
        'equity_curve': equity_curve
    }

# Test all valid combinations
results = []
for sm, lg in product(short_mas, long_mas):
    if sm >= lg:
        continue
        
    try:
        result = backtest_strategy(data.copy(), sm, lg)
        results.append({
            'Short MA': sm,
            'Long MA': lg,
            'Return %': result['return_pct'],
            'Success Rate %': result['success_rate'],
            'Result': result
        })
    except Exception as e:
        continue

# Find the best performing combination
if results:
    best_combo = max(results, key=lambda x: x['Return %'])
    sm = best_combo['Short MA']
    lg = best_combo['Long MA']
    result = best_combo['Result']
    
    print(f"\n=== Best MA Combination ===")
    print(f"Short MA: {sm} periods | Long MA: {lg} periods")
    print(f"Strategy Return: {best_combo['Return %']:.1f}%")
    print(f"Success Rate: {best_combo['Success Rate %']:.1f}%")
    print(f"Signals Generated: {result['signal_count']}")
    print(f"Total Trades: {len(result['trades'])}")
    
    # Create figure with LARGER candlestick chart + equity + volume
    fig = plt.figure(figsize=(16, 14))  # Increased overall figure size
    gs = fig.add_gridspec(3, 1, height_ratios=[4, 1, 1])  # Made candlestick area larger
    ax1 = fig.add_subplot(gs[0])  # Candlesticks (larger)
    ax2 = fig.add_subplot(gs[1])  # Equity curve
    ax3 = fig.add_subplot(gs[2])  # Volume
    
    # Prepare data for candlestick plot
    plot_data = data[['Open', 'High', 'Low', 'Close']].copy()
    plot_data.reset_index(inplace=True)
    
    # Check what the date column is named and rename it to 'Date'
    date_col_name = plot_data.columns[0]
    plot_data.rename(columns={date_col_name: 'Date'}, inplace=True)
    
    plot_data['Date'] = plot_data['Date'].map(date2num)
    plot_data = plot_data[['Date', 'Open', 'High', 'Low', 'Close']]

    # Plot candlesticks with better visibility
    candlestick_ohlc(ax1, plot_data.values, width=0.001, colorup='limegreen', colordown='red', alpha=0.8)

    # Plot moving averages with thicker lines
    result['ma_sm'].plot(ax=ax1, label=f'MA-{sm}', color='blue', linewidth=2.5)
    result['ma_lg'].plot(ax=ax1, label=f'MA-{lg}', color='orange', linewidth=2.5)
    
    # Plot signals with LARGER markers and better visibility
    buy_signals = result['signals'] == 1
    sell_signals = result['signals'] == -1
    
    # Buy signals - larger and more visible
    ax1.plot(data.index[buy_signals], data['Close'][buy_signals], 
            '^', markersize=12, color='lime', label='Buy', alpha=1, 
            markeredgewidth=2, markeredgecolor='darkgreen')
    
    # Sell signals - larger and more visible
    ax1.plot(data.index[sell_signals], data['Close'][sell_signals], 
            'v', markersize=12, color='red', label='Sell', alpha=1, 
            markeredgewidth=2, markeredgecolor='darkred')
    
    # Shade area between MAs with more transparency
    ax1.fill_between(data.index, result['ma_sm'], result['ma_lg'], 
                    where=(result['ma_sm'] >= result['ma_lg']), 
                    facecolor='green', alpha=0.15)
    ax1.fill_between(data.index, result['ma_sm'], result['ma_lg'], 
                    where=(result['ma_sm'] < result['ma_lg']), 
                    facecolor='red', alpha=0.15)
    
    # Format candlestick plot with larger fonts
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
    ax1.set_title(f'Intraday MA Crossover Strategy: {sm}/{lg} periods | Return: {best_combo["Return %"]:.1f}% | Success: {best_combo["Success Rate %"]:.1f}%', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Equity curve plot
    ax2.plot(data.index, result['equity_curve'], label='Equity', color='purple', linewidth=2.5)
    ax2.set_title('Strategy Equity Curve', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=9)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Mark trades on equity curve with larger markers
    ax2.plot(data.index[buy_signals], np.array(result['equity_curve'])[buy_signals], 
            '^', markersize=8, color='lime', alpha=1, markeredgewidth=1.5, markeredgecolor='darkgreen')
    ax2.plot(data.index[sell_signals], np.array(result['equity_curve'])[sell_signals], 
            'v', markersize=8, color='red', alpha=1, markeredgewidth=1.5, markeredgecolor='darkred')
    
    # Volume plot (colored by price direction)
    up = data[data['Close'] >= data['Open']]
    down = data[data['Close'] < data['Open']]
    ax3.bar(up.index, up['Volume'], color='limegreen', alpha=0.7, width=0.6)
    ax3.bar(down.index, down['Volume'], color='red', alpha=0.7, width=0.6)
    ax3.set_title('Trading Volume', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Volume', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=9)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print trade-by-trade results
    print("\nTrade Details:")
    trades_df = pd.DataFrame(result['trades'])
    if not trades_df.empty:
        print(trades_df.to_string())
    else:
        print("No trades executed")
else:
    print("No valid combinations found")
