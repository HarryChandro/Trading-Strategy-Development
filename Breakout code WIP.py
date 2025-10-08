import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def calculate_support_resistance(data, window=20):
    """Calculate support and resistance levels using rolling high/low"""
    data = data.copy()
    data['Resistance'] = data['High'].rolling(window=window).max()
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Volume_MA'] = data['Volume'].rolling(window=window).mean()
    return data

def detect_breakouts(data, volume_multiplier=1.2):
    """Detect breakout signals with more sensitive parameters"""
    signals = []
    in_position = False
    entry_price = 0
    entry_date = None
    
    print(f"Analyzing {len(data)} data points...")
    
    for i in range(20, len(data)):
        if pd.isna(data.iloc[i]['Resistance']) or pd.isna(data.iloc[i]['Support']):
            continue
            
        current = data.iloc[i]
        prev = data.iloc[i-1]
        
        # Buy signal: Price breaks above resistance with increased volume
        resistance_break = (current['Close'] > current['Resistance'] and 
                           prev['Close'] <= prev['Resistance'])
        
        volume_condition = (current['Volume'] > current['Volume_MA'] * volume_multiplier)
        
        if resistance_break and volume_condition:
            if not in_position:
                signals.append(('BUY', data.index[i], current['Close'], current['Volume']))
                in_position = True
                entry_price = current['Close']
                entry_date = data.index[i]
                print(f"BUY Signal at {data.index[i]}: Price ${current['Close']:.2f}, "
                      f"Volume: {current['Volume']:,.0f} ({(current['Volume']/current['Volume_MA']):.1f}x MA)")
        
        # Sell signal: Price breaks below support OR stop loss
        elif in_position:
            support_break = (current['Close'] < current['Support'] and 
                            prev['Close'] >= prev['Support'])
            
            # Also sell if significant loss (stop loss)
            stop_loss = current['Close'] < entry_price * 0.95  # 5% stop loss
            
            if support_break or stop_loss:
                pnl = ((current['Close'] - entry_price) / entry_price) * 100
                holding_period = (data.index[i] - entry_date).days
                signals.append(('SELL', data.index[i], current['Close'], pnl, holding_period))
                in_position = False
                print(f"SELL Signal at {data.index[i]}: Price ${current['Close']:.2f}, "
                      f"PNL: {pnl:+.1f}%, Held: {holding_period} days")
    
    return signals

def analyze_breakout_sustainability(data, signals):
    """Analyze breakout sustainability vs fakeouts"""
    results = {
        'total_breakouts': 0,
        'sustained_breakouts': 0,
        'fakeouts': 0,
        'profitable_trades': 0,
        'total_trades': 0
    }
    
    trade_results = []
    
    for i in range(len(signals)):
        if signals[i][0] == 'BUY' and i + 1 < len(signals) and signals[i+1][0] == 'SELL':
            results['total_trades'] += 1
            buy_price = signals[i][2]
            sell_price = signals[i+1][2]
            pnl = signals[i+1][3]
            
            # Check if breakout was sustained (price stayed above resistance for at least 2 days)
            buy_idx = data.index.get_loc(signals[i][1])
            sustained = True
            
            # Check next 2 days for sustainability
            for j in range(1, 3):
                if buy_idx + j < len(data):
                    if data.iloc[buy_idx + j]['Close'] < data.iloc[buy_idx]['Resistance']:
                        sustained = False
                        break
            
            if sustained:
                results['sustained_breakouts'] += 1
            else:
                results['fakeouts'] += 1
            
            if pnl > 0:
                results['profitable_trades'] += 1
            
            trade_results.append({
                'buy_date': signals[i][1],
                'sell_date': signals[i+1][1],
                'buy_price': buy_price,
                'sell_price': sell_price,
                'pnl': pnl,
                'sustained': sustained,
                'holding_days': signals[i+1][4],
                'volume_ratio': signals[i][3] / data.iloc[buy_idx]['Volume_MA']
            })
    
    results['total_breakouts'] = results['sustained_breakouts'] + results['fakeouts']
    
    if results['total_trades'] > 0:
        results['win_rate'] = (results['profitable_trades'] / results['total_trades']) * 100
        if results['total_breakouts'] > 0:
            results['sustainability_rate'] = (results['sustained_breakouts'] / results['total_breakouts']) * 100
        else:
            results['sustainability_rate'] = 0
    else:
        results['win_rate'] = 0
        results['sustainability_rate'] = 0
    
    return results, trade_results

def plot_breakouts(data, signals):
    """Plot price chart with breakout signals"""
    if not signals:
        print("No signals to plot!")
        return
    
    # Create additional plots
    apds = [
        mpf.make_addplot(data['Resistance'], color='red', alpha=0.7, label='Resistance'),
        mpf.make_addplot(data['Support'], color='green', alpha=0.7, label='Support'),
    ]
    
    # Prepare buy/sell markers
    buy_dates = [s[1] for s in signals if s[0] == 'BUY']
    buy_prices = [s[2] for s in signals if s[0] == 'BUY']
    
    sell_dates = [s[1] for s in signals if s[0] == 'SELL']
    sell_prices = [s[2] for s in signals if s[0] == 'SELL']
    
    # Create the plot
    fig, axes = mpf.plot(
        data,
        type='candle',
        style='charles',
        title=f'{ticker} Breakout Trading Strategy',
        volume=True,
        figratio=(12, 8),
        ylabel='Price (USD)',
        addplot=apds,
        returnfig=True
    )
    
    # Add buy/sell markers
    if buy_dates:
        axes[0].scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy', zorder=5)
    if sell_dates:
        axes[0].scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell', zorder=5)
    
    axes[0].legend()
    plt.show()

# Main execution
ticker = "NVDA"

print("Fetching data...")
# Fetch data
data = yf.download(ticker, start="2023-01-01", end="2024-02-01")
print(f"Downloaded {len(data)} rows")

# Check if we have data
if len(data) == 0:
    print("No data downloaded! Check your ticker symbol and date range.")
else:
    # Clean up column names
    data.columns = ['_'.join(col).strip() for col in data.columns]
    
    # Find the correct column names
    print("Available columns:", data.columns.tolist())
    
    # Map columns - adjust based on actual column names
    column_map = {}
    for col in data.columns:
        if 'Open' in col:
            column_map[col] = 'Open'
        elif 'High' in col:
            column_map[col] = 'High'
        elif 'Low' in col:
            column_map[col] = 'Low'
        elif 'Close' in col:
            column_map[col] = 'Close'
        elif 'Volume' in col:
            column_map[col] = 'Volume'
    
    data.rename(columns=column_map, inplace=True)
    
    print("Columns after mapping:", data.columns.tolist())
    
    # Calculate support/resistance
    print("Calculating support/resistance levels...")
    data = calculate_support_resistance(data)
    
    # Remove NaN values
    data = data.dropna()
    print(f"Data points after removing NaN: {len(data)}")
    
    # Detect breakouts with more sensitive parameters
    print("Detecting breakouts...")
    signals = detect_breakouts(data, volume_multiplier=1.2)  # Reduced volume requirement
    
    # Analyze sustainability
    if signals:
        results, trade_results = analyze_breakout_sustainability(data, signals)
        
        # Print results
        print("\n" + "=" * 60)
        print("BREAKOUT TRADING STRATEGY ANALYSIS")
        print("=" * 60)
        print(f"Total Breakouts Detected: {results['total_breakouts']}")
        print(f"Sustained Breakouts: {results['sustained_breakouts']} ({results['sustainability_rate']:.1f}%)")
        print(f"Fakeouts: {results['fakeouts']} ({100 - results['sustainability_rate']:.1f}%)")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Profitable Trades: {results['profitable_trades']} ({results['win_rate']:.1f}% win rate)")
        
        if trade_results:
            print("\nIndividual Trade Results:")
            for i, trade in enumerate(trade_results, 1):
                print(f"Trade {i}: {trade['buy_date'].date()} to {trade['sell_date'].date()}, "
                      f"Price: ${trade['buy_price']:.2f} → ${trade['sell_price']:.2f}, "
                      f"PNL: {trade['pnl']:+.1f}%, Sustained: {trade['sustained']}")
        
        # Plot the chart with signals
        plot_breakouts(data, signals)
        
        # Additional analysis
        print("\n" + "=" * 60)
        print("CONTINUATION PREDICTION ANALYSIS")
        print("=" * 60)
        
        sustained_trades = [t for t in trade_results if t['sustained']]
        fakeout_trades = [t for t in trade_results if not t['sustained']]
        
        if sustained_trades:
            avg_sustained_pnl = np.mean([t['pnl'] for t in sustained_trades])
            avg_sustained_volume = np.mean([t['volume_ratio'] for t in sustained_trades])
            print(f"Sustained breakouts ({len(sustained_trades)}): Avg PNL: {avg_sustained_pnl:+.1f}%, "
                  f"Avg Volume: {avg_sustained_volume:.1f}x MA")
        
        if fakeout_trades:
            avg_fakeout_pnl = np.mean([t['pnl'] for t in fakeout_trades])
            avg_fakeout_volume = np.mean([t['volume_ratio'] for t in fakeout_trades])
            print(f"Fakeouts ({len(fakeout_trades)}): Avg PNL: {avg_fakeout_pnl:+.1f}%, "
                  f"Avg Volume: {avg_fakeout_volume:.1f}x MA")
        
    else:
        print("No breakout signals detected!")
        print("Possible reasons:")
        print("- Price never broke above resistance with sufficient volume")
        print("- Try adjusting volume_multiplier parameter")
        print("- Try a different ticker or time period")
        print("- Support/resistance levels might be too wide")
        
        # Show some statistics for debugging
        print(f"\nData range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        print(f"Average volume: {data['Volume'].mean():,.0f}")
        print(f"Max volume/MA ratio: {(data['Volume'] / data['Volume_MA']).max():.1f}x")