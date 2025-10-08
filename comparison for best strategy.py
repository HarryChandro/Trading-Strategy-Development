import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from mplfinance.original_flavor import candlestick_ohlc
import warnings
from matplotlib.dates import date2num
from datetime import datetime, timedelta
import pytz
import optuna
from matplotlib import patches as mpatches
import mplfinance as mpf

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def is_valid_ticker(ticker: str) -> bool:
    """
    Checks if a given ticker symbol is valid on Yahoo Finance.

    Parameters:
        ticker (str): The ticker symbol to check.

    Returns:
        bool: True if the ticker is valid, False otherwise.
    """
    try:
        ticker_data = yf.Ticker(ticker)
        info = ticker_data.info  # Get the metadata
        return bool(info and 'regularMarketPrice' in info and info['regularMarketPrice'] is not None)
    except Exception:
        return False


# Example interactive usage
if __name__ == "__main__":
    ticker = input("Enter a ticker symbol: ").strip().upper()
    if is_valid_ticker(ticker):
        print(f"'{ticker}' is a valid ticker on Yahoo Finance.")
    else:
        print(f"'{ticker}' is NOT a valid ticker.")
        

def full_data():
    try:
        # Getting final datapoint
        start_date = datetime.now()-timedelta(days=7)
        end_date = datetime.now() + timedelta(days=1)    
        
        print(f"Requesting data from {start_date.date()} to {end_date.date()}")
        
        data = yf.download(
    ticker,
    start=start_date.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    interval="1m",
    auto_adjust=True,
    prepost=True,
    progress=False
    )
        # Debug info
        if not data.empty:
            print(f"Data retrieved: {len(data)} rows")
            print(f"Date range: {data.index[0]} to {data.index[-1]}")
            print(f"Unique dates: {sorted(set(data.index.date))}")
            
            # Check specifically for Aug 22 data
            all_data = data[data.index.date == pd.Timestamp('2025-10-01').date()]
            print(f"August 22 data points: {len(all_data)}")
        else:
            print("No data retrieved - empty DataFrame")
            
        return data
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Get data with error handling
data = full_data()

if data is None or data.empty:
    print("Failed to fetch intraday data. Please check your internet connection or try different dates.")
    exit()

# Print available date range
print(f"\nData available from {data.index[0]} to {data.index[-1]}")
print(f"Data frequency: {pd.infer_freq(data.index)}")
print(f"Number of data points: {len(data)}")

data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

s_lb = 5   # Short MA lower bound 
s_ub = 20   # Short MA upper bound 
l_lb = 20   # Long MA lower bound  
l_ub = 100   # Long MA upper bound 

holding_period = 1  
initial_balance = 200

def calculate_success_rate(data, signals):
    trades = []
    buy_price = None
    buy_date = None
    
    for i in range(len(data)):
        if signals.iloc[i] == 1 and buy_price is None:
            buy_price = data['Close'].iloc[i]
            buy_date = data.index[i]
        elif signals.iloc[i] == -1 and buy_price is not None:
            sell_price = data['Close'].iloc[i]
            sell_date = data.index[i]
            profit_pct = (sell_price - buy_price) / buy_price * 100
            trades.append({
                'Buy Date': buy_date, 'Sell Date': sell_date,
                'Buy Price': buy_price, 'Sell Price': sell_price,
                'Profit %': profit_pct, 'Success': profit_pct > 0
            })
            buy_price = None
            buy_date = None
    
    if buy_price is not None:
        sell_price = data['Close'].iloc[-1]
        profit_pct = (sell_price - buy_price) / buy_price * 100
        trades.append({
            'Buy Date': buy_date, 'Sell Date': data.index[-1],
            'Buy Price': buy_price, 'Sell Price': sell_price,
            'Profit %': profit_pct, 'Success': profit_pct > 0
        })
    
    if not trades:
        return 0.0, []
    
    success_rate = sum(trade['Success'] for trade in trades) / len(trades) * 100
    return success_rate, trades

def backtest_strategy_ema(data, sm, lg, stop_loss_ema=None):
    # Calculate EMAs
    ma_sm = data['Close'].ewm(span=sm, adjust=False).mean()
    ma_lg = data['Close'].ewm(span=lg, adjust=False).mean()

    # Generate signals: 1 = buy, -1 = sell
    signals = pd.Series(0, index=data.index)
    signals[(ma_sm > ma_lg) & (ma_sm.shift(1) <= ma_lg.shift(1))] = 1
    signals[(ma_sm < ma_lg) & (ma_sm.shift(1) >= ma_lg.shift(1))] = -1

    balance = initial_balance
    position = 0
    buy_price = 0
    buy_date = None
    equity_curve = []
    trades = []

    for i in range(len(data)):
        close = data['Close'].iloc[i]
        current_value = balance + (position * close if position > 0 else 0)
        equity_curve.append(current_value)

        # Skip trading in the last 'holding_period' days
        if i >= len(data) - holding_period:
            continue

        # Check for stop-loss first
        if position > 0 and stop_loss_ema is not None:
            stop_price = buy_price * (1 - stop_loss_ema / 100)
            if close <= stop_price:
                # Exit at stop-loss price
                balance = position * stop_price
                trades.append({
                    'Buy Date': buy_date,
                    'Sell Date': data.index[i],
                    'Buy Price': buy_price,
                    'Sell Price': stop_price,
                    'Profit %': -stop_loss_ema,
                    'Success': False,
                    'Stop Loss Hit': True
                })
                position = 0
                buy_price = 0
                buy_date = None
                continue  # Skip further checks for this candle

        # Buy signal
        if signals.iloc[i] == 1 and position == 0:
            position = balance / close
            balance = 0
            buy_price = close
            buy_date = data.index[i]

        # Sell signal
        elif signals.iloc[i] == -1 and position > 0:
            balance = position * close
            trades.append({
                'Buy Date': buy_date,
                'Sell Date': data.index[i],
                'Buy Price': buy_price,
                'Sell Price': close,
                'Profit %': (close - buy_price) / buy_price * 100,
                'Success': (close - buy_price) > 0,
                'Stop Loss Hit': False
            })
            position = 0
            buy_price = 0
            buy_date = None

    # Exit any remaining open position at the last close
    if position > 0:
        balance = position * data['Close'].iloc[-1]
        trades.append({
            'Buy Date': buy_date,
            'Sell Date': data.index[-1],
            'Buy Price': buy_price,
            'Sell Price': data['Close'].iloc[-1],
            'Profit %': (data['Close'].iloc[-1] - buy_price) / buy_price * 100,
            'Success': (data['Close'].iloc[-1] - buy_price) > 0,
            'Stop Loss Hit': False
        })

    # Calculate success rate
    success_rate = sum(t['Success'] for t in trades) / len(trades) * 100 if trades else 0

    return {
        'return_pct': (balance / initial_balance - 1) * 100,
        'signal_count': len(signals[signals != 0]),
        'success_rate': success_rate,
        'trades': trades,
        'signals': signals,
        'ma_sm': ma_sm,
        'ma_lg': ma_lg,
        'equity_curve': equity_curve
    }



# Optuna optimisation on short and long ema
def ema_objective(trial):
    sm = trial.suggest_int('short_ema', s_lb, s_ub-1)
    lg = trial.suggest_int('long_ema', l_lb, l_ub-1)
    
    if sm >= lg:
        return float('-inf')
    
    try:
        result = backtest_strategy_ema(data.copy(), sm, lg)
        return result['return_pct']
    except Exception:
        return float('-inf')
# Add callback to print each trial
# def print_trial(study, trial):
#     print(f"Trial {trial.number}: short_ema={trial.params['short_ema']}, long_ema={trial.params['long_ema']}, value={trial.value:.2f}")

# === EMA OPTIMIZATION ===
print("Starting EMA optimization...")
study_ema = optuna.create_study(direction='maximize')
study_ema.optimize(ema_objective, n_trials=250, show_progress_bar=True)

# Get best EMA parameters
best_ema_params = study_ema.best_params
sm = best_ema_params['short_ema']
lg = best_ema_params['long_ema']

# Run EMA backtest once, after EMA optimization, before any RSI optimization
ema_result = backtest_strategy_ema(data.copy(), sm, lg, stop_loss_ema=5)

# Print and plot EMA results
print(f"\n=== Best EMA Combination ===")
print(f"Short EMA: {sm} periods | Long EMA: {lg} periods")
print(f"Strategy Return: {ema_result['return_pct']:.1f}%")
print(f"Success Rate: {ema_result['success_rate']:.1f}%")
print(f"Signals Generated: {ema_result['signal_count']}")
print(f"Total Trades: {len(ema_result['trades'])}")

# Create figure with candlestick chart + equity + volume
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[4, 1, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Prepare data for candlestick plot
plot_data = data[['Open', 'High', 'Low', 'Close']].copy()
plot_data.reset_index(inplace=True)
date_col_name = plot_data.columns[0]
plot_data.rename(columns={date_col_name: 'Date'}, inplace=True)
plot_data['Date'] = plot_data['Date'].map(date2num)
plot_data = plot_data[['Date', 'Open', 'High', 'Low', 'Close']]

# Plot candlesticks
candlestick_ohlc(ax1, plot_data.values, width=0.001, colorup='limegreen', colordown='red', alpha=0.8)

# Plot EMA
ema_result['ma_sm'].plot(ax=ax1, label=f'EMA-{sm}', color='blue', linewidth=2.5)
ema_result['ma_lg'].plot(ax=ax1, label=f'EMA-{lg}', color='orange', linewidth=2.5)

# Plot signals
buy_signals = ema_result['signals'] == 1
sell_signals = ema_result['signals'] == -1

ax1.plot(data.index[buy_signals], data['Close'][buy_signals], 
        '^', markersize=12, color='lime', label='Buy', alpha=1, 
        markeredgewidth=2, markeredgecolor='darkgreen')

ax1.plot(data.index[sell_signals], data['Close'][sell_signals], 
        'v', markersize=12, color='red', label='Sell', alpha=1, 
        markeredgewidth=2, markeredgecolor='darkred')

# Format plot
ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
ax1.set_title(f'{ticker} EMA Crossover Strategy\n{sm}/{lg} periods | Return: {ema_result["return_pct"]:.1f}% | Success: {ema_result["success_rate"]:.1f}%', 
             fontsize=14, fontweight='bold', pad=20)
ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Equity curve plot
ax2.plot(data.index, ema_result['equity_curve'], label='Equity', color='purple', linewidth=2.5)
ax2.set_title('Strategy Equity Curve', fontsize=12, fontweight='bold')
ax2.set_ylabel('Portfolio Value ($)', fontsize=11)
ax2.grid(True, alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# Mark trades on equity curve
ax2.plot(data.index[buy_signals], np.array(ema_result['equity_curve'])[buy_signals], 
        '^', markersize=8, color='lime', alpha=1, markeredgewidth=1.5, markeredgecolor='darkgreen')
ax2.plot(data.index[sell_signals], np.array(ema_result['equity_curve'])[sell_signals], 
        'v', markersize=8, color='red', alpha=1, markeredgewidth=1.5, markeredgecolor='darkred')


plt.tight_layout()
plt.show()

# Prepare EMA report
ema_report = []

ema_report.append("\n=== Best EMA Combination ===")
ema_report.append(f"Short EMA: {sm} periods | Long EMA: {lg} periods")
ema_report.append(f"Strategy Return: {ema_result['return_pct']:.1f}%")
ema_report.append(f"Success Rate: {ema_result['success_rate']:.1f}%")
ema_report.append(f"Signals Generated: {ema_result['signal_count']}")
ema_report.append(f"Total Trades: {len(ema_result['trades'])}")

# Trade details
trades_df = pd.DataFrame(ema_result['trades'])
if not trades_df.empty:
    ema_report.append("\nTrade Details for EMA:")
    ema_report.append(trades_df.to_string())
else:
    ema_report.append("\nNo trades executed")

# Final status and data check
ema_report.append("\n=== FINAL STATUS ===")
ema_report.append(f"Latest data point: {data.index[-1]}")
current_time = datetime.now()
ema_report.append(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Calculate delay
latest_data_time = data.index[-1].tz_localize(None) if hasattr(data.index[-1], 'tz_localize') else data.index[-1].replace(tzinfo=None)
hours_diff = (current_time - latest_data_time).total_seconds() / 3600
ema_report.append(f"Data is {hours_diff:.1f} hours behind current time")

if hours_diff > 4:
    ema_report.append("Significant data delay detected!")
    ema_report.append("Recommendations:")
    ema_report.append("1. Check Yahoo Finance website directly")
    ema_report.append("2. Try a different data source")
    ema_report.append("3. Wait for data to update")
    ema_report.append(f"4. Check if {ticker} trades on your local exchanges during market hours")

# Print everything at once, flush immediately
print("\n".join(ema_report), flush=True)

# --- Then start RSI optimization ---
print("Running optimization to find best RSI levels...", flush=True)
# ... your RSI Optuna code follows ...


# Calculate RSI
def calculate_rsi(data, period=7):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data)

data = data.dropna()

# Backtest function with RSI strategy and stop loss
def backtest_rsi_strategy(data, overbought_level, oversold_level, stop_loss_pct=0.03, initial_capital=200):
    # Generate signals
    data = data.copy()
    data['Signal'] = 0
    data['Position'] = 0

    # RSI-based signals
    data.loc[data['RSI'] < oversold_level, 'Signal'] = 1  # Buy signal
    data.loc[data['RSI'] > overbought_level, 'Signal'] = -1  # Sell signal

    # Calculate trade performance with initial capital tracking and stop loss
    trades = []
    entry_price = None
    entry_date = None
    stop_loss_price = None
    current_capital = initial_capital
    capital_history = [{'Date': data.index[0], 'Capital': current_capital}]
    units = 0
    position = 0  # 0 = no position, 1 = long position

    for i in range(len(data)):
        current_low = data['Low'].iloc[i]
        current_high = data['High'].iloc[i]
        current_close = data['Close'].iloc[i]
        current_date = data.index[i]
        
        # Check for stop loss trigger first if in a position
        if position == 1 and entry_price is not None and current_low <= stop_loss_price:
            # Stop loss triggered
            exit_price = stop_loss_price
            exit_date = current_date
            
            # Calculate returns based on units purchased
            returns = (exit_price - entry_price) / entry_price * 100
            capital_gain = units * (exit_price - entry_price)
            current_capital = units * exit_price  # Update portfolio
            
            trades.append({
                'Entry Date': entry_date,
                'Exit Date': exit_date,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Units': units,
                'Return': returns,
                'Capital Gain': capital_gain,
                'Ending Capital': current_capital,
                'Success': returns > 0,
                'Stop Loss Hit': True
            })
            
            capital_history.append({'Date': exit_date, 'Capital': current_capital, 'Action': 'STOP LOSS'})
            
            # Reset position
            entry_price = None
            entry_date = None
            stop_loss_price = None
            units = 0
            position = 0
        
        # Check for regular exit signal (if in a position and not stopped out)
        if position == 1 and data['Signal'].iloc[i] == -1 and entry_price is not None:
            exit_price = current_close
            exit_date = current_date
            
            # Calculate returns based on units purchased
            returns = (exit_price - entry_price) / entry_price * 100
            capital_gain = units * (exit_price - entry_price)
            current_capital = units * exit_price  # Update capital
            
            trades.append({
                'Entry Date': entry_date,
                'Exit Date': exit_date,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Units': units,
                'Return': returns,
                'Capital Gain': capital_gain,
                'Ending Capital': current_capital,
                'Success': returns > 0,
                'Stop Loss Hit': False
            })
            
            capital_history.append({'Date': exit_date, 'Capital': current_capital, 'Action': 'SELL'})
            
            # Reset position
            entry_price = None
            entry_date = None
            stop_loss_price = None
            units = 0
            position = 0
        
        # Check for entry signal (if not in a position)
        if position == 0 and data['Signal'].iloc[i] == 1 and current_capital > 0:
            # Entry signal
            entry_price = current_close
            entry_date = current_date
            units = current_capital / entry_price  # Calculate how many units we can buy
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            position = 1  # Set position to long
            
            capital_history.append({'Date': entry_date, 'Capital': current_capital, 'Action': 'BUY'})

    # If still in a position at the end of data, calculate unrealized PnL
    if position == 1 and entry_price is not None:
        exit_price = data['Close'].iloc[-1]
        exit_date = data.index[-1]
        returns = (exit_price - entry_price) / entry_price * 100
        capital_gain = units * (exit_price - entry_price)
        current_capital = units * exit_price
        
        # Check if stop loss would have been hit
        stop_loss_hit = exit_price <= stop_loss_price if stop_loss_price is not None else False
        
        trades.append({
            'Entry Date': entry_date,
            'Exit Date': exit_date,
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Units': units,
            'Return': returns,
            'Capital Gain': capital_gain,
            'Ending Capital': current_capital,
            'Success': returns > 0,
            'Stop Loss Hit': stop_loss_hit,
            'Unrealized': True
        })
        
        capital_history.append({'Date': exit_date, 'Capital': current_capital, 'Action': 'SELL (Unrealized)'})

    # Calculate performance metrics
    if trades:
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        successful_trades = len(trades_df[trades_df['Success'] == True])
        stop_loss_trades = len(trades_df[trades_df['Stop Loss Hit'] == True])
        success_rate = successful_trades / total_trades * 100 if total_trades > 0 else 0
        stop_loss_rate = stop_loss_trades / total_trades * 100 if total_trades > 0 else 0
        avg_return = trades_df['Return'].mean() if total_trades > 0 else 0
        
        # Calculate total return based on initial capital
        total_return = (current_capital - initial_capital) / initial_capital * 100
        
        # Calculate max drawdown
        capital_df = pd.DataFrame(capital_history)
        capital_df['Peak'] = capital_df['Capital'].cummax()
        capital_df['Drawdown'] = (capital_df['Capital'] - capital_df['Peak']) / capital_df['Peak'] * 100
        max_drawdown = capital_df['Drawdown'].min()
        
        return {
            'total_trades': total_trades,
            'success_rate': success_rate,
            'stop_loss_rate': stop_loss_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'ending_capital': current_capital,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'capital_history': capital_history
        }
    else:
        return {
            'total_trades': 0,
            'success_rate': 0,
            'stop_loss_rate': 0,
            'avg_return': 0,
            'total_return': 0,
            'ending_capital': initial_capital,
            'max_drawdown': 0,
            'trades': [],
            'capital_history': capital_history
        }

# Objective function for Optuna optimization
def objective(trial):
    # Suggest overbought and oversold levels
    overbought_level = trial.suggest_int('overbought_level', 60, 90)
    oversold_level = trial.suggest_int('oversold_level', 10, 40)
    
    # Ensure overbought level is higher than oversold level
    if overbought_level <= oversold_level:
        return 0
    
    # Run backtest with £200 initial capital and 20% stop loss
    results = backtest_rsi_strategy(data, overbought_level, oversold_level, 
                                   stop_loss_pct=0.2, initial_capital=200)
    
    # We want to maximize total return while having a reasonable number of trades
    # Penalize strategies with too few trades
    if results['total_trades'] < 3:
        return 0
    
    # Calculate score: total return adjusted by success rate
    score = results['total_return'] * (1 + results['success_rate'] / 100)
    
    return score

# === RSI OPTIMIZATION ===
print("\nRunning optimization to find best RSI levels...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # Reduced trials for faster execution

# Get best parameters
best_params = study.best_params
best_overbought = best_params['overbought_level']
best_oversold = best_params['oversold_level']
best_value = study.best_value

print("\n" + "="*60)
print("OPTIMAL RSI LEVELS FOUND")
print("="*60)
print(f"Best Overbought Level: {best_overbought}")
print(f"Best Oversold Level: {best_oversold}")
print(f"Best Score: {best_value:.2f}")

# Run backtest with optimal parameters
print("\n" + "="*60)
print("PERFORMANCE WITH OPTIMAL PARAMETERS (3% STOP LOSS)")
print("="*60)
optimal_results = backtest_rsi_strategy(data, best_overbought, best_oversold, 
                                      stop_loss_pct=0.03, initial_capital=200)

print(f"Initial Capital: £200.00")
print(f"Ending Capital: £{optimal_results['ending_capital']:.2f}")
print(f"Total Return: {optimal_results['total_return']:.2f}%")
print(f"Total Trades: {optimal_results['total_trades']}")
print(f"Success Rate: {optimal_results['success_rate']:.2f}%")
print(f"Stop Loss Hit Rate: {optimal_results['stop_loss_rate']:.2f}%")
print(f"Average Return per Trade: {optimal_results['avg_return']:.2f}%")
print(f"Maximum Drawdown: {optimal_results['max_drawdown']:.2f}%")

# Print individual trades
if optimal_results['trades']:
    print("\nINDIVIDUAL TRADES:")
    for i, trade in enumerate(optimal_results['trades']):
        status = "UNREALIZED" if 'Unrealized' in trade and trade['Unrealized'] else "CLOSED"
        stop_loss_info = " (STOP LOSS)" if trade['Stop Loss Hit'] else ""
        print(f"Trade {i+1}: {status}{stop_loss_info}")
        print(f"  Entry: {trade['Entry Date'].strftime('%Y-%m-%d %H:%M')} at ${trade['Entry Price']:.2f}")
        print(f"  Exit: {trade['Exit Date'].strftime('%Y-%m-%d %H:%M')} at ${trade['Exit Price']:.2f}")
        print(f"  Units: {trade['Units']:.4f}")
        print(f"  Return: {trade['Return']:.2f}%")
        print(f"  P&L: £{trade['Capital Gain']:.2f}")
        print(f"  Capital after trade: £{trade['Ending Capital']:.2f}")
        print()

# Prepare data for plotting with optimal parameters
plot_data = data.copy()
plot_data['Signal'] = 0
plot_data['Position'] = 0

# RSI-based signals with optimal levels
plot_data.loc[plot_data['RSI'] < best_oversold, 'Signal'] = 1  # Buy signal
plot_data.loc[plot_data['RSI'] > best_overbought, 'Signal'] = -1  # Sell signal

# Create positions for visualisation only
position = 0
for i in range(len(plot_data)):
    if plot_data['Signal'].iloc[i] == 1 and position == 0:
        plot_data.iloc[i, plot_data.columns.get_loc('Position')] = 1
        position = 1
    elif plot_data['Signal'].iloc[i] == -1 and position == 1:
        plot_data.iloc[i, plot_data.columns.get_loc('Position')] = -1
        position = 0
    else:
        plot_data.iloc[i, plot_data.columns.get_loc('Position')] = 0

# Get start and end dates for title
start_date = plot_data.index[0].strftime('%Y-%m-%d')
end_date = plot_data.index[-1].strftime('%Y-%m-%d')

# Create arrays for buy/sell signals that match the length of the data
buy_signals = pd.Series(np.nan, index=plot_data.index)
sell_signals = pd.Series(np.nan, index=plot_data.index)

buy_idx = plot_data[plot_data['Position'] == 1].index
sell_idx = plot_data[plot_data['Position'] == -1].index

if not buy_idx.empty:
    buy_signals.loc[buy_idx] = plot_data.loc[buy_idx, 'Close']  # Show on closing price
    
if not sell_idx.empty:
    sell_signals.loc[sell_idx] = plot_data.loc[sell_idx, 'Close']  # Show on closing price

# Create additional plots
apds = [
    mpf.make_addplot(plot_data['RSI'], panel=1, color='purple', ylabel='RSI'),
    mpf.make_addplot([best_overbought] * len(plot_data), panel=1, color='red', linestyle='--', alpha=0.7),
    mpf.make_addplot([best_oversold] * len(plot_data), panel=1, color='green', linestyle='--', alpha=0.7),
]

# Add markers for buy and sell signals directly on candles
if not buy_idx.empty:
    apds.append(mpf.make_addplot(buy_signals, 
                                scatter=True, 
                                markersize=100, 
                                marker='^', 
                                color='lime',
                                panel=0))
if not sell_idx.empty:
    apds.append(mpf.make_addplot(sell_signals, 
                                scatter=True, 
                                markersize=100, 
                                marker='v', 
                                color='red',
                                panel=0))

# Create custom style
mc = mpf.make_marketcolors(up='g', down='r')
s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':')

# Plot candlestick chart with RSI and signals
fig, axes = mpf.plot(
    plot_data,
    type='candle',
    style=s,
    title=f'{ticker} Price with Optimal RSI Signals\n{start_date} to {end_date}',
    volume=False,
    figratio=(12, 8),
    ylabel='Price (USD)',
    addplot=apds,
    returnfig=True
)

green_patch = mpatches.Patch(color='green', label=f'Buy Signal (RSI < {best_oversold})')
red_patch = mpatches.Patch(color='red', label=f'Sell Signal (RSI > {best_overbought})')
axes[0].legend(handles=[green_patch, red_patch], loc='upper left')

# Add text box with performance summary
performance_text = f"Initial: £200.00\nEnding: £{optimal_results['ending_capital']:.2f}\nTotal Return: {optimal_results['total_return']:.1f}%\nSuccess Rate: {optimal_results['success_rate']:.1f}%\nStop Loss Rate: {optimal_results['stop_loss_rate']:.1f}%"
axes[0].text(0.02, 0.02, performance_text, transform=axes[0].transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            verticalalignment='bottom')

# Add RSI levels to the RSI panel
axes[2].text(0.02, 0.95, f"Overbought: {best_overbought}", transform=axes[2].transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
            verticalalignment='top')
axes[2].text(0.02, 0.05, f"Oversold: {best_oversold}", transform=axes[2].transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3),
            verticalalignment='bottom')

plt.show()

# Create capital growth chart
capital_df = pd.DataFrame(optimal_results['capital_history'])
capital_df.set_index('Date', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(capital_df.index, capital_df['Capital'], label='Portfolio Value', linewidth=2)
plt.axhline(y=200, color='r', linestyle='--', alpha=0.7, label='Initial Capital (£200)')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (£)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

optuna.visualization.plot_optimization_history(study).show()

# Ensure both result variables exist
try:
    ema_return = ema_result['return_pct']
    rsi_return = optimal_results['total_return']

    print("\n" + "="*60)
    print("📊 STRATEGY PERFORMANCE COMPARISON")
    print("="*60)
    print(f"EMA Crossover Return: {ema_return:.2f}%")
    print(f"RSI Strategy Return:  {rsi_return:.2f}%")

    if ema_return > rsi_return:
        print(f"\n🏆 Best Strategy: EMA Crossover Short EMA: {sm} periods | Long EMA: {lg} periods) giving a return of {ema_return:.2f}% Return")
        best_strategy = "EMA"
        best_return = ema_return
    elif rsi_return > ema_return:
        print(f"\n🏆 Best Strategy: RSI Strategy ({rsi_return:.2f}% Return)")
        best_strategy = "RSI"
        best_return = rsi_return
    else:
        print(f"\n🤝 Both strategies performed equally ({ema_return:.2f}% Return)")
        best_strategy = "Equal"
        best_return = ema_return

    print("="*60)
    print(f"✅ Best Performing Strategy: {best_strategy}")
    print(f"📈 Best Return: {best_return:.2f}%")
    print("="*60)

except NameError as e:
    print(f"Error: Could not compare strategies because one result is missing. ({e})")
