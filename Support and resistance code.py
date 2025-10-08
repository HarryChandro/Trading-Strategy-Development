import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

ticker = "IXHL"

# 1. Fetch data for specific date range
data = yf.download(ticker, start="2025-08-13", end="2025-08-24", interval="1h")

# 2. Flatten MultiIndex columns
data.columns = ['_'.join(col) for col in data.columns]

# 3. Rename columns to OHLCV standard
data.rename(columns={
    f'Open_{ticker}': 'Open',
    f'High_{ticker}': 'High',
    f'Low_{ticker}': 'Low',
    f'Close_{ticker}': 'Close',
    f'Volume_{ticker}': 'Volume'
}, inplace=True)

# Function to identify support and resistance levels
def find_support_resistance_levels(data, window=20):
    """
    Identify support and resistance levels using local minima and maxima
    """
    # Find local minima (support levels)
    support_levels = data.iloc[argrelextrema(data['Low'].values, np.less_equal, order=window)[0]]['Low']
    
    # Find local maxima (resistance levels)
    resistance_levels = data.iloc[argrelextrema(data['High'].values, np.greater_equal, order=window)[0]]['High']
    
    # Filter levels to only include significant ones (avoid noise)
    support_levels = support_levels[support_levels > 0]
    resistance_levels = resistance_levels[resistance_levels > 0]
    
    return support_levels, resistance_levels

# Function to backtest support/resistance bounce strategy with COMPOUND trading
def backtest_support_resistance_compound(data, support_levels, resistance_levels, initial_balance=200, 
                                       tolerance=0.02, stop_loss_pct=0.04, target_pct=0.05):
    """
    Backtest a support/resistance bounce strategy with COMPOUND trading
    Each trade uses the full capital from the previous trade
    """
    trades = []
    position = None
    entry_price = None
    entry_date = None
    stop_loss = None
    target = None
    stop_loss_hit = False
    
    # Capital tracking - starts with initial balance
    current_balance = initial_balance
    previous_balance = initial_balance  # Track previous balance for return calculation
    capital_history = []  # We'll build this differently
    
    # Create a full timeline of capital values
    capital_timeline = pd.Series(index=data.index, dtype=float)
    capital_timeline.iloc[0] = initial_balance
    
    units = 0
    
    # Convert support and resistance levels to arrays for easier comparison
    support_array = support_levels.values
    resistance_array = resistance_levels.values
    
    for i in range(len(data)):
        current_low = data['Low'].iloc[i]
        current_high = data['High'].iloc[i]
        current_close = data['Close'].iloc[i]
        current_date = data.index[i]
        
        # Update capital timeline for this period
        if position == 'long':
            # If we're in a position, update balance based on current price
            capital_timeline.iloc[i] = units * current_close
        else:
            # If not in a position, balance remains the same
            capital_timeline.iloc[i] = current_balance
        
        # Check if we're not in a position
        if position is None:
            # Look for buy opportunities at support
            for support in support_array:
                if abs(current_low - support) / support <= tolerance and current_balance > 0:
                    # Buy at support with ALL available capital (compound)
                    position = 'long'
                    entry_price = current_close
                    entry_date = current_date
                    units = current_balance / entry_price  # Calculate how many units we can buy
                    previous_balance = current_balance  # Store balance before investment
                    current_balance = 0  # All capital is invested
                    stop_loss = entry_price * (1 - stop_loss_pct)
                    # Target is next resistance level or percentage-based
                    next_resistance = resistance_array[resistance_array > entry_price]
                    if len(next_resistance) > 0:
                        target = min(next_resistance)
                    else:
                        target = entry_price * (1 + target_pct)  # Default target if no resistance found
                    
                    capital_history.append({
                        'Date': current_date, 
                        'Balance': current_balance, 
                        'Action': f'BUY {units:.4f} units at ${entry_price:.2f}'
                    })
                    break
            
            # Look for short opportunities at resistance (if not already in a long position)
            if position is None:
                for resistance in resistance_array:
                    if abs(current_high - resistance) / resistance <= tolerance and current_balance > 0:
                        # For simplicity, we'll only implement long positions in this example
                        pass
        
        # If we're in a position, check for exit conditions
        else:
            stop_loss_hit = False
            exit_price = None
            exit_date = None
            
            if position == 'long':
                # Check for stop loss hit
                if current_low <= stop_loss:
                    exit_price = stop_loss
                    exit_date = current_date
                    returns_pct = (exit_price - entry_price) / entry_price * 100
                    stop_loss_hit = True
                
                # Check for target hit
                elif current_high >= target:
                    exit_price = target
                    exit_date = current_date
                    returns_pct = (exit_price - entry_price) / entry_price * 100
                
                # If either stop loss or target was hit, close the position
                if stop_loss_hit or current_high >= target:
                    # Calculate new balance after selling (COMPOUNDING)
                    new_balance = units * exit_price
                    trade_return = new_balance - previous_balance  # Return from this trade only
                    
                    trades.append({
                        'Entry Date': entry_date,
                        'Exit Date': exit_date,
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'Units': units,
                        'Return %': returns_pct,
                        'Return $': trade_return,  # Change from previous balance to current balance
                        'Position': position,
                        'Success': returns_pct > 0,
                        'Stop Loss Hit': stop_loss_hit,
                        'Starting Balance': previous_balance,  # Balance before this trade
                        'Ending Balance': new_balance,  # Balance after this trade
                        'Trade Number': len(trades) + 1
                    })
                    
                    # Update current balance for next trade
                    current_balance = new_balance
                    previous_balance = current_balance  # Reset for next trade
                    
                    capital_history.append({
                        'Date': current_date, 
                        'Balance': current_balance, 
                        'Action': f'SELL {units:.4f} units at ${exit_price:.2f}'
                    })
                    
                    # Reset position - ready for next trade with compounded capital
                    position = None
                    units = 0
    
    # If still in a position at the end, calculate unrealized PnL
    if position == 'long':
        exit_price = data['Close'].iloc[-1]
        exit_date = data.index[-1]
        returns_pct = (exit_price - entry_price) / entry_price * 100
        new_balance = units * exit_price
        trade_return = new_balance - previous_balance  # Return from this trade only
        
        trades.append({
            'Entry Date': entry_date,
            'Exit Date': exit_date,
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Units': units,
            'Return %': returns_pct,
            'Return $': trade_return,  # Change from previous balance to current balance
            'Position': position,
            'Success': returns_pct > 0,
            'Stop Loss Hit': False,
            'Unrealized': True,
            'Starting Balance': previous_balance,  # Balance before this trade
            'Ending Balance': new_balance,  # Balance after this trade
            'Trade Number': len(trades) + 1
        })
        
        current_balance = new_balance
        
        capital_history.append({
            'Date': exit_date, 
            'Balance': current_balance, 
            'Action': f'UNREALIZED - {units:.4f} units at ${exit_price:.2f}'
        })
    
    # Fill forward the capital timeline for the remaining periods
    for i in range(len(data)):
        if pd.isna(capital_timeline.iloc[i]):
            if position == 'long':
                capital_timeline.iloc[i] = units * data['Close'].iloc[i]
            else:
                capital_timeline.iloc[i] = current_balance
    
    # Calculate performance metrics
    if trades:
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        successful_trades = len(trades_df[trades_df['Success'] == True])
        stop_loss_trades = len(trades_df[trades_df['Stop Loss Hit'] == True])
        success_rate = successful_trades / total_trades * 100 if total_trades > 0 else 0
        stop_loss_rate = stop_loss_trades / total_trades * 100 if total_trades > 0 else 0
        avg_return_pct = trades_df['Return %'].mean() if total_trades > 0 else 0
        avg_return_dollar = trades_df['Return $'].mean() if total_trades > 0 else 0
        
        # Calculate overall return with compounding
        final_balance = current_balance
        total_return_pct = (final_balance - initial_balance) / initial_balance * 100
        total_return_dollar = final_balance - initial_balance
        
        # Calculate compound annual growth rate (CAGR)
        days = (data.index[-1] - data.index[0]).days
        years = days / 365.25
        cagr = ((final_balance / initial_balance) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Calculate risk-reward ratio
        winning_trades = trades_df[trades_df['Success'] == True]
        losing_trades = trades_df[trades_df['Success'] == False]
        
        avg_win = winning_trades['Return %'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['Return %'].mean() if len(losing_trades) > 0 else 0
        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return_pct': total_return_pct,
            'total_return_dollar': total_return_dollar,
            'cagr': cagr,
            'total_trades': total_trades,
            'success_rate': success_rate,
            'stop_loss_rate': stop_loss_rate,
            'avg_return_pct': avg_return_pct,
            'avg_return_dollar': avg_return_dollar,
            'risk_reward_ratio': risk_reward_ratio,
            'trades': trades,
            'capital_history': capital_history,
            'capital_timeline': capital_timeline  # Add the full timeline
        }
    else:
        return {
            'initial_balance': initial_balance,
            'final_balance': initial_balance,
            'total_return_pct': 0,
            'total_return_dollar': 0,
            'cagr': 0,
            'total_trades': 0,
            'success_rate': 0,
            'stop_loss_rate': 0,
            'avg_return_pct': 0,
            'avg_return_dollar': 0,
            'risk_reward_ratio': 0,
            'trades': [],
            'capital_history': capital_history,
            'capital_timeline': pd.Series(index=data.index, data=initial_balance)  # Constant balance
        }

# Function to analyze level effectiveness
def analyze_level_effectiveness(data, support_levels, resistance_levels, tolerance=0.02, lookforward=5):
    """
    Analyze how often price reverses vs breaks through support/resistance levels
    """
    support_results = []
    resistance_results = []
    
    # Analyze support levels
    for level in support_levels:
        # Find all times price approached this support level
        approaches = data[abs(data['Low'] - level) / level <= tolerance]
        
        for idx in approaches.index:
            approach_idx = data.index.get_loc(idx)
            if approach_idx + lookforward < len(data):
                # Check what happened in the next 'lookforward' periods
                future_data = data.iloc[approach_idx:approach_idx + lookforward]
                min_future = future_data['Low'].min()
                max_future = future_data['High'].max()
                
                # Determine if price bounced or broke through
                if min_future < level * (1 - tolerance):  # Broke through
                    support_results.append({'level': level, 'date': idx, 'result': 'break'})
                else:  # Bounced
                    support_results.append({'level': level, 'date': idx, 'result': 'bounce'})
    
    # Analyze resistance levels
    for level in resistance_levels:
        # Find all times price approached this resistance level
        approaches = data[abs(data['High'] - level) / level <= tolerance]
        
        for idx in approaches.index:
            approach_idx = data.index.get_loc(idx)
            if approach_idx + lookforward < len(data):
                # Check what happened in the next 'lookforward' periods
                future_data = data.iloc[approach_idx:approach_idx + lookforward]
                min_future = future_data['Low'].min()
                max_future = future_data['High'].max()
                
                # Determine if price bounced or broke through
                if max_future > level * (1 + tolerance):  # Broke through
                    resistance_results.append({'level': level, 'date': idx, 'result': 'break'})
                else:  # Bounced
                    resistance_results.append({'level': level, 'date': idx, 'result': 'bounce'})
    
    # Calculate effectiveness percentages
    support_df = pd.DataFrame(support_results)
    resistance_df = pd.DataFrame(resistance_results)
    
    if len(support_df) > 0:
        support_bounce_rate = len(support_df[support_df['result'] == 'bounce']) / len(support_df) * 100
    else:
        support_bounce_rate = 0
        
    if len(resistance_df) > 0:
        resistance_bounce_rate = len(resistance_df[resistance_df['result'] == 'bounce']) / len(resistance_df) * 100
    else:
        resistance_bounce_rate = 0
    
    return {
        'support_analysis': support_df,
        'resistance_analysis': resistance_df,
        'support_bounce_rate': support_bounce_rate,
        'resistance_bounce_rate': resistance_bounce_rate
    }

# Find support and resistance levels
support_levels, resistance_levels = find_support_resistance_levels(data, window=15)

print("Support Levels Found:")
print(support_levels.tail(10))

print("\nResistance Levels Found:")
print(resistance_levels.tail(10))

# Backtest the strategy with COMPOUND trading and £200 initial balance
initial_balance = 200
strategy_results = backtest_support_resistance_compound(data, support_levels, resistance_levels, 
                                                       initial_balance=initial_balance,
                                                       stop_loss_pct=0.02, target_pct=0.05)

print("\n" + "="*80)
print("SUPPORT/RESISTANCE BOUNCE STRATEGY RESULTS (COMPOUND TRADING)")
print("="*80)
print(f"Initial Balance: £{strategy_results['initial_balance']:.2f}")
print(f"Final Balance: £{strategy_results['final_balance']:.2f}")
print(f"Total Return: £{strategy_results['total_return_dollar']:.2f}")
print(f"Total Return: {strategy_results['total_return_pct']:.2f}%")
print(f"Compound Annual Growth Rate (CAGR): {strategy_results['cagr']:.2f}%")
print(f"Total Trades: {strategy_results['total_trades']}")
print(f"Success Rate: {strategy_results['success_rate']:.2f}%")
print(f"Stop Loss Hit Rate: {strategy_results['stop_loss_rate']:.2f}%")
print(f"Average Return per Trade: {strategy_results['avg_return_pct']:.2f}%")
print(f"Average Return per Trade: £{strategy_results['avg_return_dollar']:.2f}")
print(f"Risk-Reward Ratio: {strategy_results['risk_reward_ratio']:.2f}")

# Analyze level effectiveness
analysis_results = analyze_level_effectiveness(data, support_levels, resistance_levels)

print("\n" + "="*80)
print("LEVEL EFFECTIVENESS ANALYSIS")
print("="*80)
print(f"Support Bounce Rate: {analysis_results['support_bounce_rate']:.2f}%")
print(f"Resistance Bounce Rate: {analysis_results['resistance_bounce_rate']:.2f}%")

# Print individual trades with compounding effect
if strategy_results['trades']:
    print("\nINDIVIDUAL TRADES (COMPOUNDED):")
    for i, trade in enumerate(strategy_results['trades']):
        stop_loss_info = " (STOP LOSS)" if trade['Stop Loss Hit'] else ""
        unrealized_info = " (UNREALIZED)" if 'Unrealized' in trade and trade['Unrealized'] else ""
        print(f"Trade {trade['Trade Number']}: {trade['Position'].upper()}{stop_loss_info}{unrealized_info}")
        print(f"  Entry: {trade['Entry Date'].strftime('%Y-%m-%d')} at ${trade['Entry Price']:.2f}")
        print(f"  Exit: {trade['Exit Date'].strftime('%Y-%m-%d')} at ${trade['Exit Price']:.2f}")
        print(f"  Units: {trade['Units']:.4f}")
        print(f"  Starting Balance: £{trade['Starting Balance']:.2f}")
        print(f"  Return: {trade['Return %']:.2f}% (£{trade['Return $']:.2f})")
        print(f"  Ending Balance: £{trade['Ending Balance']:.2f}")
        print()

# Prepare data for plotting
plot_data = data.copy()

# Get start and end dates for title
start_date = plot_data.index[0].strftime('%Y-%m-%d')
end_date = plot_data.index[-1].strftime('%Y-%m-%d')

# Create additional plots for support/resistance levels
apds = []

# Add support levels
for level in support_levels.unique():
    apds.append(mpf.make_addplot([level] * len(plot_data), color='green', linestyle='--', alpha=0.5))

# Add resistance levels
for level in resistance_levels.unique():
    apds.append(mpf.make_addplot([level] * len(plot_data), color='red', linestyle='--', alpha=0.5))

# Plot candlestick chart with support/resistance levels
mpf.plot(
    plot_data,
    type='candle',
    style='charles',
    title=f'{ticker} Price with Support/Resistance Levels\n{start_date} to {end_date}',
    volume=False,
    figratio=(12, 8),
    ylabel='Price (USD)',
    addplot=apds
)

# Create capital growth chart with proper daily values
plt.figure(figsize=(12, 6))
plt.plot(strategy_results['capital_timeline'].index, strategy_results['capital_timeline'].values, 
         label='Portfolio Value', linewidth=2)
plt.axhline(y=initial_balance, color='r', linestyle='--', alpha=0.7, label=f'Initial Balance (£{initial_balance})')
plt.title('Portfolio Value Over Time (Compound Trading)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (£)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Create a separate plot to show level effectiveness
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Support level effectiveness
if len(analysis_results['support_analysis']) > 0:
    support_counts = analysis_results['support_analysis']['result'].value_counts()
    ax1.pie(support_counts.values, labels=support_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    ax1.set_title('Support Level Effectiveness')
else:
    ax1.text(0.5, 0.5, 'No support level data', horizontalalignment='center', verticalalignment='center')
    ax1.set_title('Support Level Effectiveness')

# Resistance level effectiveness
if len(analysis_results['resistance_analysis']) > 0:
    resistance_counts = analysis_results['resistance_analysis']['result'].value_counts()
    ax2.pie(resistance_counts.values, labels=resistance_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    ax2.set_title('Resistance Level Effectiveness')
else:
    ax2.text(0.5, 0.5, 'No resistance level data', horizontalalignment='center', verticalalignment='center')
    ax2.set_title('Resistance Level Effectiveness')

plt.tight_layout()
plt.show()