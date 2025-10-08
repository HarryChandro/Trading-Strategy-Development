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
import time
import smtplib
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==================== CONFIGURATION - EDIT THESE VALUES ====================
# Email settings for alerts
EMAIL_CONFIG = {
    "email": "harrykumar3008@gmail.com",  # Your Gmail address
    "password": "ndonwcrxxkbamweq",  # Gmail app password (16 characters, no spaces)
    "phone_email": "harrykumar3008@gmail.com",  # Where to send alerts
    "alert_email": "harrykumar3008@gmail.com"  # Where to send alerts
}

# Trading settings
TRADING_CONFIG = {
    "ticker": "IXHL",  # Stock symbol to track
    "check_interval": 900,  # Check every 15 minutes (900 seconds)
    "signal_threshold": 0.5,  # Minimum strength for signals
    "timezone": "Europe/London",  # Timezone for trading hours
    "initial_balance": 10000,  # Initial balance for backtesting
    "holding_period": 3,  # Holding period for trades
    "short_ema_range": (1, 20),  # Range for short EMA optimization
    "long_ema_range": (21, 50),  # Range for long EMA optimization
    "reoptimize_interval": 86400,  # Re-optimize every 24 hours (86400 seconds)
    "min_data_points": 100,  # Minimum data points for optimization
    "enable_email_alerts": False  # Set to True only after email is working
}

# ==================== END OF CONFIGURATION ====================

class IXHLTradingBot:
    def __init__(self, config):
        self.config = config
        self.ticker = config["ticker"]
        self.last_signal = 0
        self.signals_history = []
        self.optimal_short_ema = 12
        self.optimal_long_ema = 26
        self.last_optimization = None
        self.best_return = 0
        
    def setup_email(self):
        """Email setup with better error handling"""
        if not self.config["enable_email_alerts"]:
            print("ℹ️  Email alerts disabled in configuration - running in console mode")
            return True
            
        try:
            print("Testing email configuration...")
            server = smtplib.SMTP('smtp.gmail.com', 587, timeout=10)
            server.starttls()
            server.login(self.config["email"], self.config["password"])
            server.quit()
            print("✓ Email configuration successful")
            return True
        except socket.gaierror:
            print("✗ Network error: Cannot connect to email servers")
            print("💡 Running in console mode. Alerts will display on screen.")
            return False
        except Exception as e:
            print(f"✗ Email setup failed: {e}")
            print("💡 Running in console mode. Alerts will display on screen.")
            return False
    
    def get_intraday_data(self):
        """Robust data fetching function with multiple fallbacks"""
    try:
        # Get current date and time
        current_time = datetime.now()
        print(f"Current local time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Try multiple approaches to get the most recent data
        print("Trying different approaches to fetch data...")
        
        # APPROACH 1: Try with very recent period (last 1-2 days)
        for days_back in [1, 2]:
            try:
                print(f"Trying last {days_back} day(s) with 15m intervals...")
                data = yf.download(
                    self.ticker,
                    period="1mo",
                    interval="1h",
                    progress=False,
                    auto_adjust=True,
                    prepost=True
                )
                
                if not data.empty:
                    print(f"Success with {days_back} day period")
                    # Convert to specified timezone
                    if data.index.tz is not None:
                        tz = pytz.timezone(self.config["timezone"])
                        data.index = data.index.tz_convert(tz)
                    return data
            except Exception as e:
                print(f"Error with {days_back} day period: {e}")
        
        # APPROACH 2: Try with specific date range including today
        try:
            print("Trying specific date range including today...")
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=24)  # Last 24 hours
            
            data = yf.download(
                self.ticker,
                start=start_date,
                end=end_date,
                interval="15m",
                progress=False,
                auto_adjust=True,
                prepost=True
            )
            
            if not data.empty:
                print("Success with specific date range")
                if data.index.tz is not None:
                    tz = pytz.timezone(self.config["timezone"])
                    data.index = data.index.tz_convert(tz)
                return data
        except Exception as e:
            print(f"Error with specific date range: {e}")
        
        # APPROACH 3: Try different data sources (using different parameters)
        print("Trying alternative data fetching methods...")
        
        # Try with different intervals
        intervals = ["5m", "15m", "30m", "60m"]
        for interval in intervals:
            try:
                data = yf.download(
                    self.ticker,
                    period="2d",
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    prepost=True
                )
                if not data.empty:
                    print(f"Success with {interval} interval")
                    if data.index.tz is not None:
                        tz = pytz.timezone(self.config["timezone"])
                        data.index = data.index.tz_convert(tz)
                    return data
            except:
                continue
        
        # FINAL FALLBACK: Get whatever data is available
        print("Falling back to available data...")
        data = yf.download(
            self.ticker,
            period="7d",
            interval="1d",
            progress=False,
            auto_adjust=True
        )
        
        # Convert to timezone if possible
        if data.index.tz is not None:
            tz = pytz.timezone(self.config["timezone"])
            data.index = data.index.tz_convert(tz)
        
        return data
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    
    def calculate_success_rate(self, data, signals):
        """Calculate trading success rate"""
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

    def backtest_strategy_ema(self, data, sm, lg):
        """Backtest EMA crossover strategy"""
        # Calculate Exponential Moving Averages
        ma_sm = data['Close'].ewm(span=sm, adjust=False).mean()
        ma_lg = data['Close'].ewm(span=lg, adjust=False).mean()
        
        # Generate Signals
        signals = pd.Series(0, index=data.index)
        signals[(ma_sm > ma_lg) & (ma_sm.shift(1) <= ma_lg.shift(1))] = 1
        signals[(ma_sm < ma_lg) & (ma_sm.shift(1) >= ma_lg.shift(1))] = -1
        
        # Trade simulation
        balance = self.config["initial_balance"]
        position = 0
        equity_curve = []
        
        for i in range(len(data)):
            current_value = balance + (position * data['Close'].iloc[i] if position > 0 else 0)
            equity_curve.append(current_value)
            
            if i >= len(data) - self.config["holding_period"]:
                continue
                
            if signals.iloc[i] == 1 and position <= 0:
                position = balance / data['Close'].iloc[i]
                balance = 0
            elif signals.iloc[i] == -1 and position > 0:
                balance = position * data['Close'].iloc[i]
                position = 0
        
        # Final portfolio value
        if position > 0:
            balance = position * data['Close'].iloc[-1]
            equity_curve[-1] = balance
        
        # Calculate success rate
        success_rate, trades = self.calculate_success_rate(data, signals)
        
        return {
            'return_pct': (balance/self.config["initial_balance"]-1)*100,
            'signal_count': len(signals[signals != 0]),
            'success_rate': success_rate,
            'trades': trades,
            'signals': signals,
            'ma_sm': ma_sm,
            'ma_lg': ma_lg,
            'equity_curve': equity_curve
        }

    def find_optimal_ema_parameters(self):
        """Find the best EMA combination for maximum returns"""
        print("🔍 Finding optimal EMA parameters...")
        
        # Get sufficient historical data
        data = self.get_intraday_data(period="1mo", interval="1h")
        if data is None or len(data) < self.config["min_data_points"]:
            print("⚠️  Not enough data for optimization, using default parameters")
            return self.optimal_short_ema, self.optimal_long_ema
        
        results = []
        short_lb, short_ub = self.config["short_ema_range"]
        long_lb, long_ub = self.config["long_ema_range"]
        
        short_mas = range(short_lb, short_ub)
        long_mas = range(long_lb, long_ub)
        
        # Test all valid combinations
        for sm, lg in product(short_mas, long_mas):
            if sm >= lg:
                continue
                
            try:
                result = self.backtest_strategy_ema(data.copy(), sm, lg)
                results.append({
                    'Short EMA': sm,
                    'Long EMA': lg,
                    'Return %': result['return_pct'],
                    'Success Rate %': result['success_rate'],
                    'Signals': result['signal_count'],
                    'Result': result
                })
            except Exception as e:
                continue
        
        if results:
            # Find best combination by return percentage
            best_combo = max(results, key=lambda x: x['Return %'])
            self.optimal_short_ema = best_combo['Short EMA']
            self.optimal_long_ema = best_combo['Long EMA']
            self.best_return = best_combo['Return %']
            
            print(f"✅ Optimal parameters found:")
            print(f"   Short EMA: {self.optimal_short_ema} periods")
            print(f"   Long EMA: {self.optimal_long_ema} periods")
            print(f"   Expected Return: {self.best_return:.1f}%")
            print(f"   Success Rate: {best_combo['Success Rate %']:.1f}%")
            print(f"   Signals Generated: {best_combo['Signals']}")
            
        else:
            print("⚠️  No valid combinations found, using default parameters")
        
        self.last_optimization = datetime.now()
        return self.optimal_short_ema, self.optimal_long_ema

    def calculate_current_signal(self, data):
        """Calculate current trading signal using optimal EMAs"""
        if data is None or len(data) < self.optimal_long_ema:
            return 0, 0, 0
        
        # Calculate current EMAs
        ema_short = data['Close'].ewm(span=self.optimal_short_ema, adjust=False).mean()
        ema_long = data['Close'].ewm(span=self.optimal_long_ema, adjust=False).mean()
        
        # Calculate signal strength
        current_diff = (ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1] * 100
        prev_diff = (ema_short.iloc[-2] - ema_long.iloc[-2]) / ema_long.iloc[-2] * 100
        
        signal_strength = abs(current_diff)
        
        # Generate signals
        if (ema_short.iloc[-1] > ema_long.iloc[-1] and 
            ema_short.iloc[-2] <= ema_long.iloc[-2] and
            signal_strength >= self.config["signal_threshold"]):
            return 1, signal_strength, data['Close'].iloc[-1]  # BUY
            
        elif (ema_short.iloc[-1] < ema_long.iloc[-1] and 
              ema_short.iloc[-2] >= ema_long.iloc[-2] and
              signal_strength >= self.config["signal_threshold"]):
            return -1, signal_strength, data['Close'].iloc[-1]  # SELL
            
        else:
            return 0, signal_strength, data['Close'].iloc[-1]  # HOLD
    
    def send_alert(self, action, price, signal_strength):
        """Send trading alert to console and optionally email"""
        message = f"🚨 {self.ticker} {action} SIGNAL 🚨\n\n"
        message += f"Price: ${price:.2f}\n"
        message += f"Signal Strength: {signal_strength:.2f}%\n"
        message += f"EMA Parameters: {self.optimal_short_ema}/{self.optimal_long_ema}\n"
        message += f"Expected Return: {self.best_return:.1f}%\n"
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"\nAction: {action} {self.ticker}\n"
        
        print(f"\n{'='*60}")
        print(message)
        print(f"{'='*60}")
        
        # Try to send email if enabled
        if self.config["enable_email_alerts"]:
            try:
                self.send_email_alert(action, price, signal_strength, message)
            except Exception as e:
                print(f"⚠️  Could not send email alert: {e}")
    
    def send_email_alert(self, action, price, signal_strength, message):
        """Send trading alert via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config["email"]
            msg['To'] = self.config["alert_email"]
            msg['Subject'] = f'🚨 {self.ticker} {action} Alert - ${price:.2f}'
            
            body = message + f"\n\nThis is an automated alert from your IXHL Trading Bot with optimal EMA parameters."
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP('smtp.gmail.com', 587, timeout=10)
            server.starttls()
            server.login(self.config["email"], self.config["password"])
            text = msg.as_string()
            server.sendmail(self.config["email"], self.config["alert_email"], text)
            server.quit()
            
            print("✓ Email alert sent")
            
        except Exception as e:
            print(f"✗ Failed to send email alert: {e}")
            raise
    
    def is_market_open(self):
        """Check if market is likely open (simplified)"""
        now = datetime.now(pytz.timezone(self.config["timezone"]))
        
        # Basic market hours check (9:30 AM to 4:00 PM local time)
        if now.weekday() >= 5:  # Weekend
            return False
            
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def needs_reoptimization(self):
        """Check if it's time to reoptimize parameters"""
        if self.last_optimization is None:
            return True
        
        time_since_optimization = (datetime.now() - self.last_optimization).total_seconds()
        return time_since_optimization >= self.config["reoptimize_interval"]
    
    def run_bot(self):
        """Main bot execution"""
        print(f"\n{'='*70}")
        print(f"🚀 STARTING IXHL TRADING BOT WITH OPTIMAL EMA FINDING")
        print(f"{'='*70}")
        print(f"Ticker: {self.ticker}")
        print(f"Check Interval: {self.config['check_interval']} seconds")
        print(f"Reoptimize Every: {self.config['reoptimize_interval']/3600:.1f} hours")
        print(f"EMA Search Range: {self.config['short_ema_range']} to {self.config['long_ema_range']}")
        print(f"{'='*70}")
        
        # Setup email (will continue even if it fails)
        self.setup_email()
        
        # Initial optimization
        self.find_optimal_ema_parameters()
        
        print("\n🤖 Bot is running! Press Ctrl+C to stop")
        print("📊 Signals will display on screen below...")
        
        try:
            while True:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Reoptimize if needed
                if self.needs_reoptimization():
                    self.find_optimal_ema_parameters()
                
                # Check if market is likely open
                if not self.is_market_open():
                    print(f"{current_time} - Market closed, waiting...")
                    time.sleep(self.config["check_interval"])
                    continue
                
                # Fetch current data
                data = self.get_intraday_data(period="2d", interval="1h")
                
                if data is not None and len(data) > 0:
                    signal, strength, price = self.calculate_current_signal(data)
                    
                    # Log current status
                    status_msg = f"{current_time} - Price: ${price:.2f}, Signal: {signal}, Strength: {strength:.2f}%"
                    status_msg += f", EMAs: {self.optimal_short_ema}/{self.optimal_long_ema}"
                    print(status_msg)
                    
                    # Store signal history
                    self.signals_history.append({
                        'timestamp': datetime.now(),
                        'signal': signal,
                        'strength': strength,
                        'price': price,
                        'short_ema': self.optimal_short_ema,
                        'long_ema': self.optimal_long_ema
                    })
                    
                    # Send alert on signal change
                    if signal != self.last_signal and signal != 0:
                        action = "BUY" if signal == 1 else "SELL"
                        self.send_alert(action, price, strength)
                        self.last_signal = signal
                
                # Wait for next check
                time.sleep(self.config["check_interval"])
                
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
            self.generate_report()
        except Exception as e:
            print(f"\n❌ Error in bot execution: {e}")
            self.generate_report()
    
    def generate_report(self):
        """Generate a summary report"""
        if not self.signals_history:
            print("No signals generated during this session")
            return
        
        print(f"\n{'='*60}")
        print("📊 TRADING SESSION REPORT")
        print(f"{'='*60}")
        
        # Calculate statistics
        buy_signals = [s for s in self.signals_history if s['signal'] == 1]
        sell_signals = [s for s in self.signals_history if s['signal'] == -1]
        
        print(f"Total signals: {len(self.signals_history)}")
        print(f"Buy signals: {len(buy_signals)}")
        print(f"Sell signals: {len(sell_signals)}")
        print(f"Optimal EMAs used: {self.optimal_short_ema}/{self.optimal_long_ema}")
        print(f"Best expected return: {self.best_return:.1f}%")
        
        if buy_signals:
            avg_buy_strength = sum(s['strength'] for s in buy_signals) / len(buy_signals)
            print(f"Average buy strength: {avg_buy_strength:.2f}%")
        
        if sell_signals:
            avg_sell_strength = sum(s['strength'] for s in sell_signals) / len(sell_signals)
            print(f"Average sell strength: {avg_sell_strength:.2f}%")
        
        print(f"\nLast 5 signals:")
        for signal in self.signals_history[-5:]:
            action = "BUY" if signal['signal'] == 1 else "SELL" if signal['signal'] == -1 else "HOLD"
            print(f"  {signal['timestamp'].strftime('%H:%M:%S')} - {action} at ${signal['price']:.2f} (EMAs: {signal['short_ema']}/{signal['long_ema']})")

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Merge configurations
    full_config = {**EMAIL_CONFIG, **TRADING_CONFIG}
    
    # Initialize and run bot
    bot = IXHLTradingBot(full_config)
    bot.run_bot()