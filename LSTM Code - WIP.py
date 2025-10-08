import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import sys
import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# --------------------------
# CONFIGURATION
# --------------------------
# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Data parameters
TICKER = "SOL-USD"
START_DATE = "2023-01-01"
END_DATE = "2024-01-01"
FORECAST_DAYS = 30
EXTENDED_END_DATE = "2024-02-01"  # 30 days after END_DATE

# Volatility parameters for crypto-like predictions - UPDATED VALUES
VOLATILITY_SCALE = 1  # 2% base volatility (more realistic for crypto)
GARCH_ALPHA = 50      # GARCH model parameters
GARCH_BETA = 20
JUMP_PROBABILITY = 0.03   # Probability of large price jumps (slightly higher)
JUMP_SCALE = 0.05       # 5% size of price jumps

# Data split
TRAIN_VAL_SPLIT_RATIO = 0.8
TRAIN_RATIO_WITHIN_TRAINVAL = 0.8

# Optuna parameters
N_TRIALS = 2  # Number of optimization trials
TIMEOUT = 1800  # Stop study after 30 minutes
PRUNING = True  # Enable pruning

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------
# CRYPTO VOLATILITY FUNCTIONS
# --------------------------
def calculate_garch_volatility(returns, alpha=0.2, beta=0.7):
    """Calculate GARCH volatility for crypto-like fluctuations"""
    n = len(returns)
    volatility = np.zeros(n)
    volatility[0] = np.std(returns) if len(returns) > 0 else 0.02
    
    for i in range(1, n):
        volatility[i] = np.sqrt(0.1 + alpha * returns[i-1]**2 + beta * volatility[i-1]**2)
    
    return volatility

def add_crypto_volatility(predictions, historical_data, volatility_scale=VOLATILITY_SCALE):
    """Add crypto-like volatility to predictions"""
    # Calculate historical volatility
    returns = np.diff(historical_data) / historical_data[:-1]
    hist_vol = np.std(returns) if len(returns) > 0 else 0.02
    
    # Initialize GARCH-like volatility
    n_predictions = len(predictions)
    crypto_volatility = np.zeros(n_predictions)
    crypto_volatility[0] = hist_vol * volatility_scale
    
    # Add volatility clustering (GARCH effect)
    for i in range(1, n_predictions):
        # Random shock with crypto-like properties
        shock = np.random.normal(0, crypto_volatility[i-1])
        
        # Update volatility (GARCH-like process)
        crypto_volatility[i] = crypto_volatility[i-1] * (0.9 + 0.1 * np.random.random())
        
        # Add the shock to the prediction
        predictions[i] = predictions[i] * (1 + shock)
        
        # Occasionally add large jumps (crypto characteristic)
        if np.random.random() < JUMP_PROBABILITY:
            jump_direction = 1 if np.random.random() > 0.5 else -1
            jump_size = JUMP_SCALE * (1 + np.random.exponential(0.5))
            predictions[i] = predictions[i] * (1 + jump_direction * jump_size)
    
    return predictions

def apply_fat_tails(predictions, alpha=1.7):
    """Apply fat-tailed distribution (common in crypto)"""
    # Use Student's t-distribution for fat tails
    t_dist_returns = np.random.standard_t(alpha, len(predictions)) * 0.01
    
    for i in range(1, len(predictions)):
        predictions[i] = predictions[i] * (1 + t_dist_returns[i])
    
    return predictions

def add_autocorrelation(predictions, momentum_strength=0.15, mean_reversion_strength=0.05):
    """Add momentum/mean-reversion autocorrelation"""
    for i in range(2, len(predictions)):
        # Some momentum effect
        prev_return = (predictions[i-1] - predictions[i-2]) / predictions[i-2]
        
        # Add momentum with some mean reversion
        if abs(prev_return) > 0.03:  # Strong moves tend to reverse
            predictions[i] = predictions[i] * (1 - np.sign(prev_return) * mean_reversion_strength)
        else:  # Small moves tend to continue
            predictions[i] = predictions[i] * (1 + momentum_strength * prev_return)
    
    return predictions

def simulate_crypto_patterns(predictions, historical_prices):
    """Apply all crypto-like patterns to predictions"""
    # Start with the original predictions
    crypto_predictions = predictions.copy()
    
    # Add various crypto characteristics
    crypto_predictions = add_crypto_volatility(crypto_predictions, historical_prices)
    crypto_predictions = apply_fat_tails(crypto_predictions)
    crypto_predictions = add_autocorrelation(crypto_predictions)
    
    # Ensure no negative prices
    crypto_predictions = np.maximum(crypto_predictions, historical_prices[-1] * 0.1)
    
    return crypto_predictions

# --------------------------
# DATA LOADING AND PREPROCESSING
# --------------------------
def load_and_preprocess_data(ticker, start_date, end_date):
    """Load data and add technical indicators"""
    try:
        # Download data with auto_adjust explicitly set
        df = yf.download(ticker, start=start_date, end=end_date, 
                        progress=False, auto_adjust=True)
        
        # Handle different yfinance column formats
        if isinstance(df.columns, pd.MultiIndex):
            # New yfinance format: Flatten multi-index columns
            df.columns = ['_'.join(col).strip('_').lower() for col in df.columns]
        else:
            # Old yfinance format: Simple lowercase conversion
            df.columns = [str(col).lower() for col in df.columns]
        
        # Check for different column naming patterns and standardize
        column_mappings = [
            # Pattern 1: Standard lowercase names
            {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'},
            # Pattern 2: With ticker suffix (most common)
            {'open_sol-usd': 'open', 'high_sol-usd': 'high', 'low_sol-usd': 'low', 
             'close_sol-usd': 'close', 'volume_sol-usd': 'volume'},
            # Pattern 3: With ticker prefix
            {'sol-usd_open': 'open', 'sol-usd_high': 'high', 'sol-usd_low': 'low', 
             'sol-usd_close': 'close', 'sol-usd_volume': 'volume'},
        ]
        
        # Try each mapping pattern
        for mapping in column_mappings:
            # Check if all required columns exist in this mapping
            if all(col in df.columns for col in mapping.keys()):
                df = df.rename(columns=mapping)
                break
        else:
            # If no mapping worked, try to find the best match
            available_cols = df.columns.tolist()
            print(f"Available columns: {available_cols}")
            
            # Try to find the close price column (most important)
            close_col = None
            for col in available_cols:
                if 'close' in col.lower():
                    close_col = col
                    break
            
            if close_col is None:
                raise ValueError("Could not find close price column")
                
            # Create a simple mapping with the close column
            df = df.rename(columns={close_col: 'close'})
            
            # Try to find other columns
            for standard_col in ['open', 'high', 'low', 'volume']:
                for available_col in available_cols:
                    if standard_col in available_col.lower():
                        df = df.rename(columns={available_col: standard_col})
                        break
        
        # Verify we have at least the close column
        if 'close' not in df.columns:
            raise ValueError("Close price column not found after renaming")
            
        # Create missing columns if needed
        if 'open' not in df.columns:
            df['open'] = df['close'] * (0.99 + 0.02 * np.random.random(len(df)))
        if 'high' not in df.columns:
            df['high'] = df[['open', 'close']].max(axis=1) * (1 + 0.01 * np.random.random(len(df)))
        if 'low' not in df.columns:
            df['low'] = df[['open', 'close']].min(axis=1) * (1 - 0.01 * np.random.random(len(df)))
        if 'volume' not in df.columns:
            df['volume'] = 1000000 * (1 + np.random.random(len(df)))
        
        # Calculate technical indicators with more crypto-focused features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        
        # Add volatility-based features
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Add momentum indicators
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        df.dropna(inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# --------------------------
# DATA PREPARATION
# --------------------------
def create_dataset(df, window_size):
    """Create sequences with multiple features"""
    X, y = [], []
    # Use more features for better crypto pattern recognition
    features = ['close', 'returns', 'volatility', 'macd', 'high_low_ratio', 'volume_ratio', 'momentum_5']
    
    for i in range(len(df) - window_size):
        window = df[features].iloc[i:i+window_size].values
        last_price = df['close'].iloc[i + window_size - 1]
        next_price = df['close'].iloc[i + window_size]
        delta = (next_price - last_price) / last_price  # Percentage change
        X.append(window)
        y.append(delta)
    
    return np.array(X), np.array(y)

# --------------------------
# LSTM MODEL
# --------------------------
class PriceForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, 1)
        )
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# --------------------------
# TRAINING FUNCTION
# --------------------------
def train_model(model, train_loader, val_loader, epochs, device, trial=None):
    """Train the LSTM model with optional Optuna pruning"""
    optimizer = optim.Adam(model.parameters(), lr=model.learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).flatten()
            loss = criterion(outputs, batch_y)
            loss.backward()
            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).flatten()
                val_loss += criterion(outputs, batch_y).item()
        
        val_loss /= len(val_loader)
        
        # Report to Optuna if in trial
        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_loss

# --------------------------
# FUTURE PREDICTION WITH CRYPTO VOLATILITY
# --------------------------
def forecast_future(model, last_window, scaler, device, steps=30):
    """Predict future prices iteratively with crypto-like volatility"""
    predictions = []
    current_window = last_window.copy()
    
    # Get the last actual price to ensure continuity
    last_actual_price = current_window[-1, 0]
    
    # Get base prediction without volatility
    for i in range(steps):
        scaled_window = scaler.transform(current_window.reshape(-1, 7)).reshape(1, -1, 7)
        x_input = torch.tensor(scaled_window, dtype=torch.float32).to(device)
        
        model.eval()
        with torch.no_grad():
            pred_delta = model(x_input).item()
        
        last_price = current_window[-1, 0]
        
        # For the first prediction, start exactly from the last actual price
        if i == 0:
            pred_price = last_actual_price * (1 + pred_delta)
        else:
            pred_price = last_price * (1 + pred_delta)
        
        # Update the window with predicted values
        new_row = np.array([
            pred_price,  # close
            pred_delta,  # returns
            current_window[-1, 2],  # volatility (keep recent)
            current_window[-1, 3],  # macd (keep recent)
            current_window[-1, 4],  # high_low_ratio (keep recent)
            current_window[-1, 5],  # volume_ratio (keep recent)
            pred_delta   # momentum_5 (approximate)
        ])
        
        predictions.append(pred_price)
        current_window = np.vstack([current_window[1:], new_row])
    
    # Apply crypto-like volatility patterns to the predictions
    historical_prices = last_window[:, 0]  # Historical close prices
    crypto_predictions = simulate_crypto_patterns(np.array(predictions), historical_prices)
    
    # Ensure the first prediction starts exactly from the last actual price
    crypto_predictions[0] = last_actual_price * (1 + (crypto_predictions[0] - last_actual_price) / last_actual_price)
    
    return crypto_predictions.tolist()
# --------------------------
# MONTE CARLO SIMULATION FOR UNCERTAINTY
# --------------------------
def monte_carlo_forecast(model, last_window, scaler, device, steps=30, n_simulations=5):
    """Run multiple simulations to capture prediction uncertainty"""
    all_predictions = []
    
    for _ in range(n_simulations):
        # Reset random seed for each simulation to get different results
        np.random.seed(int(time.time() * 1000) % 2**32)
        
        predictions = forecast_future(model, last_window, scaler, device, steps)
        all_predictions.append(predictions)
    
    return np.array(all_predictions)

# --------------------------
# OPTUNA OPTIMIZATION
# --------------------------
def objective(trial):
    """Objective function for Optuna optimization"""
    # Suggest hyperparameters
    params = {
        'window_size': trial.suggest_categorical('window_size', [10, 20, 30, 50]),
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'epochs': trial.suggest_int('epochs', 30, 150),
    }
    
    # Prepare datasets with current window size
    X_train, y_train = create_dataset(train_df, params['window_size'])
    X_val, y_val = create_dataset(val_df, params['window_size'])
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 7)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, 7)).reshape(X_val.shape)
    
    # Create dataloaders
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ),
        batch_size=params['batch_size'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        ),
        batch_size=params['batch_size']
    )
    
    # Initialize model
    model = PriceForecaster(
        input_size=7,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(DEVICE)
    
    # Add learning rate to model for access in train_model
    model.learning_rate = params['learning_rate']
    
    # Train model with Optuna pruning
    val_loss = train_model(
        model, train_loader, val_loader, 
        params['epochs'], DEVICE, trial
    )
    
    return val_loss

def optimize_hyperparameters():
    """Run Optuna optimization"""
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=5) if PRUNING else None
    )
    
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)
    
    print("Number of finished trials: ", len(study.trials))
    
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Validation Loss: {trial.value:.6f}")
    
    print("  Best Hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return trial.params

# --------------------------
# MAIN EXECUTION
# --------------------------
if __name__ == "__main__":
    print(f"\n[1] Loading and preprocessing {TICKER} data from {START_DATE} to {END_DATE}...")
    df = load_and_preprocess_data(TICKER, START_DATE, END_DATE)
    
    if df is None:
        print("Failed to load data. Exiting.")
        sys.exit(1)
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Data available from {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Available columns: {df.columns.tolist()}")
    
    print("\n[2] Preparing datasets...")
    n = len(df)
    c = int(TRAIN_VAL_SPLIT_RATIO * n)
    b = int(TRAIN_RATIO_WITHIN_TRAINVAL * c)
    
    train_df = df.iloc[:b]
    val_df = df.iloc[b:c]
    test_df = df.iloc[c:]
    
    print(f"Training data: {len(train_df)} samples")
    print(f"Validation data: {len(val_df)} samples")
    print(f"Test data: {len(test_df)} samples")
    
    print("\n[3] Optimizing hyperparameters with Optuna...")
    try:
        best_params = optimize_hyperparameters()
    except Exception as e:
        print(f"Hyperparameter optimization failed: {e}")
        print("Using default parameters...")
        best_params = {
            'window_size': 30,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        }
    
    print("\n[4] Training final model with best parameters...")
    # Prepare datasets with optimal window size
    X_train, y_train = create_dataset(train_df, best_params['window_size'])
    X_val, y_val = create_dataset(val_df, best_params['window_size'])
    X_test, y_test = create_dataset(test_df, best_params['window_size'])
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 7)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, 7)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 7)).reshape(X_test.shape)
    
    # Create dataloaders
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ),
        batch_size=best_params['batch_size'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        ),
        batch_size=best_params['batch_size']
    )
    
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test_scaled, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        ),
        batch_size=best_params['batch_size']
    )
    
    # Initialize and train final model
    final_model = PriceForecaster(
        input_size=7,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(DEVICE)
    
    final_model.learning_rate = best_params['learning_rate']
    
    final_val_loss = train_model(
        final_model, train_loader, val_loader,
        best_params['epochs'], DEVICE
    )
    
    print(f"\nFinal Model Validation Loss: {final_val_loss:.6f}")
    
    print("\n[5] Generating future predictions with Monte Carlo simulation...")
    last_window = df[['close', 'returns', 'volatility', 'macd', 'high_low_ratio', 'volume_ratio', 'momentum_5']].iloc[-best_params['window_size']:].values
    
    # Run multiple simulations
    monte_carlo_predictions = monte_carlo_forecast(final_model, last_window, scaler, DEVICE, FORECAST_DAYS, 10)
    
    # Use median as our main prediction
    future_prices = np.median(monte_carlo_predictions, axis=0)
    future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, FORECAST_DAYS+1)]
    
    # Calculate prediction intervals
    upper_quantile = np.quantile(monte_carlo_predictions, 0.75, axis=0)
    lower_quantile = np.quantile(monte_carlo_predictions, 0.25, axis=0)
    
    print("\n[6] Fetching actual data for comparison...")
    actual_df = yf.download(TICKER, start=END_DATE, end=EXTENDED_END_DATE, progress=False)
    
    # Process actual data columns same way as before
    if isinstance(actual_df.columns, pd.MultiIndex):
        actual_df.columns = ['_'.join(col).strip('_').lower() for col in actual_df.columns]
    
    # Handle column names
    col_mapping = {
        'close': 'close',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'volume': 'volume',
        'adj close': 'close'
    }
    
    # Try to find the close price column
    close_col = None
    for col in actual_df.columns:
        if 'close' in col.lower():
            close_col = col
            break
    
    if close_col:
        actual_df = actual_df.rename(columns={close_col: 'close'})
    
    # Align dates and prices
    actual_prices = actual_df['close'].values if 'close' in actual_df.columns else []
    actual_dates = actual_df.index
    
    # Trim to only the forecast period
    if len(actual_prices) > 0:
        actual_prices = actual_prices[:FORECAST_DAYS]
        actual_dates = actual_dates[:FORECAST_DAYS]
    
    print("\n[7] Calculating prediction metrics...")
    if len(actual_prices) > 0 and len(actual_prices) == len(future_prices):
        mae = mean_absolute_error(actual_prices, future_prices)
        mse = mean_squared_error(actual_prices, future_prices)
        rmse = np.sqrt(mse)
        accuracy = 100 * (1 - (mae / np.mean(actual_prices)))
        
        # Calculate direction accuracy
        actual_direction = np.sign(np.diff(actual_prices))
        predicted_direction = np.sign(np.diff(future_prices))
        direction_accuracy = 100 * np.mean(actual_direction == predicted_direction) if len(actual_direction) > 0 else 0
        
        print(f"\nPrediction Accuracy Metrics:")
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"Price Level Accuracy: {accuracy:.2f}%")
        print(f"Direction Accuracy: {direction_accuracy:.2f}%")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Date': actual_dates,
            'Predicted': future_prices,
            'Actual': actual_prices,
            'Error ($)': future_prices - actual_prices,
            'Error (%)': 100 * (future_prices - actual_prices) / actual_prices
        })
        
        print("\nDetailed Comparison:")
        print(comparison_df.to_string(index=False, float_format="%.2f"))
    else:
        print(f"Warning: Could not get actual data for comparison")
        if len(actual_prices) == 0:
            print("No actual price data available")
        else:
            print(f"Got {len(actual_prices)} days of actual data (expected {FORECAST_DAYS})")
        actual_prices = []
        actual_dates = []
    
    print("\n[8] Plotting comparison results with crypto-like predictions...")
    plt.figure(figsize=(16, 10))

# Historical data
    plt.plot(df.index, df['close'], label='Historical Prices', color='blue', alpha=0.7, linewidth=2)

# Future predictions with confidence interval - start directly from last point
    plt.plot([df.index[-1]] + future_dates, [df['close'].iloc[-1]] + future_prices.tolist(), 
         'r-', label='Median Prediction', linewidth=3)

    plt.fill_between(future_dates, lower_quantile, upper_quantile, 
            alpha=0.3, color='red', label='25-75% Prediction Interval')

# Add individual simulation paths to show volatility
    for i in range(min(3, len(monte_carlo_predictions))):  # Show first 3 simulations
        plt.plot([df.index[-1]] + future_dates, [df['close'].iloc[-1]] + monte_carlo_predictions[i].tolist(), 
             'gray', alpha=0.3, linewidth=1)

# Actual data if available
    if len(actual_prices) > 0:
    # Plot actual data starting from the end of historical
        plt.plot([df.index[-1]] + actual_dates.tolist(), [df['close'].iloc[-1]] + actual_prices.tolist(), 
             'g-', label='Actual Prices', linewidth=3)

# Formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.title(f'{TICKER} Price Forecast with Crypto-like Volatility', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price (USD)', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

# Add metrics annotation if available
    if len(actual_prices) > 0:
        metrics_text = (f"Prediction Metrics:\n"
                   f"MAE: ${mae:.2f}\n"
                   f"RMSE: ${rmse:.2f}\n"
                   f"Direction Acc: {direction_accuracy:.1f}%")
        plt.annotate(metrics_text, xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()