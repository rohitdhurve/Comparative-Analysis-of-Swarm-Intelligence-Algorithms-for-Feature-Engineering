import pandas as pd
import numpy as np
import pandas_ta as ta

# Read the CSV file
df = pd.read_csv('^NSEI.csv')
df['Volume'] = df['Volume'].astype(float)
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

stock_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
date_columns = ["Date"]

# Apply log transformation to the selected columns
df[stock_columns] = np.log1p(df[stock_columns] + 1e-6)
# Define the targets (y)
targets = ["Open", "Close"]  # "Open" and "Close" as targets


# Function to calculate technical indicators
def add_technical_indicators(data):
    data.ta.rsi(length=14, append=True)

    # Calculate MACD
    short_window = 12
    long_window = 26
    signal_window = 9
    short_ema = data['Close'].ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    data['MACD_12_26'] = macd
    data['MACD_12_26_9'] = signal_line

    # Calculate ATR
    atr_window = 14
    tr1 = data['High'] - data['Low']
    tr2 = abs(data['High'] - data['Close'].shift(1))
    tr3 = abs(data['Low'] - data['Close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1)
    true_range = np.max(tr, axis=1)
    atr = true_range.rolling(window=atr_window).mean()
    data['ATR_14'] = atr

    # Additional technical indicators
    data.ta.ema(length=50, append=True)
    data.ta.ema(length=200, append=True)
    data.ta.sma(length=20, append=True)
    data.ta.sma(length=50, append=True)
    data.ta.sma(length=200, append=True)
    data.ta.ema(length=12, append=True)
    data.ta.ema(length=26, append=True)

    return data


# Add technical indicators
df = add_technical_indicators(df)

# Define the new features with technical indicators
technical_indicators = ["RSI_14", "MACD_12_26", "MACD_12_26_9", "EMA_50", "EMA_200", "ATR_14", "SMA_20", "SMA_50",
                        "SMA_200", "EMA_12", "EMA_26"]
features = technical_indicators  # Combine original stock columns with technical indicators

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

# Remove null values from train_data
train_data = train_data.dropna()

print(train_data)
# Remove null values from test_data
test_data = test_data.dropna()
print(test_data)