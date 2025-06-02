import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
import pynance as pn
import matplotlib.pyplot as plt
import glob
import os
import logging as logger

# Load stock price data using yfinance (example: Apple stock)
def fetch_data():
    """
    Fetch stock Yahoo Finance data from local CSV files in the yfinance folder.
    
    Returns:
        bool: True if data is successfully loaded and validated, False otherwise.
    """
    try:
        logger.info("--------- Fetching data for Stock Price Analysis -----------")
        data_folder = "../data/yfinance_data/yfinance_data/"
        csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
        if not csv_files:
            logger.error(f"No CSV files found in {data_folder}")
            return False
        list_annotation = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']
    
        if len(csv_files) != len(list_annotation):
            logger.warning("Mismatch between number of files and tickers!")
        list_data = []
        for file, ticker in zip(csv_files, list_annotation):
            df = pd.read_csv(file)
            df['Ticker'] = ticker
            list_data.append(df)
        combined_data = pd.concat(list_data, ignore_index=True)
        logger.info(f"Combined {len(list_data)} files into one DataFrame with {len(combined_data)} rows.")
        data = combined_data
        logger.info("Data loaded successfully")
        return data
    except Exception as e:
        return f"Error fetching data: {str(e)}"

# Calculate technical indicators using TA-Lib
def calculate_technical_indicators(data):
    print("Calculating technical indicators...")
    try:
        close = np.array(data['Close'], dtype=float)
    
        # Simple Moving Average (SMA)
        data['SMA20'] = ta.SMA(close, timeperiod=20)
        
        # Relative Strength Index (RSI)
        data['RSI'] = ta.RSI(close, timeperiod=14)
        
        # MACD
        data['MACD'], data['MACD_Signal'], _ = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        return data
    except Exception as e:
        print(f"Error: {e}")
    

# Calculate financial metrics using PyNance
def calculate_financial_metrics(data):
    print("Calculating financial metrics...")
    try:
        # Daily returns
        daily_returns = pn.data.returns(data['Close'])
        
        # Annualized volatility
        volatility = pn.volatility.annualized(daily_returns, periods=252)
        
        # Sharpe Ratio (assuming risk-free rate of 0.02)
        sharpe_ratio = pn.sharpe_ratio(daily_returns, periods=252, rf_rate=0.02)
        
        # Maximum Drawdown
        max_drawdown = pn.max_drawdown(data['Close'])
        
        metrics = {
            'Annualized Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown
        }
        return metrics
    except Exception as e:
        print(f"Error: {e}")
    

# Visualize the data
def visualize_data(data, ticker='AAPL'):
    print("Generating visualizations...")
    try:
        plt.figure(figsize=(12, 8))
    
        # Plot Close price and SMA
        plt.subplot(3, 1, 1)
        plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
        plt.plot(data['Date'], data['SMA20'], label='20-day SMA', color='orange')
        plt.title(f'{ticker} Stock Price and 20-day SMA')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        
        # Plot RSI
        plt.subplot(3, 1, 2)
        plt.plot(data['Date'], data['RSI'], label='RSI (14)', color='green')
        plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
        plt.axhline(y=30, color='b', linestyle='--', label='Oversold (30)')
        plt.title('Relative Strength Index (RSI)')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True)
        
        # Plot MACD
        plt.subplot(3, 1, 3)
        plt.plot(data['Date'], data['MACD'], label='MACD', color='purple')
        plt.plot(data['Date'], data['MACD_Signal'], label='Signal Line', color='red')
        plt.title('MACD')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
