import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import talib
from talib import abstract
import numpy as np
import logging
import glob
from datetime import datetime, timedelta
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPriceAnalyzer:
    """
    A class for performing quantitative analysis on stock price using TA-Lib and visualization.
    """
    
    def __init__(self):
        """
        Initialize the StockAnalyzer with a ticker symbol and date range.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in 'YYYY-MM-DD' format (default: 1 year ago)
            end_date (str): End date in 'YYYY-MM-DD' format (default: today)
        """
        self.data = None
        self.indicators = {}
        
    def fetch_data(self):
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
            self.data = combined_data
            print(self.data.head())

            return f"Successfully fetched {len(self.data)} rows of data"

        except Exception as e:
            return f"Error fetching data: {str(e)}"
        
        
    def calculate_sma(self, period = 20):
        try:
            sma = talib.SMA(self.data['Close'], timeperiod=period)
            plt.plot(sma, label = f"SMA {period} Days")
            plt.title("Stock Prices and SMA ")
            plt.legend()
            plt.grid()
            plt.show()
            return sma
        except Exception as e:
            print(f"Error calculating SMA: {e}") 
            return None
        
        
        
    def calculate_indicators(self):
        """
        Calculate various technical ind
        icators using TA-Lib.
        """
        if self.data is None:
            logger.error("No data available to calculate indicators")
            return False
            
        try:
            # Convert DataFrame to TA-Lib format (numpy arrays)
            open_prices = self.data['Open'].values
            high_prices = self.data['High'].values
            low_prices = self.data['Low'].values
            close_prices = self.data['Close'].values
            volume = self.data['Volume'].values
            
            # Calculate moving averages
            self.indicators['SMA_20'] = talib.SMA(close_prices, timeperiod=20)
            self.indicators['SMA_50'] = talib.SMA(close_prices, timeperiod=50)
            self.indicators['SMA_200'] = talib.SMA(close_prices, timeperiod=200)
            
            # Calculate RSI
            self.indicators['RSI_14'] = talib.RSI(close_prices, timeperiod=14)
            
            # Calculate MACD
            macd, macdsignal, macdhist = talib.MACD(close_prices)
            self.indicators['MACD'] = macd
            self.indicators['MACD_Signal'] = macdsignal
            self.indicators['MACD_Hist'] = macdhist
            
            # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(close_prices)
            self.indicators['BB_Upper'] = upper
            self.indicators['BB_Middle'] = middle
            self.indicators['BB_Lower'] = lower
            
            # Calculate Stochastic Oscillator
            slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
            self.indicators['Stoch_Slowk'] = slowk
            self.indicators['Stoch_Slowd'] = slowd
            
            logger.info("Successfully calculated technical indicators")
            return pd.DataFrame(self.indicators)
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return False
    
    def visualize_data(self, ticker='STOCK'):
        """
        Create visualizations of the stock data and indicators.
        
        Args:
            ticker (str): Stock ticker symbol for title (default: 'STOCK')
        """
        if self.data is None or not self.indicators:
            logger.error("No data or indicators available for visualization")
            return False
            
        try:
            
            fig, axes = plt.subplots(5, 1, figsize=(14, 20), gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]})
            
            # Price and Moving Averages
            axes[0].plot(self.data.index, self.data['Close'], label='Close Price', color='blue', alpha=0.5)
            axes[0].plot(self.data.index, self.indicators['SMA_20'], label='20-day SMA', color='orange')
            axes[0].plot(self.data.index, self.indicators['SMA_50'], label='50-day SMA', color='green')
            axes[0].plot(self.data.index, self.indicators['SMA_200'], label='200-day SMA', color='red')
            
            # Plot Bollinger Bands
            axes[0].plot(self.data.index, self.indicators['BB_Upper'], label='BB Upper', color='gray', linestyle='--', alpha=0.7)
            axes[0].plot(self.data.index, self.indicators['BB_Middle'], label='BB Middle', color='gray', alpha=0.7)
            axes[0].plot(self.data.index, self.indicators['BB_Lower'], label='BB Lower', color='gray', linestyle='--', alpha=0.7)
            
            axes[0].set_title(f'{ticker} Price and Indicators')
            axes[0].legend()
            axes[0].grid(True)
            
            # Volume and OBV
            axes[1].bar(self.data.index, self.data['Volume'], color='blue', alpha=0.3, label='Volume')
            axes[1].plot(self.data.index, self.indicators['OBV'], color='purple', label='OBV')
            axes[1].set_title('Volume and On-Balance Volume')
            axes[1].legend()
            axes[1].grid(True)
            
            # RSI
            axes[2].plot(self.data.index, self.indicators['RSI_14'], label='RSI 14', color='purple')
            axes[2].axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought')
            axes[2].axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold')
            axes[2].set_title('Relative Strength Index (RSI)')
            axes[2].set_ylim(0, 100)
            axes[2].legend()
            axes[2].grid(True)
            
            # MACD
            axes[3].plot(self.data.index, self.indicators['MACD'], label='MACD', color='blue')
            axes[3].plot(self.data.index, self.indicators['MACD_Signal'], label='Signal Line', color='orange')
            axes[3].bar(self.data.index, self.indicators['MACD_Hist'], label='MACD Histogram', color='gray', alpha=0.5)
            axes[3].set_title('MACD')
            axes[3].legend()
            axes[3].grid(True)
            
            # Stochastic Oscillator
            axes[4].plot(self.data.index, self.indicators['Stoch_Slowk'], label='Stoch %K', color='blue')
            axes[4].plot(self.data.index, self.indicators['Stoch_Slowd'], label='Stoch %D', color='orange')
            axes[4].axhline(80, color='red', linestyle='--', alpha=0.5)
            axes[4].axhline(20, color='green', linestyle='--', alpha=0.5)
            axes[4].set_title('Stochastic Oscillator')
            axes[4].set_ylim(0, 100)
            axes[4].legend()
            axes[4].grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{ticker}_technical_analysis.png')
            plt.close()
            
            logger.info("Successfully created visualizations")
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            return False
    
    def analyze(self, ticker='STOCK'):
        """
        Run the complete analysis pipeline.
        
        Args:
            ticker (str): Stock ticker symbol for visualization (default: 'STOCK')
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.data is None:
                logger.error("No data available for analysis")
                return False
                
            if not self.calculate_indicators():
                return False
                
            if not self.visualize_data(ticker):
                return False
                
            logger.info("Analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            return False