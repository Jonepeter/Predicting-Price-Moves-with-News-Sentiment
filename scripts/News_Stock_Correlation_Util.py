import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
import logging as logger
from scipy.stats import pearsonr
import glob
import os

# Configure NLTK to use local punkt tokenizer
nltk.data.path.append('../data')

# Verify punkt tokenizer availability
try:
    nltk.data.find('tokenizers/punkt')
    print("Local punkt tokenizer found.")
except LookupError:
    print("Error: punkt tokenizer not found in ./nltk_data. Please download it using nltk.download('punkt', download_dir='./nltk_data').")
    exit(1)

class NewsStockCorrelationAnalyzer:
    def __init__(self):
        
        pass
    
    def load_news_data(self, file_path=None):
        """
        Loads FNSPID data from a local CSV file and converts the 'date' column to datetime with US/Eastern timezone.

        Parameters:
            file_path (str): Path to the FNSPID data CSV file. Defaults to 'news_data.csv'.

        Returns:
            pd.DataFrame: DataFrame containing FNSPID data, or None if an error occurs.
        """
        print(f"----------------Loading FNSPID data from {file_path}------------")
        try:
            data = pd.read_csv(file_path)
            data['date'] = pd.to_datetime(data['date'], format="ISO8601")
            return data
        except FileNotFoundError:
            print(f"Error: {file_path} not found.")
            return None
        except Exception as e:
            print(f"Error loading FNSPID data: {e}")
            return None

    def fetch_stock_data(self):
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
                df['Date'] = pd.to_datetime(df['Date'], format="ISO8601")
                df['Ticker'] = ticker
                list_data.append(df)

            combined_data = pd.concat(list_data, ignore_index=True)
            logger.info(f"Combined {len(list_data)} files into one DataFrame with {len(combined_data)} rows.")
            self.data = combined_data
            self.data = self.data.rename(columns=lambda x: x.lower().replace(' ', '_'))

            return self.data
        except Exception as e:
            return f"Error fetching data: {str(e)}"
    
    def align_datasets(self, news_data, stock_data):
        """
        Aligns FNSPID news and stock datasets by normalizing dates and merging on the 'Date' column.

        Parameters:
            news_data (pd.DataFrame): DataFrame containing news data with a 'Date' column.
            stock_data (pd.DataFrame): DataFrame containing stock data with a 'Date' column.

        Returns:
            pd.DataFrame: Merged DataFrame containing aligned news and stock data, or None if an error occurs.
        """
        print("Aligning datasets by date...")
        try:
            if news_data is None or stock_data is None:
                print("Error: One or both datasets are missing.")
                return None
            if 'date' not in news_data.columns or 'date' not in stock_data.columns:
                print("Error: 'Date' column missing in one or both datasets.")
                return None
            news_data['date'] = pd.to_datetime(news_data['date'], utc=True).dt.date
            stock_data['date'] = pd.to_datetime(stock_data['date'], utc=True).dt.date
            merged_data = pd.merge(news_data, stock_data, on='date', how='inner')
            return merged_data
        except Exception as e:
            print(f"Error aligning datasets: {str(e)}")
            return None

    def calculate_sentiment(self, data):
        """
            Performs sentiment analysis on headlines in the data using TextBlob and punkt tokenizer.

            Parameters:
                data (pd.DataFrame): DataFrame containing a 'headline' column for sentiment analysis.

            Returns:
                pd.DataFrame: DataFrame with an additional 'Sentiment' column, or None if an error occurs.
        """
        print("Calculating sentiment scores with punkt tokenizer...")
        try:
            if data is None or 'headline' not in data.columns:
                print("Error: Invalid or missing data/headline column.")
                return None
            def get_sentiment(headline):
                sentences = nltk.tokenize.wordpunct_tokenize(headline)
                sentiments = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
                return sum(sentiments) / len(sentiments) if sentiments else 0.0
            # Function to categorize polarity scores
            def categorize_sentiment(score):
                if isinstance(score, float) or isinstance(score, int):  # Ensure score is numeric
                    if score >= 0.05:
                        return 'positive'
                    elif score <= -0.05:
                        return 'negative'
                    else:
                        return 'neutral'
                return 'neutral'  # Handle non-numeric values
            data['Sentiment'] = data['headline'].apply(get_sentiment)
            data['sentiment_label'] = data['Sentiment'].apply(categorize_sentiment)
            return data
        except Exception as e:
            print(f"Error calculating sentiment: {str(e)}")
            return None

    def aggregate_sentiments(self, data):
        """
            Aggregates sentiment scores by date, computing the mean sentiment per day.

            Parameters:
                data (pd.DataFrame): DataFrame containing 'Date' and 'Sentiment' columns.

            Returns:
                pd.DataFrame: DataFrame with daily average sentiment scores, or None if an error occurs.
        """
        print("Aggregating daily sentiment scores...")
        try:
            if data is None or 'date' not in data.columns or 'Sentiment' not in data.columns:
                print("Error: Invalid or missing data/Date/Sentiment columns.")
                return None
            daily_sentiment = data.groupby('Date')['Sentiment'].mean().reset_index()
            return daily_sentiment
        except Exception as e:
            print(f"Error aggregating sentiments: {str(e)}")
            return None

    def calculate_daily_returns(self, data):
        """
        Calculates daily stock returns based on the 'Close' price column.

        Parameters:
            data (pd.DataFrame): DataFrame containing a 'Close' column with stock prices.

        Returns:
            pd.DataFrame: DataFrame with an additional 'Daily_Return' column, or None if an error occurs.
        """
        print("Calculating daily stock returns...")
        try:
            if data is None or 'close' not in data.columns:
                print("Error: Invalid or missing data/Close column.")
                return None
            data['Daily_Return'] = data['close'].pct_change() * 100
            return data
        except Exception as e:
            print(f"Error calculating daily returns: {str(e)}")
            return None

    def correlation_analysis(self, sentiment_data, stock_data):
        """
            Performs Pearson correlation analysis between sentiment scores and daily stock returns.

            Parameters:
                sentiment_data (pd.DataFrame): DataFrame containing 'Date' and 'Sentiment' columns.
                stock_data (pd.DataFrame): DataFrame containing 'Date' and 'Daily_Return' columns.

            Returns:
                tuple: (correlation coefficient, p-value, merged DataFrame), or (None, None, None) if an error occurs.
        """
        print("Performing correlation analysis...")
        try:
            if sentiment_data is None or stock_data is None:
                print("Error: One or both datasets are missing.")
                return None, None, None
            merged_data = pd.merge(sentiment_data, stock_data[['Date', 'Daily_Return']], on='Date', how='inner')
            merged_data = merged_data.dropna()
            
            if len(merged_data) < 2:
                print("Insufficient data for correlation analysis.")
                return None, None, None
            
            correlation, p_value = pearsonr(merged_data['Sentiment'], merged_data['Daily_Return'])
            return correlation, p_value, merged_data
        except Exception as e:
            print(f"Error in correlation analysis: {str(e)}")
            return None, None, None