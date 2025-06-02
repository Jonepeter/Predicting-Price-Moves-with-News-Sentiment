import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class SentimentAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        """
            Preprocesses text by lowercasing, removing punctuation, tokenizing,
            removing stopwords, and lemmatizing.

            Parameters:
                text (str): Input text to preprocess.

            Returns:
                str: Preprocessed text as a single string.
        """
        
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens if token not in self.stop_words and token.isalpha()]
        # Lemmatize tokens
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        # Join tokens back into a single string
        return ' '.join(tokens)

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
                # Preprocess the headline
                cleaned_headline = self.preprocess_text(headline)
                # Calculate sentiment polarity on cleaned text
                if cleaned_headline:  # Check if cleaned text is non-empty
                    return TextBlob(cleaned_headline).sentiment.polarity
                return 0.0  # Return neutral sentiment for empty cleaned text

            data['Sentiment'] = data['headline'].apply(get_sentiment)
            return data
        except Exception as e:
            print(f"Error calculating sentiment: {str(e)}")
            return None