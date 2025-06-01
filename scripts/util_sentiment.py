import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,wordpunct_tokenize
from datetime import datetime


# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords') 
nltk_path = "../data"
if nltk_path not in nltk.data.path:
    nltk.data.path.append(nltk_path)

class MovieNewsSentiment:
    """
    A class to analyze descriptive statistics of news article data,
    including headline lengths, article count by publisher, trends
    in publication dates and others
    """

    def __init__(self):
        """
        Initialize the MovieNewsSentiment with the path to the dataset.
        """
        self.data = None
        self.headlines = []
        self.cleaned_headlines = []

    def load_data(self,file_path, date_columns):
        """
        Load the dataset into a pandas DataFrame with error handling.
        
        Args:
            file_path: 
        """
        try:
            self.data = pd.read_csv(file_path, parse_dates=[date_columns])
            print("Data loaded successfully.")
            return self.data
        except FileNotFoundError:
            print("Error: File not found.")
        except pd.errors.EmptyDataError:
            print("Error: The file is empty.")
        except Exception as e:
            print(f"Unexpected error occurred while loading data: {e}")

    def analyze_headline_lengths(self):
        """
        Analyze and print statistics of headline text lengths.
        """
        if self.data is None:
            print("Data not loaded. Please call load_data() first.")
            return

        try:
            self.data['headline_length'] = self.data['headline'].astype(str).str.len()
            stats = self.data['headline_length'].describe()
            print("\n Headline Length Statistics:\n", stats.astype('int64'))
        except KeyError:
            print("Error: 'headline' column not found.")
        except Exception as e:
            print(f"Unexpected error in headline length analysis: {e}")

    def count_articles_by_publisher(self):
        """
        Count and display the number of articles by each publisher.
        """
        if self.data is None:
            print("Data not loaded. Please call load_data() first.")
            return

        try:
            publisher_counts = self.data['publisher'].value_counts()
            print("\nArticles per Publisher:\n", publisher_counts)

            # # Publisher by number of articles Visualization 
            # plt.figure(figsize=(10, 5))
            # sns.barplot(x=publisher_counts.index, y=publisher_counts.values)
            # plt.xticks(rotation=45, ha='right')
            # plt.title('Articles per Publisher')
            # plt.xlabel('Publisher')
            # plt.ylabel('Number of Articles')
            # plt.tight_layout()
            # plt.show()
        except KeyError:
            print("Error: 'publisher' column not found.")
        except Exception as e:
            print(f"Unexpected error in publisher count: {e}")  
        
    # 
    def analyze_publication_dates(self):
        """
        Analyze trends in publication dates and day-of-week frequencies.
        """
        if self.data is None:
            print("Data not loaded. Please call load_data() first.")
            return

        try:
            self.data['date'] = pd.to_datetime(self.data['date'], format = "ISO8601")

            # Frequency by date
            daily_counts = self.data['date'].dt.date.value_counts().sort_index()
            print("\nArticles per Date:\n", daily_counts)

            plt.figure(figsize=(10, 4))
            daily_counts.plot(marker='o')
            
            plt.title('Articles Over Time')
            plt.xlabel('Date')
            plt.ylabel('Number of Articles')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Frequency by weekday
            self.data['day_of_week'] = self.data['date'].dt.day_name()
            weekday_counts = self.data['day_of_week'].value_counts()
            print("\nArticles by Day of the Week:\n", weekday_counts)

            plt.figure(figsize=(7, 4))
            sns.barplot(x=weekday_counts.index, y=weekday_counts.values)
            plt.title('Articles by Day of the Week')
            plt.xlabel('Day')
            plt.ylabel('Number of Articles')
            plt.tight_layout()
            plt.show()
        except KeyError:
            print("Error: 'date' column not found.")
        except Exception as e:
            print(f"Unexpected error in date analysis: {e}")
    
    
    def preprocess_text(self):
        """
        Clean and tokenize the headlines.
        """
        try:
            stop_words = set(stopwords.words('english'))
            self.cleaned_headlines = []

            for line in self.data['headline'].tolist():
                # Lowercase, remove punctuation/numbers
                text = re.sub(r'[^a-zA-Z\s]', '', line.lower())
                tokens = wordpunct_tokenize(text)
                filtered = [w for w in tokens if w not in stop_words and len(w) > 2]
                self.cleaned_headlines.append(" ".join(filtered))

            print("Preprocessing completed.")
        except Exception as e:
            print(f"Error in text preprocessing: {e}")
            
    def extract_keywords(self, top_n=10):
        """
        Extract and display top keywords using TF-IDF.
        
        Args:
            top_n (int): Number of top keywords to display.
        """
        try:
            tfidf = TfidfVectorizer(max_df=0.85, min_df=1, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(self.cleaned_headlines)
            feature_names = tfidf.get_feature_names_out()
            summed_tfidf = tfidf_matrix.sum(axis=0)

            keyword_scores = [(feature_names[i], summed_tfidf[0, i]) for i in range(len(feature_names))]
            sorted_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)[:top_n]

            print("\nTop Keywords (TF-IDF):")
            for keyword, score in sorted_keywords:
                print(f"{keyword}: {score:.4f}")
        except Exception as e:
            print(f"Error extracting keywords: {e}")

    def perform_topic_modeling(self, num_topics=30, num_words=5):
        """
        Perform topic modeling using Latent Dirichlet Allocation (LDA).
        
        Args:
            num_topics (int): Number of topics to identify.
            num_words (int): Number of words per topic.
        """
        try:
            tf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=1, stop_words='english')
            tf = tf_vectorizer.fit_transform(self.cleaned_headlines)

            lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, random_state=42)
            lda.fit(tf)

            print(f"\nTop {num_topics} Topics:")
            for topic_idx, topic in enumerate(lda.components_):
                top_features = topic.argsort()[:-num_words - 1:-1]
                topic_keywords = [tf_vectorizer.get_feature_names_out()[i] for i in top_features]
                print(f"Topic {topic_idx + 1}: {' | '.join(topic_keywords)}")
        except Exception as e:
            print(f"Error in topic modeling: {e}")
            
    def publication_by_day(self):
        """
        Analyze and plot the number of articles published per day.
        """
        try:
            daily_counts = self.data['date'].dt.day.value_counts().sort_index()

            plt.figure(figsize=(10, 5))
            daily_counts.plot(kind='line', marker='o')
            plt.title('Articles Published per Day')
            plt.xlabel('Date')
            plt.ylabel('Article Count')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in daily publication analysis: {e}")

    def publication_by_hour(self):
        """
        Analyze and plot the number of articles published per hour of the day.
        """
        try:
            hourly_counts = self.data['date'].dt.hour.value_counts().sort_index()

            plt.figure(figsize=(8, 4))
            hourly_counts.plot(kind='bar', color='skyblue')
            plt.title('Articles Published by Hour of Day')
            plt.xlabel('Hour of Day')
            plt.ylabel('Article Count')
            plt.xticks(range(0, 24))
            plt.grid(axis='y')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in hourly publication analysis: {e}")
    
    # instance method used to find top publishers 
    def top_publishers(self, top = 10):
        """_summary_

        Args:
            top (int, optional): Enter number of top publishers you want. 
                                  Defaults to 10.
        
        Return:
                top publishers dataframe
                
        """     
        try:
            publisher_count = self.data['publisher'].value_counts().head(top)
            print(f"----------- Top {top} ------------")
            print(str(publisher_count).center(20))
            
        except Exception as e:
            return f"Error: {e}"
    
    def extract_email_domain(self):
        """
            Extract email domain from the publishers detail and
            display number of article in each domain
        """        
        try:
            email_domain = self.data['publisher'].apply(lambda x : x if re.match(r'[^@]+@[^@]+\.[^@]+',x) else None)
            domains = email_domain.apply(lambda x:str(x).split('@')[-1])
            # domain_count = domains.value_counts()
            print(domains.value_counts()) 
        except Exception as e:
            print(f"Error:  {e}")
