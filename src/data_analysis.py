import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import Counter

# Load the dataset
def load_data(file_path):
    """Loads the dataset from the given file path."""
    return pd.read_csv('../data/raw_analyst_ratings.csv')

# Headline length analysis
def analyze_headline_length(data):
    """Analyzes and visualizes headline length statistics."""
    data['headline_length'] = data['headline'].apply(len)
    print(data['headline_length'].describe())

# Tokenize headlines and find common words
def analyze_common_words(data):
    """Finds and prints the most common words in the headlines."""
    all_words = word_tokenize(" ".join(data['headline'].astype(str)))
    common_words = Counter(all_words).most_common(10)
    print("Most common words:", common_words)

# Publisher article count analysis
def analyze_publishers(data):
    """Analyzes and visualizes the distribution of articles per publisher."""
    publisher_counts = data['publisher'].value_counts()
    publisher_counts.plot(kind='bar', figsize=(10, 6))
    plt.title("Articles per Publisher")
    plt.xlabel("Publisher")
    plt.ylabel("Number of Articles")
    plt.show()

# Publication date trends
def analyze_publication_dates(data):
    """Analyzes and visualizes publication trends over time."""
    data['date'] = pd.to_datetime(data['date'])
    daily_counts = data['date'].dt.date.value_counts().sort_index()
    daily_counts.plot(kind='line', figsize=(12, 6), title="Publication Frequency Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Articles")
    plt.show()

# Main function to run the EDA
def main():
    file_path = '../data/news_data.csv'  # Adjust path if needed
    data = load_data(file_path)

    # Perform analysis
    analyze_headline_length(data)
    analyze_common_words(data)
    analyze_publishers(data)
    analyze_publication_dates(data)

if __name__ == "__main__":
    main()
