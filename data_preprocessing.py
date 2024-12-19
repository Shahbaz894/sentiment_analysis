import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_data(text):
    """Clean and preprocess review text."""
    try:
        if not isinstance(text, str):
            return ""
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        # Remove special characters and lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        # Tokenize, remove stopwords, and lemmatize
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ""


def preprocess_data(input_file, output_file):
    """Load, clean, encode, and save preprocessed data."""
    try:
        # Load data
        data = pd.read_csv(input_file)

        # Check for required columns
        if 'review' not in data.columns or 'sentiment' not in data.columns:
            raise ValueError("Input file must contain 'review' and 'sentiment' columns.")

        # Handle missing values
        data.dropna(subset=['review', 'sentiment'], inplace=True)

        # Clean the review text
        data['review'] = data['review'].apply(clean_data)

        # Encode sentiment labels
        label_encoder = LabelEncoder()
        data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

        # Save preprocessed data
        data.to_csv(output_file, index=False)
        print(f"Preprocessed data saved to {output_file}")

        return data, label_encoder
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Example usage
    input_path = "IMDB Dataset.csv"  # Input dataset file path
    output_path = "clean_data.csv"   # Path to save cleaned data

    processed_data, encoder = preprocess_data(input_path, output_path)

    # Display a sample of the cleaned data
    if processed_data is not None:
        print(processed_data.head())
