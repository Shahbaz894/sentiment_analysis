import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def preprocess_data(input_file, output_file):
    """Preprocess the IMDB dataset."""
    data = pd.read_csv(input_file)

    # Clean the data
    data['review'] = data['review'].fillna('')  # Handle missing values
    data['review'] = data['review'].str.replace(r'<[^>]*>', '', regex=True)  # Remove HTML tags
    data['review'] = data['review'].str.replace(r'[^a-zA-Z\s]', '', regex=True)  # Remove non-alphabetic characters
    data['review'] = data['review'].str.lower()  # Convert to lowercase

    # Save cleaned data
    data.to_csv(output_file, index=False)
    return data

def train_model(cleaned_data, tokenizer_file, model_file):
    """Train the LSTM model with additional layers and save tokenizer and model."""
    # Split data into training and labels
    reviews = cleaned_data['review']
    sentiments = cleaned_data['sentiment'].map({'positive': 1, 'negative': 0}).values  # Encode labels

    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)
    padded_sequences = pad_sequences(sequences, maxlen=150)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, sentiments, test_size=0.2, random_state=42)

    # Build enhanced model
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=150),  # Embedding layer
        Dropout(0.3),  # Dropout for regularization
        Bidirectional(LSTM(128, return_sequences=True)),  # Bidirectional LSTM
        Dropout(0.3),
        Bidirectional(LSTM(64)),  # Second Bidirectional LSTM
        Dropout(0.3),
        Dense(64, activation='relu'),  # Fully connected layer
        Dropout(0.3),
        Dense(32, activation='relu'),  # Another fully connected layer
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    # Compile the model
    optimizer = Adam(learning_rate=0.001)  # Adjusted learning rate
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=8, batch_size=32, validation_data=(X_test, y_test))

    # Save tokenizer and model
    with open(tokenizer_file, 'wb') as file:
        pickle.dump(tokenizer, file)
    model.save(model_file)
    print(f"Model saved to {model_file} and tokenizer saved to {tokenizer_file}")


# Example usage
# Preprocess data and save to a CSV file
cleaned_data = preprocess_data('IMDB Dataset.csv', 'clean_data.csv')

# Train the model using the preprocessed data
train_model(cleaned_data, 'models/tokenizer.pkl', 'models/sentiment_model.h5')
