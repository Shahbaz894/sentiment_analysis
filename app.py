import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = load_model('models/sentiment_model.h5')

# Load the saved tokenizer
with open('models/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Streamlit interface
st.title("Movie Review Sentiment Analysis")
st.markdown("**Enter a movie review to predict its sentiment:**")

# Input text area for the review
review = st.text_area("Movie Review", height=150, placeholder="Type your movie review here...")

# Predict sentiment when the button is clicked
if st.button("Predict Sentiment"):
    if review.strip():  # Check if the review is not empty
        # Preprocess the input review
        sequence = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequence, maxlen=150)
        
        # Make prediction
        prediction = model.predict(padded)[0][0]  # Extract the scalar prediction value
        
        # Determine sentiment
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        confidence = f"{prediction * 100:.2f}%"
        
        # Display results
        st.write(f"**Predicted Sentiment:** {sentiment}")
        st.write(f"**Confidence Level:** {confidence}")
    else:
        st.error("Please enter a movie review before predicting.")

# Add some additional information
st.markdown(
    """
    **How it works:**
    - This app uses a trained LSTM model to classify movie reviews as Positive or Negative.
    - The review is tokenized, padded, and passed to the model for prediction.
    """
)
