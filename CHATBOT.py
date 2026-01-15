# Import necessary libraries
import nltk
import random
import string
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data files for tokenization and lemmatization
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Read the dataset (text file) and convert content to lowercase
with open('data.txt', 'r', errors='ignore') as file:
    raw_data = file.read().lower()

# Tokenize the dataset into sentences and words
sent_tokens = nltk.sent_tokenize(raw_data)   # Sentence-level tokenization
word_tokens = nltk.word_tokenize(raw_data)   # Word-level tokenization

# Function to lemmatize tokens (reduce words to their base form)
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# Function to normalize text: lowercase, remove punctuation, tokenize, and lemmatize
def normalize_text(text):
    return lemmatize_tokens(
        nltk.word_tokenize(
            text.lower().translate(str.maketrans('', '', string.punctuation))
        )
    )

# Predefined greeting inputs and responses
greeting_inputs = ("hello", "hi", "hey")
greeting_responses = ["Hi!", "Hello!", "Hey there!", "Hi, how can I help you?"]

# Function to check if user input is a greeting
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_responses)
    return None

# Function to generate chatbot response using TF-IDF and cosine similarity
def chatbot_response(user_input):
    # Add user input temporarily to sentence tokens
    sent_tokens.append(user_input)
    
    # Convert text into TF-IDF vectors (with normalization and stopword removal)
    vectorizer = TfidfVectorizer(tokenizer=normalize_text, stop_words='english')
    tfidf = vectorizer.fit_transform(sent_tokens)
    
    # Compute similarity between user input and all sentences
    similarity = cosine_similarity(tfidf[-1], tfidf)
    index = similarity.argsort()[0][-2]   # Get index of most similar sentence
    similarity_score = similarity[0][index]
    
    # Remove user input from sentence tokens (to avoid duplication)
    sent_tokens.pop()
    
    # Return appropriate response based on similarity score
    if similarity_score == 0:
        return "Sorry, I did not understand that."
    else:
        return sent_tokens[index]

# ---------------- MAIN CHAT LOOP ----------------
print("CODTECH Chatbot: Hello! Type 'bye' to exit.")

while True:
    # Take user input
    user_input = input("You: ").lower()
    
    # Exit condition
    if user_input == 'bye':
        print("CODTECH Chatbot: Goodbye!")
        break
    else:
        # Check if input is a greeting
        greet = greeting(user_input)
        if greet:
            print("CODTECH Chatbot:", greet)
        else:
            # Generate chatbot response
            print("CODTECH Chatbot:", chatbot_response(user_input))
