import nltk
import random
import string
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required nltk data
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Read data file
with open('data.txt', 'r', errors='ignore') as file:
    raw_data = file.read().lower()

# Tokenization
sent_tokens = nltk.sent_tokenize(raw_data)
word_tokens = nltk.word_tokenize(raw_data)

# Lemmatization function
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def normalize_text(text):
    return lemmatize_tokens(nltk.word_tokenize(text.lower().translate(
        str.maketrans('', '', string.punctuation)
    )))

# Greeting responses
greeting_inputs = ("hello", "hi", "hey")
greeting_responses = ["Hi!", "Hello!", "Hey there!", "Hi, how can I help you?"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_responses)
    return None

# Generate response using TF-IDF
def chatbot_response(user_input):
    sent_tokens.append(user_input)
    
    vectorizer = TfidfVectorizer(tokenizer=normalize_text, stop_words='english')
    tfidf = vectorizer.fit_transform(sent_tokens)
    
    similarity = cosine_similarity(tfidf[-1], tfidf)
    index = similarity.argsort()[0][-2]
    similarity_score = similarity[0][index]
    
    sent_tokens.pop()
    
    if similarity_score == 0:
        return "Sorry, I did not understand that."
    else:
        return sent_tokens[index]

# Main loop
print("CODTECH Chatbot: Hello! Type 'bye' to exit.")

while True:
    user_input = input("You: ").lower()
    
    if user_input == 'bye':
        print("CODTECH Chatbot: Goodbye!")
        break
    else:
        greet = greeting(user_input)
        if greet:
            print("CODTECH Chatbot:", greet)
        else:
            print("CODTECH Chatbot:", chatbot_response(user_input))
