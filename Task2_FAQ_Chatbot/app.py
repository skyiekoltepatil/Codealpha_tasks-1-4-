import json
import string
import numpy as np
from flask import Flask, request, jsonify, render_template

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ensure NLTK data is downloaded
import os
os.environ['NLTK_ALLOW_PROXIED_URLOPEN'] = '1'

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

app = Flask(__name__)

# Load FAQ data
with open('faq_data.json', 'r') as f:
    faq_data = json.load(f)

faq_questions = [item['question'] for item in faq_data]
faq_answers = [item['answer'] for item in faq_data]

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Preprocess all FAQ questions
preprocessed_questions = [preprocess(q) for q in faq_questions]

# Initialize and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    if not user_input:
        return jsonify({"answer": "Please ask a question."})

    preprocessed_input = preprocess(user_input)
    # Handle empty input after preprocessing
    if not preprocessed_input.strip():
        return jsonify({"answer": "Please ask a valid question."})
        
    input_vector = vectorizer.transform([preprocessed_input])
    
    # Compute cosine similarity
    similarities = cosine_similarity(input_vector, tfidf_matrix)[0]
    
    # Find the best match
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[best_match_idx]
    
    # Threshold for matching
    if best_match_score > 0.2:
        response = faq_answers[best_match_idx]
        matched = faq_questions[best_match_idx]
    else:
        response = "I'm sorry, I don't understand your question. Here are some topics I know about: returning items, tracking orders, international shipping, contacting support, payment methods, password reset, and order cancellation."
        matched = None
        
    return jsonify({
        "answer": response,
        "score": float(best_match_score),
        "matched_question": matched
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
