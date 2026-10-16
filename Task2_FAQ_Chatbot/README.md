# Task 2: AI FAQ Chatbot

An intelligent, NLP-driven FAQ Chatbot wrapped in an authentic macOS iMessage conversational interface. It matches user inquiries against an extensible knowledge base (`faq_data.json`) using TF-IDF vectorization and cosine similarity.

## ✨ Features

- **NLP Preprocessing Pipeline**: Lowercasing, punctuation stripping, tokenization, and WordNet lemmatization.
- **TF-IDF & Cosine Similarity**: Converts preprocessed queries into semantic vector spaces, finding the most relevant response using a cosine similarity threshold (> 0.2).
- **Graceful Fallbacks**: Provides helpful topic suggestions whenever queries fall below the confidence boundary.
- **macOS iMessage Theme**: Complete with traffic lights, floating message bubbles, avatar indicators, and animated typing ellipses.
- **Interactive Suggestion Chips**: Quick-access question buttons for return policies, shipping, order tracking, and cancellations.
- **Automated Test Suite**: Unit tests covering exact match, semantic queries, and edge cases.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python3 app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

### 3. Run Tests
```bash
python3 test_chatbot.py
```

## 🛠️ Tech Stack

- **Backend**: Python 3, Flask
- **NLP & Machine Learning**: NLTK, Scikit-Learn (TfidfVectorizer, cosine_similarity), NumPy
- **Frontend**: HTML5, Vanilla CSS3, JavaScript (Fetch API)
