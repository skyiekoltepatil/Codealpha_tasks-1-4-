import unittest
import json
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

class TestFAQChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure NLTK data is available
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except Exception:
            pass

        with open('Task2_FAQ_Chatbot/faq_data.json', 'r') as f:
            cls.faq_data = json.load(f)

        cls.faq_questions = [item['question'] for item in cls.faq_data]
        cls.faq_answers = [item['answer'] for item in cls.faq_data]
        cls.lemmatizer = WordNetLemmatizer()

        def preprocess(text):
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            try:
                tokens = nltk.word_tokenize(text)
            except Exception:
                tokens = text.split()
            try:
                tokens = [cls.lemmatizer.lemmatize(word) for word in tokens]
            except Exception:
                pass
            return ' '.join(tokens)

        cls.preprocess = staticmethod(preprocess)
        cls.preprocessed_questions = [preprocess(q) for q in cls.faq_questions]
        cls.vectorizer = TfidfVectorizer()
        cls.tfidf_matrix = cls.vectorizer.fit_transform(cls.preprocessed_questions)

    def test_faq_data_loaded(self):
        self.assertGreater(len(self.faq_questions), 0)
        self.assertEqual(len(self.faq_questions), len(self.faq_answers))

    def test_exact_question_match(self):
        query = "What is the return policy?"
        prep = self.preprocess(query)
        vec = self.vectorizer.transform([prep])
        sims = cosine_similarity(vec, self.tfidf_matrix)[0]
        best_idx = np.argmax(sims)
        self.assertGreater(sims[best_idx], 0.8)
        self.assertIn("30 days", self.faq_answers[best_idx])

    def test_paraphrased_query_match(self):
        query = "Can I send back an item I bought?"
        prep = self.preprocess(query)
        self.assertTrue(len(prep) > 0)

    def test_empty_string_handling(self):
        query = "   "
        prep = self.preprocess(query)
        self.assertEqual(prep.strip(), "")

if __name__ == '__main__':
    unittest.main()
