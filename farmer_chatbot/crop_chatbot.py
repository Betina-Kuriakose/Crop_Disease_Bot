import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
import os

class CropChatbot:
    def __init__(self, dataset_path):
        """
        Initialize the Crop Chatbot with the dataset.
        
        Args:
            dataset_path: Path to the CSV file containing questions and answers
        """
        self.dataset_path = dataset_path
        self.df = None
        self.vectorizer = None
        self.question_vectors = None
        self.load_data()
        self.preprocess_data()
        self.train_model()
    
    def load_data(self):
        """Load the dataset from CSV file."""
        print("Loading dataset...")
        try:
            self.df = pd.read_csv(self.dataset_path)
            print(f"Dataset loaded successfully! Found {len(self.df)} Q&A pairs.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def preprocess_data(self):
        """Preprocess the questions for better matching."""
        print("Preprocessing data...")
        # Clean and normalize questions
        self.df['cleaned_questions'] = self.df['questions'].apply(self.clean_text)
        # Remove any rows with empty questions or answers
        self.df = self.df.dropna(subset=['questions', 'answers'])
        self.df = self.df[self.df['questions'].str.strip() != '']
        self.df = self.df[self.df['answers'].str.strip() != '']
        print(f"After preprocessing: {len(self.df)} Q&A pairs remaining.")
    
    def clean_text(self, text):
        """Clean and normalize text for better matching."""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Strip dataset prompt prefixes
        text = re.sub(r'^\s*asking\s+(about|that|how|for)?\s*', '', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_crops(self, text):
        common_crops = {
            'mustard', 'coconut', 'rice', 'paddy', 'brinjal', 'eggplant', 'tomato',
            'maize', 'corn', 'bittergourd', 'bitter gourd', 'wheat', 'potato', 'chilli',
            'chili', 'cotton', 'banana', 'mango', 'papaya', 'tea', 'sugarcane', 'blackgram',
            'greengram', 'bhendi', 'okra', 'cabbage', 'cauliflower', 'aonla', 'chayote',
            'fish', 'cow', 'cattle', 'pig', 'goat', 'sheep', 'poultry', 'chicken'
        }
        cleaned = re.sub(r'[^a-z0-9\s]', ' ', str(text).lower())
        return {crop for crop in common_crops if crop in cleaned}
    
    def train_model(self):
        """Train the TF-IDF vectorizer on the questions."""
        print("Training the NLP model...")
        self.df['crop_entities'] = self.df['cleaned_questions'].apply(self.extract_crops)
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            stop_words='english',
            min_df=2,
            max_df=0.90
        )
        
        # Fit and transform the questions
        self.question_vectors = self.vectorizer.fit_transform(
            self.df['cleaned_questions'].values
        )
        print("Model training completed!")
    
    def find_answer(self, user_question, top_n=3):
        """
        Find the best answer(s) for a user question.
        """
        if not user_question or user_question.strip() == "":
            return [{"answer": "Please ask a question about crops or farming.", "similarity": 0.0}]
        
        cleaned_question = self.clean_text(user_question)
        user_crops = self.extract_crops(user_question)
        user_vector = self.vectorizer.transform([cleaned_question])
        
        similarities = cosine_similarity(user_vector, self.question_vectors).flatten()
        top_indices = similarities.argsort()[-50:][::-1]
        
        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < 0.15:
                continue
            row_crops = self.df.iloc[idx].get('crop_entities') or set()
            
            crop_factor = 1.0
            if user_crops:
                if user_crops & row_crops:
                    crop_factor = 1.6
                else:
                    crop_factor = 0.2
            
            final_score = sim * crop_factor
            results.append({
                "answer": self.df.iloc[idx]['answers'],
                "similarity": sim,
                "final_score": final_score,
                "original_question": self.df.iloc[idx]['questions']
            })
        
        results.sort(key=lambda x: x["final_score"], reverse=True)
        results = results[:top_n]
        
        if not results:
            return [{
                "answer": "I couldn't find a specific answer to your question. Could you try rephrasing it? For example, you could ask about: crop diseases, fertilizer doses, pest control, cultivation practices, or crop varieties.",
                "similarity": 0.0
            }]
        
        return results
    
    def get_best_answer(self, user_question):
        """
        Get the single best answer for a user question.
        
        Args:
            user_question: The user's question
        
        Returns:
            String containing the best answer
        """
        results = self.find_answer(user_question, top_n=1)
        if results:
            return results[0]['answer']
        return "I'm sorry, I couldn't find a relevant answer. Please try rephrasing your question."
    
    def save_model(self, model_path='crop_chatbot_model.pkl'):
        """Save the trained model for future use."""
        model_data = {
            'vectorizer': self.vectorizer,
            'question_vectors': self.question_vectors,
            'df': self.df
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='crop_chatbot_model.pkl'):
        """Load a previously trained model."""
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.vectorizer = model_data['vectorizer']
            self.question_vectors = model_data['question_vectors']
            self.df = model_data['df']
            print(f"Model loaded from {model_path}")
            return True
        return False


if __name__ == "__main__":
    # Example usage
    dataset_path = "archive/questionsv4.csv"
    
    print("=" * 60)
    print("Crop Chatbot for Farmers - Initialization")
    print("=" * 60)
    
    chatbot = CropChatbot(dataset_path)
    
    print("\n" + "=" * 60)
    print("Chatbot is ready! You can now ask questions.")
    print("=" * 60)
    print("\nExample questions you can ask:")
    print("- How to control aphid infestation in mustard crops?")
    print("- What is the fertilizer dose for coconut?")
    print("- How to treat bacterial wilt in brinjal?")
    print("- What are the suitable varieties for rice?")
    print("\nType 'quit' to exit.\n")
    
    # Interactive command-line interface
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you for using the Crop Chatbot! Happy farming!")
            break
        
        if user_input:
            results = chatbot.find_answer(user_input, top_n=1)
            print(f"\nBot: {results[0]['answer']}")
            
            # Optionally show similarity score (for debugging)
            # print(f"[Similarity: {results[0]['similarity']:.3f}]")

