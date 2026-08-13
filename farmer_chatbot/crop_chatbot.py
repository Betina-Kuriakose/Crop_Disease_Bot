import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
import os

class CropChatbot:
    CROP_ALIASES = {
        'mustard': ['mustard'],
        'coconut': ['coconut'],
        'rice': ['rice', 'paddy', 'ahu', 'sali', 'boro'],
        'brinjal': ['brinjal', 'eggplant'],
        'tomato': ['tomato'],
        'maize': ['maize', 'corn'],
        'bittergourd': ['bittergourd', 'bitter gourd', 'biter gourd'],
        'wheat': ['wheat'],
        'potato': ['potato'],
        'chilli': ['chilli', 'chili'],
        'cotton': ['cotton'],
        'banana': ['banana'],
        'mango': ['mango'],
        'papaya': ['papaya'],
        'tea': ['tea'],
        'sugarcane': ['sugarcane'],
        'blackgram': ['blackgram', 'black gram'],
        'greengram': ['greengram', 'green gram'],
        'bhendi': ['bhendi', 'okra'],
        'cabbage': ['cabbage'],
        'cauliflower': ['cauliflower'],
        'aonla': ['aonla'],
        'chayote': ['chayote'],
        'pumpkin': ['pumpkin', 'pumkin'],
        'fish': ['fish', 'fishes', 'carp', 'carps', 'chitala', 'pond'],
        'cow': ['cow', 'cows', 'cattle', 'milk', 'dairy'],
        'goat': ['goat', 'goats']
    }

    INTENT_KEYWORDS = {
        "fertilizer": {
            "fertilizer", "fertiliser", "fertilizers", "fertilisers", "manure",
            "nutrient", "nutrients", "dose", "doses", "application", "apply",
            "npk", "nitrogen", "phosphorus", "potash", "potassium", "urea", "dap", "mop"
        },
        "disease_prevention": {
            "disease", "diseases", "prevent", "prevention", "protect", "protection",
            "control", "biosecurity", "healthy", "immunity", "sanitation", "hygiene",
            "clean", "vaccination", "vaccine", "wilt", "blight", "rot", "canker", "virus", "dying"
        },
        "seed_treatment": {
            "treat", "treated", "treatment", "chemical", "bavistin", "captaf",
            "fungicide", "dressing", "disinfection", "germination", "seed"
        },
        "season_sowing": {
            "season", "sowing", "sow", "sowed", "planting", "plant", "planting season", "timing",
            "month", "months", "period", "harvest", "time"
        },
        "variety": {
            "variety", "varieties", "hybrid", "breed", "cultivar"
        },
        "soil": {
            "soil", "loam", "sandy", "land", "ph", "drainage"
        }
    }

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
            self.df.fillna("", inplace=True)
            print(f"Dataset loaded successfully! Found {len(self.df)} Q&A pairs.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def preprocess_data(self):
        """Preprocess the questions for better matching."""
        print("Preprocessing data...")
        # Remove any rows with empty questions or answers
        self.df = self.df.dropna(subset=['questions', 'answers'])
        self.df = self.df[self.df['questions'].str.strip() != '']
        self.df = self.df[self.df['answers'].str.strip() != '']

        # Determine processed questions
        if "question_processed" in self.df.columns and self.df["question_processed"].notna().any():
            self.df['cleaned_questions'] = self.df['question_processed'].apply(self.clean_text)
        else:
            self.df['cleaned_questions'] = self.df['questions'].apply(self.clean_text)

        self.df['crop_entities'] = self.df['questions'].apply(self.extract_crops)
        print(f"After preprocessing: {len(self.df)} Q&A pairs remaining.")
    
    def clean_text(self, text):
        """Clean and normalize text for better matching."""
        if pd.isna(text) or not text:
            return ""
        text = str(text).lower()
        # Strip dataset prompt prefixes and fix typos
        text = re.sub(r'^\s*asking\s+(about|that|how|for)?\s*', '', text)
        text = text.replace("pumkin", "pumpkin").replace("friut", "fruit").replace("sawing", "sowing")
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_crops(self, text):
        """Extract matched crop entity canonical names from text."""
        if not text:
            return set()
        cleaned = re.sub(r'[^a-z0-9\s]', ' ', str(text).lower()).replace("pumkin", "pumpkin")
        found = set()
        for main_crop, aliases in self.CROP_ALIASES.items():
            for alias in aliases:
                if re.search(r'\b' + re.escape(alias) + r'\b', cleaned):
                    found.add(main_crop)
                    break
        return found

    def detect_intents(self, text):
        """Detect high-level intent categories from text."""
        if not text:
            return set()
        cleaned_words = set(self.clean_text(text).split())
        detected = set()
        for intent, kw_set in self.INTENT_KEYWORDS.items():
            if cleaned_words & kw_set:
                detected.add(intent)
        return detected

    def detect_primary_intent(self, text):
        """Detect single primary intent for classification."""
        intents = self.detect_intents(text)
        return next(iter(intents), None) if intents else None
    
    def train_model(self):
        """Train the TF-IDF vectorizer on the questions."""
        print("Training the NLP model...")
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

    def find_best_answer(self, query, top_n=100):
        """Find the single best answer dict for a query using crop matching, intent matching, and TF-IDF similarity."""
        if not query or not query.strip():
            return None

        raw_query = query
        query_clean = self.clean_text(query)
        if not query_clean:
            query_clean = raw_query.lower().strip()

        query_crops = self.extract_crops(raw_query)
        query_intents = self.detect_intents(raw_query)

        q_vec = self.vectorizer.transform([query_clean])
        sims = cosine_similarity(q_vec, self.question_vectors)[0]

        top_indices = sims.argsort()[-top_n:][::-1]

        results = []
        for idx in top_indices:
            sim = float(sims[idx])
            if sim < 0.10:
                continue

            row = self.df.iloc[int(idx)]
            answer_str = str(row["answers"]).strip().lower()
            question_str = str(row["questions"]).strip().lower()

            # Filter out trivial dummy answers
            if len(answer_str) < 5 or answer_str == question_str:
                continue

            row_crops = row.get("crop_entities") or set()
            row_intents = self.detect_intents(str(row["questions"]) + " " + str(row["answers"]))

            crop_factor = 1.0
            if query_crops:
                if query_crops & row_crops:
                    crop_factor = 1.6
                else:
                    crop_factor = 0.25  # Cross-crop penalty

            intent_bonus = 0.0
            if query_intents:
                common_intents = query_intents & row_intents
                intent_bonus = len(common_intents) * 0.25

            final_score = (sim * crop_factor) + intent_bonus
            results.append({
                "question": row["questions"],
                "answer": row["answers"],
                "score": sim,
                "final_score": final_score,
                "intent_match_score": len(query_intents & row_intents) if query_intents else 0
            })

        if not results:
            return None

        results.sort(key=lambda item: item["final_score"], reverse=True)
        return results[0]
    
    def find_answer(self, user_question, top_n=3):
        """Find top N answers for CLI usage."""
        if not user_question or user_question.strip() == "":
            return [{"answer": "Please ask a question about crops or farming.", "similarity": 0.0}]

        best = self.find_best_answer(user_question, top_n=50)
        if best:
            return [{
                "answer": best["answer"],
                "similarity": best["score"],
                "final_score": best["final_score"],
                "original_question": best["question"]
            }]

        return [{
            "answer": "I couldn't find a specific answer to your question. Could you try rephrasing it? For example, you could ask about: crop diseases, fertilizer doses, pest control, cultivation practices, or crop varieties.",
            "similarity": 0.0
        }]
    
    def format_answer(self, user_question, raw_answer):
        """Refine raw dataset answer into warm, structured farmer advisory using Gemini AI."""
        try:
            from nlp_utils import format_with_generative_ai
            return format_with_generative_ai(user_question, raw_answer)
        except Exception:
            return raw_answer

    def get_best_answer(self, user_question, smooth=False):
        """Get single best answer string, optionally formatted with Generative AI."""
        results = self.find_answer(user_question, top_n=1)
        if results:
            raw = results[0]['answer']
            if smooth:
                return self.format_answer(user_question, raw)
            return raw
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

