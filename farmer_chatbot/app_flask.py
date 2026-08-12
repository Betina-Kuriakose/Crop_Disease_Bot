import os
import json
from functools import wraps

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    redirect,
    url_for,
    session,
)
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import base64
import io

# Translation support (Prefer deep-translator, fallback to googletrans)
DEEP_TRANSLATE_AVAILABLE = False
try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATE_AVAILABLE = True
except Exception:
    DEEP_TRANSLATE_AVAILABLE = False

try:
    from googletrans import Translator
    translator = Translator()
    TRANSLATOR_AVAILABLE = True
except Exception as e:
    translator = None
    TRANSLATOR_AVAILABLE = False

try:
    from langdetect import detect
    DETECT_AVAILABLE = True
except Exception:
    detect = None
    DETECT_AVAILABLE = False

# Voice support (optional - PyAudio can be tricky on Windows)
try:
    import speech_recognition as sr
    from voice_utils import VoiceAssistant
    VOICE_AVAILABLE = True
except ImportError as e:
    VOICE_AVAILABLE = False
    print(f"Voice features not available: {e}")
except Exception as e:
    VOICE_AVAILABLE = False
    print(f"Voice features initialization failed: {e}")

USE_NLP_PIPELINE = os.getenv("FARM_BOT_DISABLE_NLP", "").lower() not in {"1", "true", "yes"}

if USE_NLP_PIPELINE:
    try:
        from nlp_utils import get_nlp_processor, format_with_generative_ai
        nlp_processor = get_nlp_processor()
        NLP_AVAILABLE = nlp_processor is not None
    except Exception as exc:
        nlp_processor = None
        NLP_AVAILABLE = False
        format_with_generative_ai = lambda q, a, **kw: a
        print(f"NLP enhancements not available: {exc}")
else:
    nlp_processor = None
    NLP_AVAILABLE = False
    format_with_generative_ai = lambda q, a, **kw: a
    print("NLP enhancements disabled via FARM_BOT_DISABLE_NLP environment variable")

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

INTENT_FALLBACKS = {
    "fertilizer": (
        "For fertilizer guidance, please mention the crop, growth stage, and any soil test "
        "results so I can recommend an appropriate nutrient schedule."
    ),
    "disease_prevention": (
        "Disease prevention advice depends on the crop or livestock plus symptoms/history. "
        "Please share the species, growth stage, and recent issues so I can outline hygiene, "
        "nutrition, and vaccination practices rather than prescribing medicines blindly."
    ),
}

MEDICATION_TERMS = {
    "enroflox",
    "enrofloxacin",
    "tablet",
    "tab",
    "capsule",
    "injection",
    "inj",
    "ml",
    "dose",
    "mg",
    "antibiotic",
    "medicine",
    "drug",
}

BASE_DIR = os.path.dirname(__file__)

# Load environment variables from .env file
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Also try to read from .evn file (if it exists)
EVN_FILE = os.path.join(BASE_DIR, ".evn")
if os.path.exists(EVN_FILE):
    with open(EVN_FILE, "r") as f:
        evn_content = f.read().strip()
        if evn_content:
            # If the file contains just the URI, use it directly
            if evn_content.startswith("mongodb"):
                os.environ["MONGO_URI"] = evn_content
                print(f"[Config] Loaded MONGO_URI from .evn file")
            # If it's in KEY=VALUE format, parse it
            elif "=" in evn_content:
                for line in evn_content.split("\n"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
                print(f"[Config] Loaded environment variables from .evn file")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "crop-advisor-secret")
app.config["SESSION_COOKIE_NAME"] = os.environ.get("SESSION_COOKIE_NAME", "cropadvisor_session")
CORS(app)  # Enable CORS for API access

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "crop_advisor")
MONGO_USER_COLLECTION = os.environ.get("MONGO_USER_COLLECTION", "users")

mongo_client = None
users_collection = None

# Mask password in URI for logging
def mask_uri(uri):
    """Mask password in MongoDB URI for safe logging."""
    if "@" in uri and "://" in uri:
        parts = uri.split("://")
        if len(parts) == 2:
            scheme = parts[0]
            rest = parts[1]
            if "@" in rest:
                user_pass, host = rest.split("@", 1)
                if ":" in user_pass:
                    user, _ = user_pass.split(":", 1)
                    return f"{scheme}://{user}:***@{host}"
    return uri

# MongoDB connection (Express-style pattern)
mongo_client = None
db = None
users_collection = None

def initialize_mongodb():
    """Initialize MongoDB database and collection, creating them if they don't exist."""
    global mongo_client, db, users_collection
    
    try:
        print(f"[MongoDB] Attempting connection to: {mask_uri(MONGO_URI)}")
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Trigger server selection to fail fast if not reachable
        mongo_client.server_info()
        
        # Get database (MongoDB creates it automatically on first write)
        db = mongo_client[MONGO_DB_NAME]
        
        # Get collection (MongoDB creates it automatically on first write)
        users_collection = db[MONGO_USER_COLLECTION]
        
        # Explicitly create collection by inserting and deleting a dummy document if empty
        # This ensures the collection exists even before first user signup
        try:
            # Try to create index - this will create the collection if it doesn't exist
            users_collection.create_index("username", unique=True)
            print(f"[OK] Created/verified index on '{MONGO_USER_COLLECTION}.username'")
        except Exception as idx_err:
            print(f"[WARN] Index creation note: {idx_err}")
        
        # Verify collection exists by checking database
        collection_names = db.list_collection_names()
        if MONGO_USER_COLLECTION in collection_names:
            print(f"[OK] Collection '{MONGO_USER_COLLECTION}' exists")
        else:
            print(f"[INFO] Collection '{MONGO_USER_COLLECTION}' will be created on first write")
        
        print(f"[OK] MongoDB database connection established successfully")
        print(f"   Database: {MONGO_DB_NAME}")
        print(f"   Collection: {MONGO_USER_COLLECTION}")
        print(f"   Ready to store user credentials!")
        
        return True
    except PyMongoError as mongo_exc:
        print(f"[ERROR] [MongoDB] Connection unavailable: {mongo_exc}")
        print(f"   URI used: {mask_uri(MONGO_URI)}")
        mongo_client = None
        db = None
        users_collection = None
        return False
    except Exception as e:
        print(f"[ERROR] [MongoDB] Unexpected error: {e}")
        mongo_client = None
        db = None
        users_collection = None
        return False

# Initialize MongoDB connection
initialize_mongodb()

# Load dataset (prefer pre-cleaned CSV if available)
RAW_DATA_PATH = os.path.join(BASE_DIR, "archive", "questionsv4.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "archive", "questions_cleaned.csv")

if os.path.exists(CLEAN_DATA_PATH):
    df = pd.read_csv(CLEAN_DATA_PATH)
    ACTIVE_DATASET = CLEAN_DATA_PATH
else:
    df = pd.read_csv(RAW_DATA_PATH)
    ACTIVE_DATASET = RAW_DATA_PATH

df.fillna("", inplace=True)

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

def strip_prompt_prefix(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'^\s*asking\s+(about|that|how|for)?\s*', '', text)
    text = text.replace("pumkin", "pumpkin").replace("friut", "fruit").replace("sawing", "sowing")
    return re.sub(r'\s+', ' ', text).strip()

def extract_crop_entities(text: str) -> set:
    if not text:
        return set()
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', text.lower()).replace("pumkin", "pumpkin")
    found = set()
    for main_crop, aliases in CROP_ALIASES.items():
        for alias in aliases:
            if re.search(r'\b' + re.escape(alias) + r'\b', cleaned):
                found.add(main_crop)
                break
    return found

# Create corpus from questions
if "question_processed" in df.columns and df["question_processed"].notna().any():
    df['processed_questions'] = df['question_processed'].apply(strip_prompt_prefix)
elif NLP_AVAILABLE and nlp_processor:
    df['processed_questions'] = df['questions'].apply(lambda q: strip_prompt_prefix(nlp_processor.clean_text(q)))
else:
    df['processed_questions'] = df['questions'].apply(strip_prompt_prefix)

df['token_set'] = df['processed_questions'].apply(lambda txt: set((txt or "").split()))
df['crop_entities'] = df['processed_questions'].apply(extract_crop_entities)
corpus = df['processed_questions'].tolist()
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.90
)
tfidf_matrix = vectorizer.fit_transform(corpus)

# Initialize voice assistant if available
voice_assistant = None
if VOICE_AVAILABLE:
    try:
        voice_assistant = VoiceAssistant()
    except Exception as e:
        print(f"Voice assistant initialization failed: {e}")


def ensure_default_user():
    """Seed a default admin user only if collection is empty (fallback)."""
    if users_collection is None:
        return

    # Only create default user if collection is completely empty
    user_count = users_collection.count_documents({})
    if user_count == 0:
        default_username = os.environ.get("DEFAULT_ADMIN_USER", "farmer_admin")
        default_password = os.environ.get("DEFAULT_ADMIN_PASS", "demo123")
        
        try:
            users_collection.insert_one(
                {
                    "username": default_username,
                    "password_hash": generate_password_hash(default_password),
                    "role": "admin",
                    "created_at": pd.Timestamp.now().isoformat()
                }
            )
            print(
                f"[MongoDB] Seeded default admin user '{default_username}' (collection was empty). "
                "Change DEFAULT_ADMIN_PASS after first login."
            )
        except Exception as e:
            print(f"[MongoDB] Could not seed default user: {e}")


def login_required(view_func):
    """Simple session-based access control decorator."""

    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped_view


# Ensure default user exists if MongoDB is connected
if users_collection is not None:
    try:
        ensure_default_user()
    except Exception as e:
        print(f"Warning: Could not ensure default user: {e}")

def detect_language(text: str) -> str:
    """Detect the language of the input text."""
    if not DETECT_AVAILABLE or not text.strip():
        return "en"
    try:
        lang = detect(text)
        return lang
    except Exception:
        return "en"

def translate_to_english(text: str, src_lang: str) -> str:
    """Translate text to English if needed."""
    if not text:
        return text
    if src_lang.startswith("en"):
        return text
    if DEEP_TRANSLATE_AVAILABLE:
        try:
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception:
            pass
    if TRANSLATOR_AVAILABLE:
        try:
            res = translator.translate(text, src="auto", dest="en")
            return res.text
        except Exception:
            pass
    if NLP_AVAILABLE and nlp_processor and nlp_processor.can_translate(src_lang, "en"):
        translated = nlp_processor.translate(text, src_lang, "en")
        if translated:
            return translated
    return text


def translate_from_english(text: str, target_lang: str) -> str:
    """Translate text from English to target language."""
    if not text or not target_lang or target_lang.startswith("en"):
        return text
    base_lang = target_lang.split("-")[0]
    if DEEP_TRANSLATE_AVAILABLE:
        try:
            return GoogleTranslator(source="auto", target=base_lang).translate(text)
        except Exception:
            pass
    if TRANSLATOR_AVAILABLE:
        try:
            res = translator.translate(text, src="en", dest=base_lang)
            return res.text
        except Exception:
            pass
    if NLP_AVAILABLE and nlp_processor and nlp_processor.can_translate("en", base_lang):
        translated = nlp_processor.translate(text, "en", base_lang)
        if translated:
            return translated
    return text


def prepare_query(text: str) -> str:
    """Normalize query using the notebook-inspired NLP pipeline."""
    if NLP_AVAILABLE and nlp_processor:
        return nlp_processor.normalize_question(text)
    return text.lower().strip()


def detect_intents(text: str) -> set:
    """Detect high-level intents from the query."""
    if not text:
        return set()
    cleaned = set(strip_prompt_prefix(text).split())
    detected = set()
    for intent, kw_set in INTENT_KEYWORDS.items():
        if cleaned & kw_set:
            detected.add(intent)
    return detected

def detect_intent(text: str) -> str | None:
    """Detect single primary intent for backwards compatibility."""
    intents = detect_intents(text)
    return next(iter(intents), None) if intents else None

def answer_has_medication(text: str) -> bool:
    """Return True if the answer instructs medication usage."""
    if not text:
        return False
    lowered = text.lower()
    return any(token in lowered for token in MEDICATION_TERMS)

def find_best_answer(query: str, top_n=100, intent: str | None = None):
    """Find the best answer(s) for a query using crop entity matching and TF-IDF similarity."""
    if not query or not query.strip():
        return None
    
    raw_query = query
    query_clean = strip_prompt_prefix(query)
    if NLP_AVAILABLE and nlp_processor:
        query_clean = nlp_processor.clean_text(query_clean)

    if not query_clean:
        query_clean = raw_query.lower().strip()
    
    query_crops = extract_crop_entities(raw_query)
    query_intents = detect_intents(raw_query)
    q_vec = vectorizer.transform([query_clean])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    
    top_indices = sims.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        sim = float(sims[idx])
        if sim < 0.10:
            continue
        best_row = df.iloc[int(idx)]
        answer_str = str(best_row["answers"]).strip().lower()
        question_str = str(best_row["questions"]).strip().lower()
        
        # Filter out trivial dummy answers (e.g. question == answer)
        if len(answer_str) < 5 or answer_str == question_str:
            continue
            
        row_crops = best_row.get("crop_entities") or set()
        row_intents = detect_intents(best_row["questions"] + " " + best_row["answers"])
        
        crop_factor = 1.0
        if query_crops:
            if query_crops & row_crops:
                crop_factor = 1.6
            else:
                crop_factor = 0.25  # Heavy penalty for cross-crop mismatch
        
        intent_bonus = 0.0
        if query_intents:
            common_intents = query_intents & row_intents
            intent_bonus = len(common_intents) * 0.25
        
        final_score = (sim * crop_factor) + intent_bonus
        results.append({
            "question": best_row["questions"],
            "answer": best_row["answers"],
            "score": sim,
            "final_score": final_score,
            "intent_match_score": len(query_intents & row_intents) if query_intents else 0
        })
    
    if not results:
        return None
        
    results.sort(key=lambda item: item["final_score"], reverse=True)
    best = results[0]
    return best
'''
@app.route("/")
def index():
    """Render the main chat interface."""
    return render_template("index.html")'''
@app.route("/")
def index():
    """Public homepage"""
    return render_template("index.html")


@app.route("/about")
def about():
    """About Us Page"""
    return render_template("about.html")


@app.route("/contact")
def contact():
    """Contact Page"""
    return render_template("contact.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Render and process the login form backed by MongoDB."""
    error = None

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if not username or not password:
            error = "Username and password are required."
        elif users_collection is None:
            error = "Authentication service is unavailable. Please contact the administrator."
        else:
            user_doc = users_collection.find_one({"username": username})
            if not user_doc:
                # User doesn't exist, redirect to signup
                return redirect(url_for("signup", username=username))
            elif check_password_hash(user_doc.get("password_hash", ""), password):
                session["user_id"] = str(user_doc["_id"])
                session["username"] = user_doc.get("username")
                return redirect(url_for("chat_page"))
            else:
                error = "Invalid password. Please try again."

    return render_template("login.html", error=error)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Render and process the signup form, storing credentials in MongoDB."""
    error = None
    username = request.args.get("username", "")

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if not username or not password:
            error = "Username and password are required."
        elif password != confirm_password:
            error = "Passwords do not match."
        elif len(password) < 6:
            error = "Password must be at least 6 characters long."
        elif users_collection is None:
            error = "Registration service is unavailable. Please contact the administrator."
        else:
            # Check if username already exists
            existing_user = users_collection.find_one({"username": username})
            if existing_user:
                error = "Username already exists. Please choose a different username."
            else:
                try:
                    # Create new user in MongoDB
                    new_user = {
                        "username": username,
                        "password_hash": generate_password_hash(password),
                        "role": "user",
                        "created_at": pd.Timestamp.now().isoformat()
                    }
                    result = users_collection.insert_one(new_user)
                    
                    # Auto-login after signup
                    session["user_id"] = str(result.inserted_id)
                    session["username"] = username
                    print(f"[OK] New user registered: {username}")
                    return redirect(url_for("chat_page"))
                except Exception as e:
                    error = f"Registration failed: {str(e)}"
                    print(f"[ERROR] Signup error: {e}")

    return render_template("signup.html", error=error, username=username)


@app.route("/logout")
def logout():
    """Clear the user session."""
    session.clear()
    return redirect(url_for("login"))


@app.route("/chat")
@login_required
def chat_page():
    """Protected chat UI."""
    return render_template("chat.html", username=session.get("username"))

@app.route("/api/chat", methods=["POST"])
def chat():
    """API endpoint for chat queries."""
    data = request.json or {}
    user_text = (data.get("text") or "").strip()
    lang = (data.get("language") or "en").lower()
    
    if not user_text:
        return jsonify({"error": "Empty message"}), 400
    
    # Auto-detect language if needed
    if lang in ("auto", "", None):
        lang = detect_language(user_text)
    
    # Translate to English for processing
    query_en = translate_to_english(user_text, lang)
    
    # Determine query intent and find best answer
    intents = detect_intents(query_en)
    primary_intent = detect_intent(query_en)
    intent = primary_intent
    best = find_best_answer(query_en)
    
    if not best:
        reply_en = INTENT_FALLBACKS.get(
            primary_intent,
            "Sorry, I could not understand your question about crops. Could you please rephrase it or be more specific about the crop or issue?",
        )
    elif (
        primary_intent
        and best.get("intent_match_score", 0) == 0
        and best.get("score", 0) < 0.3
    ):
        reply_en = INTENT_FALLBACKS.get(
            primary_intent,
            "Sorry, I could not find a specific answer. Please provide more details.",
        )
    else:
        raw_answer = best["answer"]
        # Normalize answer into English if needed
        if DETECT_AVAILABLE and raw_answer:
            answer_lang = detect_language(raw_answer)
            if answer_lang and not answer_lang.startswith("en"):
                raw_answer = translate_to_english(raw_answer, answer_lang)

        # Enhance raw database answer using Generative AI in the user's question language
        formatted_ai = format_with_generative_ai(user_text, raw_answer)

        if intent == "disease_prevention" and answer_has_medication(formatted_ai):
            reply_en = INTENT_FALLBACKS["disease_prevention"]
            reply_out = translate_from_english(reply_en, lang)
        elif formatted_ai != raw_answer:
            reply_en = formatted_ai
            reply_out = formatted_ai
        else:
            reply_en = raw_answer
            reply_out = translate_from_english(reply_en, lang)
    
    return jsonify({
        "reply": reply_out,
        "meta": {
            "original_reply_en": reply_en,
            "language_detected": lang,
            "score": best.get("score") if best else None,
            "question": best.get("question") if best else None
        }
    })

@app.route("/api/speech-to-text", methods=["POST"])
def speech_to_text():
    """API endpoint for converting speech to text."""
    if not VOICE_AVAILABLE:
        return jsonify({"error": "Speech recognition not available"}), 503
    
    try:
        # Check if audio file is uploaded
        if 'audio' in request.files:
            audio_file = request.files['audio']
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                audio_file.save(tmp_file.name)
                tmp_file_path = tmp_file.name
            
            try:
                r = sr.Recognizer()
                with sr.AudioFile(tmp_file_path) as source:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = r.record(source)
                
                # Try Google Speech Recognition
                try:
                    text = r.recognize_google(audio_data)
                    return jsonify({
                        "text": text,
                        "success": True
                    })
                except sr.UnknownValueError:
                    return jsonify({
                        "error": "Could not understand audio",
                        "success": False
                    }), 400
                except sr.RequestError as e:
                    return jsonify({
                        "error": f"Speech recognition service error: {e}",
                        "success": False
                    }), 500
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        
        # Check if base64 audio data is provided
        elif 'audio_data' in request.json:
            audio_data_b64 = request.json['audio_data']
            # Decode base64 and process
            # Implementation would go here
            return jsonify({"error": "Base64 audio not yet implemented"}), 501
        
        else:
            return jsonify({"error": "No audio data provided"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route("/api/text-to-speech", methods=["POST"])
def text_to_speech():
    """API endpoint for converting text to speech."""
    if not VOICE_AVAILABLE or not voice_assistant:
        return jsonify({"error": "Text-to-speech not available"}), 503
    
    data = request.json or {}
    text = data.get("text", "").strip()
    lang = data.get("language", "en")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        # Generate audio file
        audio_file = voice_assistant.text_to_audio_file(text, lang=lang)
        
        if audio_file and os.path.exists(audio_file):
            return send_file(
                audio_file,
                mimetype='audio/mpeg',
                as_attachment=True,
                download_name='response.mp3'
            )
        else:
            return jsonify({"error": "Could not generate audio"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/languages", methods=["GET"])
def get_languages():
    """Get list of supported languages."""
    return jsonify({
        "supported_languages": [
            {"code": "en", "name": "English"},
            {"code": "hi", "name": "Hindi"},
            {"code": "ta", "name": "Tamil"},
            {"code": "te", "name": "Telugu"},
            {"code": "mr", "name": "Marathi"},
            {"code": "gu", "name": "Gujarati"},
            {"code": "bn", "name": "Bengali"},
            {"code": "kn", "name": "Kannada"},
            {"code": "ml", "name": "Malayalam"},
            {"code": "pa", "name": "Punjabi"},
            {"code": "ur", "name": "Urdu"},
        ]
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "dataset_size": len(df),
        "voice_available": VOICE_AVAILABLE,
        "translation_available": TRANSLATOR_AVAILABLE
    })

if __name__ == "__main__":
    print("Starting Crop Chatbot Flask Application...")
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    print(f"Dataset loaded: {len(df)} Q&A pairs")
    print(f"Voice features: {'Available' if VOICE_AVAILABLE else 'Not available'}")
    print(f"Translation: {'Available' if (DEEP_TRANSLATE_AVAILABLE or TRANSLATOR_AVAILABLE) else 'Not available'}")
    print(f"Generative AI RAG: {'[ENABLED]' if has_gemini else '[DISABLED - Set GEMINI_API_KEY in .env]'}")
    print("\n" + "="*70)
    print("ACCESS URLS:")
    print("="*70)
    print("Local access (BEST for microphone):")
    print("   -> http://localhost:5000")
    print("   -> http://127.0.0.1:5000")
    print("\nNetwork access (for other devices):")
    
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"   -> http://{local_ip}:5000")
        print(f"   [WARN] Microphone may require localhost or permission setup")
    except:
        print("   -> http://YOUR_IP_ADDRESS:5000")
    
    print("\n" + "="*70)
    print("MICROPHONE PERMISSION SETUP:")
    print("="*70)
    print("If microphone is BLOCKED in browser:")
    print("1. Click lock icon in address bar")
    print("2. Find 'Microphone' -> Change to 'Allow'")
    print("3. Refresh the page")
    print("4. Or use localhost URL for better support")
    print("="*70 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5000)

