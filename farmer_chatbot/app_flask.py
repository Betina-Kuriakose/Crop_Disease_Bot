import os
import json
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import base64
import io

# Translation support
try:
    from googletrans import Translator
    translator = Translator()
    TRANSLATOR_AVAILABLE = True
except Exception as e:
    print(f"Translation not available: {e}")
    translator = None
    TRANSLATOR_AVAILABLE = False

try:
    from langdetect import detect
    DETECT_AVAILABLE = True
except Exception:
    detect = None
    DETECT_AVAILABLE = False

# Voice support
try:
    import speech_recognition as sr
    from voice_utils import VoiceAssistant
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("Voice features not available")

USE_NLP_PIPELINE = os.getenv("FARM_BOT_DISABLE_NLP", "").lower() not in {"1", "true", "yes"}

if USE_NLP_PIPELINE:
    try:
        from nlp_utils import get_nlp_processor

        nlp_processor = get_nlp_processor()
        NLP_AVAILABLE = nlp_processor is not None
    except Exception as exc:
        nlp_processor = None
        NLP_AVAILABLE = False
        print(f"NLP enhancements not available: {exc}")
else:
    nlp_processor = None
    NLP_AVAILABLE = False
    print("NLP enhancements disabled via FARM_BOT_DISABLE_NLP environment variable")

INTENT_KEYWORDS = {
    "fertilizer": {
        "fertilizer",
        "fertiliser",
        "fertilizers",
        "fertilisers",
        "manure",
        "nutrient",
        "nutrients",
        "dose",
        "doses",
        "application",
        "apply",
        "npk",
        "nitrogen",
        "phosphorus",
        "potash",
        "potassium",
        "urea",
        "dap",
        "mop",
    },
    "disease_prevention": {
        "disease",
        "diseases",
        "prevent",
        "prevention",
        "protect",
        "protection",
        "control",
        "biosecurity",
        "healthy",
        "immunity",
        "sanitation",
        "hygiene",
        "clean",
        "vaccination",
        "vaccine",
    },
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

app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Load dataset (prefer pre-cleaned CSV if available)
BASE_DIR = os.path.dirname(__file__)
RAW_DATA_PATH = os.path.join(BASE_DIR, "archive", "questionsv4.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "archive", "questions_cleaned.csv")

if os.path.exists(CLEAN_DATA_PATH):
    df = pd.read_csv(CLEAN_DATA_PATH)
    ACTIVE_DATASET = CLEAN_DATA_PATH
else:
    df = pd.read_csv(RAW_DATA_PATH)
    ACTIVE_DATASET = RAW_DATA_PATH

df.fillna("", inplace=True)

# Create corpus from questions
if "question_processed" in df.columns and df["question_processed"].notna().any():
    df['processed_questions'] = df['question_processed'].fillna("").astype(str)
elif NLP_AVAILABLE and nlp_processor:
    df['processed_questions'] = df['questions'].apply(nlp_processor.clean_text)
else:
    df['processed_questions'] = df['questions'].str.lower().str.strip()

df['token_set'] = df['processed_questions'].apply(lambda txt: set((txt or "").split()))
corpus = df['processed_questions'].tolist()
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
tfidf_matrix = vectorizer.fit_transform(corpus)

# Initialize voice assistant if available
voice_assistant = None
if VOICE_AVAILABLE:
    try:
        voice_assistant = VoiceAssistant()
    except Exception as e:
        print(f"Voice assistant initialization failed: {e}")

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
    # Prefer online translator, fallback to local MBART helper
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


def detect_intent(text: str) -> str | None:
    """Detect high-level intent (fertilizer, etc.) from the query."""
    if not text:
        return None
    normalized = prepare_query(text)
    tokens = set(normalized.split())
    for intent, keywords in INTENT_KEYWORDS.items():
        if tokens & keywords:
            return intent
    return None


def answer_has_medication(text: str) -> bool:
    """Return True if the answer instructs medication usage."""
    if not text:
        return False
    lowered = text.lower()
    return any(token in lowered for token in MEDICATION_TERMS)


def rerank_results(results, intent: str | None):
    """Use intent keywords to boost relevant answers."""
    if not results:
        return None
    if not intent or intent not in INTENT_KEYWORDS:
        best = results[0]
        best.pop("token_set", None)
        best["intent_match_score"] = 0
        return best

    keywords = INTENT_KEYWORDS[intent]

    def boost(item):
        token_set = item.get("token_set") or set()
        return len(token_set & keywords)

    ranked = sorted(results, key=lambda item: (boost(item), item["score"]), reverse=True)
    best = ranked[0]
    best["intent_match_score"] = boost(best)
    best.pop("token_set", None)
    return best

def find_best_answer(query: str, top_n=5, intent: str | None = None):
    """Find the best answer(s) for a query using TF-IDF and cosine similarity."""
    if not query.strip():
        return None
    
    # Clean query
    query = prepare_query(query)

    if not query:
        return None
    
    # Vectorize query
    q_vec = vectorizer.transform([query])
    
    # Calculate similarity
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    
    # Get top N results
    results = []
    top_indices = sims.argsort()[-top_n:][::-1]
    
    for idx in top_indices:
        if sims[idx] > 0.1:  # Threshold for relevance
            best_row = df.iloc[int(idx)]
            results.append({
                "question": best_row["questions"],
                "answer": best_row["answers"],
                "score": float(sims[idx]),
                "token_set": best_row["token_set"],
            })
    
    return rerank_results(results, intent)

@app.route("/")
def index():
    """Render the main chat interface."""
    return render_template("index.html")

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
    intent = detect_intent(query_en)
    best = find_best_answer(query_en, intent=intent)
    
    if not best:
        reply_en = INTENT_FALLBACKS.get(
            intent,
            "Sorry, I could not understand your question about crops. Could you please rephrase it or be more specific about the crop or issue?",
        )
    elif (
        intent
        and best.get("intent_match_score", 0) == 0
        and best.get("score", 0) < 0.3
    ):
        reply_en = INTENT_FALLBACKS.get(
            intent,
            "Sorry, I could not find a specific answer. Please provide more details.",
        )
    else:
        reply_en = best["answer"]
        # Normalize answer into English so downstream translation works reliably
        if DETECT_AVAILABLE and reply_en:
            answer_lang = detect_language(reply_en)
            if answer_lang and not answer_lang.startswith("en"):
                reply_en = translate_to_english(reply_en, answer_lang)

        if intent == "disease_prevention" and answer_has_medication(reply_en):
            reply_en = INTENT_FALLBACKS["disease_prevention"]
    
    # Translate reply back to user's language
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
    print(f"Dataset loaded: {len(df)} Q&A pairs")
    print(f"Voice features: {'Available' if VOICE_AVAILABLE else 'Not available'}")
    print(f"Translation: {'Available' if TRANSLATOR_AVAILABLE else 'Not available'}")
    print("\n" + "="*70)
    print("🌐 ACCESS URLS:")
    print("="*70)
    print("📍 Local access (⭐ BEST for microphone):")
    print("   → http://localhost:5000")
    print("   → http://127.0.0.1:5000")
    print("\n📍 Network access (for other devices):")
    
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"   → http://{local_ip}:5000")
        print(f"   ⚠️  Microphone may require localhost or permission setup")
    except:
        print("   → http://YOUR_IP_ADDRESS:5000")
    
    print("\n" + "="*70)
    print("🎤 MICROPHONE PERMISSION SETUP:")
    print("="*70)
    print("If microphone is BLOCKED in browser:")
    print("1. Click lock icon (🔒) in address bar")
    print("2. Find 'Microphone' → Change to 'Allow'")
    print("3. Refresh the page")
    print("4. Or use localhost URL for better support")
    print("="*70 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5000)

