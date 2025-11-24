# 🌾 Farmer Chatbot Suite (Flask + NLP + Voice)

This folder hosts the full-stack crop advisory assistant used throughout the project: a Flask API, multilingual NLP core, voice input/output helpers, classic frontend, and data tooling. This README replaces scattered notes and documents every file in `farmer_chatbot` so you only need this single guide.

---

## 1. Folder Map

```
farmer_chatbot/
├── app_flask.py           # Flask app factory + REST endpoints
├── crop_chatbot.py        # High-level chat orchestration & prompt logic
├── nlp_utils.py           # Embeddings, similarity search, translation helpers
├── voice_utils.py         # Speech-to-text & text-to-speech bridges
├── data_preprocess.py     # Dataset cleaners / vector store builders
├── frontend.html          # Standalone UI variant
├── templates/index.html   # Jinja template used by Flask
├── archive/questions*.csv # Historical Q/A datasets
├── requirements-flask.txt # Python dependencies
└── README-FLASK.md        # (this file)
```

If you need to add modules, extend this list so the README always mirrors reality.

---

## 2. Key Components

| File | Purpose | Notes |
| --- | --- | --- |
| `app_flask.py` | WSGI entrypoint, registers Blueprints, API routes, middleware, and CORS. | Exposes `/api/chat`, `/api/speech-to-text`, `/api/text-to-speech`, `/api/languages`, `/health`. |
| `crop_chatbot.py` | Conversation brain: loads cleaned dataset, uses similarity search, applies heuristic/rule-based fallbacks, returns structured replies. | Imports `nlp_utils` for language detection and embeddings. |
| `nlp_utils.py` | Contains sentence transformer utilities, translation helpers, and scoring helpers. | Uses Google Translate (or fallback) plus detection logic for 11 Indic languages. |
| `voice_utils.py` | Adapters for speech services. Wraps STT (Whisper/Google) and TTS (gTTS/Pyttsx3) plus audio serialization. | API routes convert blobs ↔ text here. |
| `data_preprocess.py` | CLI-style script to clean CSVs, normalize labels, and freeze embeddings to disk. | Run this before shipping a new dataset. |
| `frontend.html` & `templates/index.html` | Two UI options. `templates/index.html` is served by Flask, while `frontend.html` lets you test the API as a static page. | Both include mic controls, language picker, and chat history rendering. |
| `archive/questions_cleaned.csv` & `questionsv4.csv` | Canonical corpora used by `crop_chatbot.py`. | Keep them synced with preprocessing outputs. |
| `requirements-flask.txt` | Locked dependency list specifically for this service. | Use virtual envs when installing. |

---

## 3. Environment Setup

1. **Create & activate a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate # macOS / Linux
   ```
2. **Install dependencies**
   ```bash
   pip install -r farmer_chatbot/requirements-flask.txt
   ```
3. **Prepare datasets and embeddings**
   ```bash
   cd farmer_chatbot
   python data_preprocess.py --input archive/questionsv4.csv --output archive/questions_cleaned.csv
   ```
   The script handles language normalization and builds embedding caches referenced by the chatbot.

### Environment variables (`.env`)

Create `farmer_chatbot/.env` (never commit it) to configure secrets and Mongo access:

```
FLASK_SECRET_KEY=replace-this-with-a-long-random-string
SESSION_COOKIE_NAME=cropadvisor_session

MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=crop_advisor
MONGO_USER_COLLECTION=users

DEFAULT_ADMIN_USER=farmer_admin
DEFAULT_ADMIN_PASS=demo123
```

Restart Flask after editing the file. You can override any value with real secrets in production.

---

## 4. Running the Stack

### Local development
```bash
cd farmer_chatbot
python app_flask.py
```
Default server: `http://localhost:5000`

### Production options
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app_flask:app
```
or containerize (Dockerfile sketch):
```
FROM python:3.11-slim
WORKDIR /app
COPY farmer_chatbot/requirements-flask.txt .
RUN pip install -r requirements-flask.txt
COPY farmer_chatbot .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app_flask:app"]
```

---

## 5. API Reference

| Method | Route | Description |
| --- | --- | --- |
| `POST` | `/api/chat` | Body: `{ "text": "...", "language": "en" }`. Returns chatbot reply plus metadata (`original_reply_en`, `language_detected`, score, matched question). |
| `POST` | `/api/speech-to-text` | Multipart form (`audio` file). Returns `{ "text": "...", "success": true }`. |
| `POST` | `/api/text-to-speech` | JSON with `text`, `language`. Responds with MP3 bytes. |
| `GET` | `/api/languages` | Lists supported languages (ISO codes + display names). |
| `GET` | `/health` | Simple readiness probe. |

All endpoints return structured JSON; CORS is enabled for browser clients.

---

## 6. Frontend & Voice UX

- **templates/index.html**: served via Flask; includes:
  - Language selector (manual or auto-detect).
  - Chat bubbles with timestamps.
  - Mic button tied to `/api/speech-to-text`.
  - TTS toggle that hits `/api/text-to-speech`.
- **frontend.html**: drop-in static tester (just open in a browser and point to the Flask host).
- **Voice flow**: browsers capture audio → `voice_utils.speech_to_text()` → chatbot reply → optional `voice_utils.text_to_speech()` streaming back an MP3. The same utilities can be reused in mobile integrations.

---

## 7. NLP & Data Workflow

1. `data_preprocess.py` cleans and deduplicates CSVs, then stores embeddings (typically sentence-transformer or TF-IDF vectors).
2. `crop_chatbot.py` loads the cleaned data + embedding index on startup.
3. Incoming requests are language-detected (`nlp_utils.detect_language`), translated to English for similarity search, and matched to top-N knowledge base responses.
4. The reply is localized back into the user’s language; metadata includes detected language, confidence score, and matched canonical question.

To refresh the knowledge base, rerun preprocessing, verify the CSVs, and restart the Flask service.

---

## 8. Testing & Troubleshooting

- **Unit style tests**: run targeted scripts, e.g. `python -m pytest tests/test_nlp_utils.py` (create tests alongside modules).
- **Smoke test**: `curl -X POST http://localhost:5000/api/chat -H "Content-Type: application/json" -d '{"text": "How to manage aphids?", "language": "en"}'`
- **Common fixes**:
  - Missing models: rerun `data_preprocess.py` or download required Transformers.
  - Translation failures: ensure network access for Google Translate; provide fallback dictionary if running offline.
  - Audio decoding issues: confirm browser sends `audio/wav` or convert before hitting the endpoint.

---

## 9. Extending the System

- **Add languages**: update supported language list in `nlp_utils.py` and adjust frontend dropdown.
- **Plug new STT/TTS services**: wrap vendor SDKs inside `voice_utils.py` to keep API responses stable.
- **Switch embedding model**: change loader in `nlp_utils.py`, regenerate vectors via `data_preprocess.py`.
- **Integrate with mobile/other apps**: call the REST endpoints; the contract is intentionally simple for cross-platform clients.

---

## 10. Maintainer Checklist

- [ ] Keep `requirements-flask.txt` in sync after dependency upgrades.
- [ ] Rebuild embeddings whenever `archive/questions*.csv` changes.
- [ ] Update this README whenever file layout or APIs change (single source of truth).
- [ ] Run at least one smoke test before deployments.

Happy building! This README now fully documents every moving part inside `farmer_chatbot`. Update it first whenever you add/rename files so the “single README” promise stays true.
