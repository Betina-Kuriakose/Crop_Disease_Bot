# 🌿 Crop Disease Bot - Farmer Chatbot Suite

A comprehensive Flask-based agricultural advisory assistant featuring multilingual NLP, voice input/output (Speech-to-Text & Text-to-Speech), and MongoDB Atlas user authentication. This application assists farmers with instant advice on crop diseases, fertilizer dosage, pest control, and agricultural best practices in 11+ languages.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.8+**
- **MongoDB** (Local instance or [MongoDB Atlas](https://www.mongodb.com/cloud/atlas))
- Internet connection (for machine translation & voice features)

---

### 2. Environment Setup & Dependencies

1. **Navigate to the project root directory:**
   ```powershell
   cd Crop_Disease_Bot
   ```

2. **Create and Activate Virtual Environment:**
   ```powershell
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

---

### 3. Environment Configuration (`.env`)

Create or update the `.env` file located inside `farmer_chatbot/.env`:

```env
# Flask Configuration
FLASK_SECRET_KEY=your_secret_key_here
SESSION_COOKIE_NAME=cropadvisor_session

# Database Configuration (MongoDB Atlas or Local)
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.o4scfv6.mongodb.net/
MONGO_DB_NAME=crop_advisor
MONGO_USER_COLLECTION=users

# Default Admin Credentials
DEFAULT_ADMIN_USER=farmer_admin
DEFAULT_ADMIN_PASS=demo123
```

---

### 4. Running the Application

```powershell
cd farmer_chatbot
python app_flask.py
```

The web server will start at: **`http://localhost:5000`**

- **Default Admin Username:** `farmer_admin`
- **Default Admin Password:** `demo123`

---

## ✨ Features

- 🌐 **Multilingual Support**: Chat in 11+ Indic languages (English, Hindi, Telugu, Tamil, Bengali, etc.).
- 🎙️ **Voice Features**: Speech-to-Text (STT) query input & Text-to-Speech (TTS) response playback.
- 🌾 **Agricultural Advisor**: Intent classification for crop diseases, treatments, fertilizers, and varieties.
- 🔒 **User Authentication**: Secure password hashing with MongoDB-backed login and signup.
- ⚡ **NLP & Vector Search**: TF-IDF similarity search enhanced with intent-based reranking.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Home / Authentication landing page |
| `GET` | `/chat` | Main interactive chatbot interface |
| `POST` | `/login` | Authenticate user session |
| `POST` | `/signup` | Register a new user |
| `GET` | `/logout` | End user session |
| `POST` | `/api/chat` | Send question and receive advisory answer JSON |
| `POST` | `/api/speech-to-text` | Convert uploaded voice recording into text query |
| `POST` | `/api/text-to-speech` | Synthesize answer string into playable MP3 audio stream |
| `GET` | `/api/languages` | List supported translation languages |
| `GET` | `/health` | Application & Database status health check |

---

## 🗂️ Project Structure

```
Crop_Disease_Bot/
├── .gitignore                   # Master Git ignore specification
├── README.md                    # Project documentation
├── requirements.txt             # Consolidated project dependencies
└── farmer_chatbot/
    ├── .env                     # Secrets & environment variables (ignored by Git)
    ├── .gitignore               # Local folder git ignore rules
    ├── app_flask.py             # Flask Web App, API Endpoints & DB Controller
    ├── crop_chatbot.py          # TF-IDF Vectorizer & Cosine Similarity search
    ├── nlp_utils.py             # NLTK cleaning & HuggingFace translation/summarization
    ├── voice_utils.py           # Speech-to-Text & gTTS Text-to-Speech engine
    ├── data_preprocess.py       # Dataset cleaning pipeline script
    ├── sample_queries.txt       # Sample farmer test queries
    ├── templates/
    │   ├── index.html           # Main chat web interface
    │   ├── login.html           # Authentication login page
    │   └── signup.html          # Authentication signup page
    └── archive/
        ├── questionsv4.csv      # Primary Q&A Knowledge Dataset
        └── questions_cleaned.csv# Preprocessed dataset corpus
```

---

## 🛠️ Troubleshooting

- **MongoDB Connection Error:**
  - Verify your IP is whitelisted in MongoDB Atlas under **Network Access** (Allow access from anywhere `0.0.0.0/0` during development).
  - Ensure the credentials in `farmer_chatbot/.env` match your Atlas database user.
- **PyAudio Warning (Optional):**
  - PyAudio is optional for desktop mic capture. The web browser microphone input works via Web APIs and standard file upload `/api/speech-to-text`.
- **Translation Connectivity:**
  - Google Translate & mBART require active internet connectivity.
