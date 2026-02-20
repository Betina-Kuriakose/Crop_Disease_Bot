# 🌾 Crop Disease Bot - Farmer Chatbot Suite

A comprehensive Flask-based crop advisory assistant with multilingual NLP, voice input/output, and MongoDB-based user authentication. This application helps farmers get instant advice on crop diseases, fertilizers, and agricultural practices in multiple languages.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- MongoDB (local or MongoDB Atlas)
- Internet connection (for translation services)

### Installation Steps

1. **Navigate to the farmer_chatbot directory:**
   ```bash
   cd farmer_chatbot
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   
   # Windows CMD
   python -m venv .venv
   .venv\Scripts\activate.bat
   
   # macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements-flask.txt
   ```

4. **Optional: Install PyAudio for microphone support (Windows):**
   ```bash
   # Method 1: Using pipwin (recommended for Windows)
   pip install pipwin
   pipwin install pyaudio
   
   # Method 2: Download wheel file
   # Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
   # Download the wheel matching your Python version
   # Then: pip install PyAudio-0.2.11-cp311-cp311-win_amd64.whl
   ```
   
   **Note:** The app works fine WITHOUT PyAudio - you just won't have microphone input. Text input and all other features work normally.

5. **Set up environment variables:**
   
   Create a `.env` file in the `farmer_chatbot` directory:
   ```env
   FLASK_SECRET_KEY=your-secret-key-here
   SESSION_COOKIE_NAME=cropadvisor_session
   
   MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/?appName=Cluster0
   MONGO_DB_NAME=crop_advisor
   MONGO_USER_COLLECTION=users
   
   DEFAULT_ADMIN_USER=farmer_admin
   DEFAULT_ADMIN_PASS=demo123
   ```
   
   **Alternative:** You can also create a `.evn` file with just the MongoDB URI:
   ```
   mongodb+srv://username:password@cluster.mongodb.net/?appName=Cluster0
   ```

### Running the Application

```bash
cd farmer_chatbot
python app_flask.py
```

The application will start on **http://localhost:5000**

---

## 📋 Features

- **Multilingual Support**: Chat in 11+ Indic languages (English, Hindi, Bengali, Telugu, Tamil, etc.)
- **Voice Input/Output**: Speech-to-text and text-to-speech capabilities
- **Crop Disease Detection**: Get advice on crop diseases and treatments
- **Fertilizer Recommendations**: Receive guidance on fertilizer usage
- **User Authentication**: MongoDB-based user management with login/signup
- **Knowledge Base**: Powered by a comprehensive Q&A dataset

---

## 🔌 API Endpoints

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/` | Main web interface (login page) |
| `GET` | `/chat` | Chat interface (requires login) |
| `POST` | `/api/chat` | Chatbot API endpoint |
| `POST` | `/api/speech-to-text` | Convert audio to text |
| `POST` | `/api/text-to-speech` | Convert text to speech (MP3) |
| `GET` | `/api/languages` | List supported languages |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/login` | Login page |
| `POST` | `/login` | Process login |
| `GET` | `/signup` | Signup page |
| `POST` | `/signup` | Process signup |
| `GET` | `/logout` | Logout user |

### Example API Usage

**Chat API:**
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "How to control aphids?", "language": "en"}'
```

**Speech-to-Text:**
```bash
curl -X POST http://localhost:5000/api/speech-to-text \
  -F "audio=@recording.wav"
```

---

## 🔐 Default Login Credentials

- **Username:** `farmer_admin`
- **Password:** `demo123`

**⚠️ Important:** Change these credentials in production by setting `DEFAULT_ADMIN_USER` and `DEFAULT_ADMIN_PASS` environment variables.

---

## 🗂️ Project Structure

```
Crop_Disease_Bot/
├── farmer_chatbot/
│   ├── app_flask.py              # Main Flask application
│   ├── crop_chatbot.py           # Chat orchestration & prompt logic
│   ├── nlp_utils.py              # NLP utilities (embeddings, translation)
│   ├── voice_utils.py            # Speech-to-text & text-to-speech
│   ├── data_preprocess.py        # Dataset preprocessing
│   ├── templates/
│   │   ├── index.html           # Main chat interface
│   │   ├── login.html           # Login page
│   │   └── signup.html          # Signup page
│   ├── archive/
│   │   ├── questionsv4.csv      # Knowledge base dataset
│   │   └── questions_cleaned.csv # Processed dataset
│   ├── requirements-flask.txt    # Python dependencies
│   ├── README-FLASK.md          # Detailed Flask documentation
│   └── SETUP.md                 # Setup guide
└── README.md                     # This file
```

---

## 🛠️ Troubleshooting

### PyAudio Error
- **Error:** "Could not find PyAudio" or "No module named '_pyaudio'"
- **Solution:** This is OK! The app works without it. Voice input just won't be available. You can still use text input and all other features.

### MongoDB Connection Error
- **Error:** "Connection unavailable" or "MongoDB connection failed"
- **Solution:** 
  - Check your `.env` or `.evn` file has the correct MongoDB URI
  - Ensure your MongoDB Atlas cluster allows connections from your IP (check Network Access in Atlas)
  - Verify your MongoDB username and password are correct
  - Check internet connection

### Import Errors
- **Error:** Missing modules (e.g., `flask`, `pandas`, `nltk`)
- **Solution:** 
  ```bash
  pip install -r farmer_chatbot/requirements-flask.txt
  ```

### Translation Not Working
- **Error:** Translation features unavailable
- **Solution:** 
  - Ensure internet connection (uses Google Translate API)
  - Check firewall settings
  - The app will still work but may not translate to all languages

### Microphone Not Working in Browser
- **Solution:**
  1. Click the lock icon (🔒) in the browser address bar
  2. Find 'Microphone' → Change to 'Allow'
  3. Refresh the page
  4. Use `localhost` URL instead of IP address for better support

---

## 📚 Additional Documentation

- **Detailed Flask Documentation:** See `farmer_chatbot/README-FLASK.md`
- **Setup Guide:** See `farmer_chatbot/SETUP.md`
- **MongoDB Setup:** See `farmer_chatbot/MONGODB_SETUP.md`

---

## 🚢 Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
cd farmer_chatbot
gunicorn -w 4 -b 0.0.0.0:5000 app_flask:app
```

### Docker (Example)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY farmer_chatbot/requirements-flask.txt .
RUN pip install -r requirements-flask.txt
COPY farmer_chatbot .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app_flask:app"]
```

---

## 📝 License

This project is part of a crop disease advisory system for farmers.

---

## 🤝 Contributing

When adding new features:
1. Update `requirements-flask.txt` if adding dependencies
2. Update this README if changing setup/usage
3. Update `farmer_chatbot/README-FLASK.md` for detailed changes
4. Test thoroughly before committing

---

**Happy Farming! 🌾**
