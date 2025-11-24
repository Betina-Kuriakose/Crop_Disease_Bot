# 🌾 Crop Chatbot for Farmers - Flask Version

A production-ready Flask application for a crop farming chatbot with multi-language support, voice input/output, and REST API.

## Features

✅ **Multi-Language Support**: 
- Auto-detect language or choose from 11 Indian languages
- Automatic translation using Google Translate
- Supports: English, Hindi, Tamil, Telugu, Marathi, Gujarati, Bengali, Kannada, Malayalam, Punjabi, Urdu

✅ **Voice Features**:
- **Speech-to-Text (STT)**: Speak your questions using browser microphone
- **Text-to-Speech (TTS)**: Hear responses in audio format

✅ **REST API**:
- `/api/chat` - Chat endpoint
- `/api/speech-to-text` - Convert audio to text
- `/api/text-to-speech` - Convert text to audio
- `/api/languages` - Get supported languages
- `/health` - Health check

✅ **Production Ready**:
- Flask backend with CORS support
- Beautiful, responsive HTML/CSS/JavaScript frontend
- Easy to deploy and scale

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements-flask.txt
   ```

2. **Ensure dataset is available**:
   - Dataset should be at: `archive/questionsv4.csv`

## Running the Application

```bash
python app_flask.py
```

The application will start on `http://localhost:5000`

## API Endpoints

### POST `/api/chat`
Send a text message and get a response.

**Request:**
```json
{
    "text": "How to control aphid infestation in mustard crops?",
    "language": "en"
}
```

**Response:**
```json
{
    "reply": "Suggested him to spray rogor@2ml/lit.at evening time.",
    "meta": {
        "original_reply_en": "Suggested him to spray rogor@2ml/lit.at evening time.",
        "language_detected": "en",
        "score": 0.85,
        "question": "asking about the control measure for aphid infestation in mustard crops"
    }
}
```

### POST `/api/speech-to-text`
Convert audio file to text.

**Request:** Form data with `audio` file

**Response:**
```json
{
    "text": "How to control aphid infestation",
    "success": true
}
```

### POST `/api/text-to-speech`
Convert text to speech audio file.

**Request:**
```json
{
    "text": "Suggested him to spray rogor",
    "language": "en"
}
```

**Response:** Audio file (MP3)

### GET `/api/languages`
Get list of supported languages.

### GET `/health`
Health check endpoint.

## Frontend Features

- **Beautiful UI**: Modern, responsive design
- **Multi-language**: Select language from dropdown or auto-detect
- **Voice Input**: Click microphone button to speak
- **Text-to-Speech**: Enable TTS to hear responses
- **Chat History**: View conversation history
- **Real-time**: Instant responses with loading indicators

## Deployment

### Using Gunicorn (Production)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app_flask:app
```

### Using Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-flask.txt .
RUN pip install -r requirements-flask.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app_flask:app"]
```

## Advantages over Streamlit

1. **Production Ready**: Better for deployment and scaling
2. **REST API**: Can be integrated with mobile apps, other services
3. **Multi-language**: Built-in translation support
4. **Flexible Frontend**: Full control over UI/UX
5. **Performance**: Better for high-traffic scenarios
6. **API Access**: Can be used by multiple clients

