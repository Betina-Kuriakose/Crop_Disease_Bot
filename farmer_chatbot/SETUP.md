# Quick Setup Guide

## Installation Steps

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements-flask.txt
   ```

2. **Optional: Install PyAudio for microphone support (Windows):**
   
   If you want voice input features, install PyAudio:
   ```bash
   # Method 1: Using pipwin (recommended for Windows)
   pip install pipwin
   pipwin install pyaudio
   
   # Method 2: Download wheel file
   # Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
   # Download the wheel matching your Python version (e.g., PyAudio-0.2.11-cp311-cp311-win_amd64.whl)
   # Then: pip install PyAudio-0.2.11-cp311-cp311-win_amd64.whl
   ```
   
   **Note:** The app works fine WITHOUT PyAudio - you just won't have microphone input. Text input and all other features work normally.

3. **Ensure `.evn` file exists with MongoDB URI:**
   ```
   mongodb+srv://betinafareago:betinafareago@cluster0.o4scfv6.mongodb.net/?appName=Cluster0
   ```

## Running the Application

```bash
cd farmer_chatbot
python app_flask.py
```

The app will start on `http://localhost:5000`

## Default Login Credentials

- **Username:** `farmer_admin`
- **Password:** `demo123`

(Change these in production by setting `DEFAULT_ADMIN_USER` and `DEFAULT_ADMIN_PASS` environment variables)

## Troubleshooting

### PyAudio Error
- **Error:** "Could not find PyAudio"
- **Solution:** This is OK! The app works without it. Voice input just won't be available. You can still use text input.

### MongoDB Connection Error
- **Error:** "Connection unavailable"
- **Solution:** 
  - Check your `.evn` file has the correct MongoDB URI
  - Ensure your MongoDB Atlas cluster allows connections from your IP
  - Check internet connection

### Import Errors
- **Error:** Missing modules
- **Solution:** Run `pip install -r requirements-flask.txt` again

