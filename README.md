## Crop Disease Bot (Starter)

### Quick Start (Windows PowerShell)

1. Create venv (optional but recommended):
   
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   
   ```powershell
   pip install -r requirements.txt
   ```

3. Run the API (reload for development):
   
   ```powershell
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. Open docs: `http://127.0.0.1:8000/docs`

### Endpoints

- `GET /health` – health check
- `POST /detect` – form-data: `image` (file), `language` (optional, default `en`)
- `POST /chat` – form-data: `query` (text), `language` (optional)
- `POST /stt` – form-data: `audio` (file), `language` (optional)
- `POST /tts` – form-data: `text` (string), `language` (optional)

### Notes

- Image detection, STT, TTS, and NLP are placeholders to be replaced with real models.
- You can wire a CSV knowledge base later (e.g., `knowledge/advice.csv`) to map detected disease labels to actionable recommendations in multiple languages.


