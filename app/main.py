from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional

from .services.image_detection import detect_disease_from_image
from .services.nlp_bot import answer_farmer_question
from .services.stt import transcribe_audio
from .services.tts import synthesize_speech


app = FastAPI(title="Crop Disease Bot", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/detect")
async def detect(image: UploadFile = File(...), language: Optional[str] = Form("en")) -> JSONResponse:
    try:
        image_bytes = await image.read()
        result = detect_disease_from_image(image_bytes=image_bytes, language=language or "en")
        return JSONResponse(result)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/chat")
async def chat(query: str = Form(...), language: Optional[str] = Form("en")) -> JSONResponse:
    try:
        answer = answer_farmer_question(query=query, language=language or "en")
        return JSONResponse({"answer": answer})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/stt")
async def stt(audio: UploadFile = File(...), language: Optional[str] = Form("en")) -> JSONResponse:
    try:
        audio_bytes = await audio.read()
        text = transcribe_audio(audio_bytes=audio_bytes, language=language or "en")
        return JSONResponse({"text": text})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/tts")
async def tts(text: str = Form(...), language: Optional[str] = Form("en")) -> JSONResponse:
    try:
        # For now we return a placeholder. A real implementation would return audio bytes (e.g. WAV/MP3)
        audio_info = synthesize_speech(text=text, language=language or "en")
        return JSONResponse({"audio": audio_info})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=400)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


