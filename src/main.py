import os
import uuid
import time
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from TTS.api import TTS
from pydantic import BaseModel
from fastapi import Body
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up FastAPI
app = FastAPI()

# ✅ Available models:
# - "tts_models/en/ljspeech/tacotron2-DDC" -> Single-speaker model (NO speaker needed)
# - "tts_models/en/vctk/vits"              -> Multi-speaker model (REQUIRES speaker)
# - "tts_models/en/ljspeech/glow-tts"      -> Single-speaker AI voice
# - "tts_models/en/vctk/glow-tts"          -> Multi-speaker expressive AI speech
TTS_MODEL =  "tts_models/en/vctk/vits"  # Change this to test different voices

# Default speaker (only for multi-speaker models)
DEFAULT_SPEAKER_ID = "p226"  # Change this based on available speakers

class TextRequest(BaseModel):
    text: str

async def generate_tts(text: str, output_path: Path):
    """Handles text-to-speech generation asynchronously."""
    try:
        logging.info(f"🔵 Loading TTS Model: {TTS_MODEL}")
        tts = TTS(TTS_MODEL)  # Load model

        # Check if model supports speakers
        if hasattr(tts, "speakers") and tts.speakers:
            logging.info(f"🎤 Multi-speaker model detected. Using speaker: {DEFAULT_SPEAKER_ID}")
            tts.tts_to_file(text=text, file_path=str(output_path), speaker=DEFAULT_SPEAKER_ID)
        else:
            logging.info("🎤 Single-speaker model detected. No speaker required.")
            tts.tts_to_file(text=text, file_path=str(output_path))

        logging.info(f"✅ Speech saved to {output_path}")
        return True
    except Exception as e:
        logging.error(f"❌ Error generating speech: {e}")
        return False

@app.post("/text_to_speech")
async def text_to_speech(request: TextRequest = Body(...)):
    """Converts text to speech using Coqui TTS and returns a WAV file."""
    try:
        job_id = str(uuid.uuid4())
        output_dir = Path(f"output/{job_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "speech.wav"

        logging.info("🔵 Starting TTS job...")
        success = await asyncio.wait_for(generate_tts(request.text, output_path), timeout=900)  # 15 min timeout

        if not success:
            raise HTTPException(status_code=500, detail="Coqui TTS failed to generate an audio file.")

        logging.info("✅ TTS job completed successfully!")

        return FileResponse(output_path, media_type="audio/wav", filename="speech.wav")

    except asyncio.TimeoutError:
        logging.error("⏳ Timeout: TTS generation took too long!")
        raise HTTPException(status_code=408, detail="TTS generation timed out. Try a shorter text.")
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logging.info("🚀 Starting FastAPI server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
