import os
import uuid
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from TTS.api import TTS
from pydantic import BaseModel
from fastapi import Body

# Set up FastAPI
app = FastAPI()

# Define request body model
class TextRequest(BaseModel):
    text: str

@app.post("/text_to_speech")
async def text_to_speech(request: TextRequest = Body(...)):
    """Converts text to speech using Coqui TTS and returns a WAV file."""
    try:
        job_id = str(uuid.uuid4())
        output_dir = Path(f"output/{job_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "speech.wav"

        # Use a high-quality AI voice model
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")  # Free human-like voice
        tts.tts_to_file(text=request.text, file_path=str(output_path))

        time.sleep(2)  # Ensure file writes correctly

        # Check if file exists
        if not output_path.exists() or output_path.stat().st_size < 2000:
            raise HTTPException(status_code=500, detail="Coqui TTS failed to generate an audio file.")

        return FileResponse(output_path, media_type="audio/wav", filename="speech.wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
