from celery import Celery
import time
from pathlib import Path
import uuid

celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "sample"
TARGET_DIR = BASE_DIR / "target"


@celery_app.task(bind=True)
def inference_task(self, target_text: str):
    unique_id = str(uuid.uuid4())
    directory = TARGET_DIR / str(unique_id)

    try:
        directory.mkdir(parents=True, exist_ok=True)
        text_file = directory / "target.txt"
        wav_file = directory / "target.wav"
        
        time.sleep(5)

        with text_file.open("w", encoding="utf-8") as f:
            f.write(target_text)
            
        with wav_file.open("w", encoding="utf-8") as f:
            f.write(b"RIFF....WAVEfmt ")

        # inference

        return {"receipt_id": str(unique_id), "status": "completed"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}
