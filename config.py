# config.py
from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA          = ROOT / "data"
RAW_DIR       = DATA / "raw"
SCRATCH_DIR   = DATA / "scratch"
AUDIO_OUT_DIR = DATA / "participant_audio"
MANIFEST_DIR  = DATA / "manifests"
LOG_DIR       = ROOT / "logs"
PER_PID_LOG   = LOG_DIR / "per_participant"
COMPLETED     = LOG_DIR / "completed.txt"
FAILED        = LOG_DIR / "failed.txt"

for d in (RAW_DIR, SCRATCH_DIR, AUDIO_OUT_DIR, MANIFEST_DIR, PER_PID_LOG):
    d.mkdir(parents=True, exist_ok=True)

USC_BASE_URL = os.getenv("USC_BASE_URL", "").rstrip("/")

SR          = 16000
MIN_SEG_SEC = 1.0
PAD_SEC     = 0.05

# Participant IDs to attempt. Temporarily narrow for the first test (e.g., ["300"]).
PARTICIPANT_IDS = ['440']