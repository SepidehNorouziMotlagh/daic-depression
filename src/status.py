# src/status.py
from config import COMPLETED, FAILED, AUDIO_OUT_DIR, PARTICIPANT_IDS

done   = {ln.strip() for ln in COMPLETED.read_text().splitlines() if ln.strip()}
failed = {ln.strip() for ln in FAILED.read_text().splitlines() if ln.strip()}
wavs   = list(AUDIO_OUT_DIR.glob("*.wav"))
total_mb = sum(p.stat().st_size for p in wavs) / 1e6

print(f"Total participants : {len(PARTICIPANT_IDS)}")
print(f"Completed          : {len(done)}")
print(f"Failed             : {len(failed)}")
print(f"WAV files on disk  : {len(wavs)}")
print(f"Total cleaned audio: {total_mb:.1f} MB")
if failed:
    print(f"Failed PIDs        : {', '.join(sorted(failed))}")