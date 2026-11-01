# src/preprocess.py
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

from config import SR, MIN_SEG_SEC, PAD_SEC


def preprocess_one(pid: str, scratch_dir: Path, out_dir: Path) -> dict:
    """
    scratch_dir should contain (flat):
        {pid}_AUDIO.wav
        {pid}_TRANSCRIPT.csv
    """
    audio_path = scratch_dir / f"{pid}_AUDIO.wav"
    trans_path = scratch_dir / f"{pid}_TRANSCRIPT.csv"

    if not audio_path.exists():
        return {"pid": pid, "ok": False, "reason": "missing audio"}
    if not trans_path.exists():
        return {"pid": pid, "ok": False, "reason": "missing transcript"}

    audio, sr = sf.read(audio_path, dtype="float32")
    if sr != SR:
        return {"pid": pid, "ok": False, "reason": f"unexpected SR {sr}"}
    if len(audio) < SR * 2:
        return {"pid": pid, "ok": False, "reason": "audio too short"}

    try:
        trans = pd.read_csv(trans_path, sep="\t")
        if trans.shape[1] < 3:
            trans = pd.read_csv(trans_path)
    except Exception:
        trans = pd.read_csv(trans_path)

    trans.columns = [c.strip().lower().replace(" ", "_") for c in trans.columns]
    start_col = "start_time" if "start_time" in trans.columns else "start"
    stop_col  = "stop_time"  if "stop_time"  in trans.columns else "stop"

    scrubbed  = trans["value"].astype(str).str.contains(
        "scrubbed_entry", case=False, na=False
    )
    speaker   = trans["speaker"].astype(str).str.lower()
    keep_mask = (~speaker.str.contains("ellie")) & (~scrubbed)
    p = trans[keep_mask].copy()
    p["dur"] = p[stop_col] - p[start_col]
    p = p[p["dur"] >= MIN_SEG_SEC]

    if len(p) == 0:
        return {"pid": pid, "ok": False, "reason": "no valid segments"}

    chunks = []
    for _, r in p.iterrows():
        s = max(0, int((r[start_col] - PAD_SEC) * SR))
        e = min(len(audio), int((r[stop_col] + PAD_SEC) * SR))
        if e <= s:
            continue
        c = audio[s:e]
        if np.mean(np.abs(c) > 1e-6) > 0.01:
            chunks.append(c)

    if not chunks:
        return {"pid": pid, "ok": False, "reason": "all chunks silent"}

    merged = np.concatenate(chunks)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / f"{pid}.wav", merged, SR)

    return {
        "pid": pid,
        "ok": True,
        "speech_sec": round(len(merged) / SR, 1),
        "n_segments": len(chunks),
    }