# src/pipeline.py
import shutil, sys, time, zipfile
from pathlib import Path
from datetime import datetime

from config import (
    RAW_DIR, SCRATCH_DIR, AUDIO_OUT_DIR,
    PER_PID_LOG, COMPLETED, FAILED, PARTICIPANT_IDS,
)
from src.download import _session, download_zip
from src.preprocess import preprocess_one


def _log(pid: str, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(PER_PID_LOG / f"{pid}.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_set(path: Path) -> set:
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def _append(path: Path, pid: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(pid + "\n")


def _selective_extract(zip_path: Path, scratch_dir: Path) -> None:
    """Extract only *_AUDIO.wav and *_TRANSCRIPT.csv into scratch_dir (flat)."""
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not (name.endswith("_AUDIO.wav") or name.endswith("_TRANSCRIPT.csv")):
                continue
            target = scratch_dir / Path(name).name
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)


def process_one(pid: str, session) -> bool:
    zip_path = RAW_DIR / f"{pid}_P.zip"
    scratch  = SCRATCH_DIR / pid
    try:
        _log(pid, f"downloading {pid}_P.zip")
        download_zip(pid, zip_path, session)
        _log(pid, f"downloaded ({zip_path.stat().st_size/1e6:.1f} MB)")

        _log(pid, "extracting audio + transcript")
        _selective_extract(zip_path, scratch)

        _log(pid, "preprocessing")
        result = preprocess_one(pid, scratch, AUDIO_OUT_DIR)
        _log(pid, f"preprocess result: {result}")
        return bool(result.get("ok"))
    except Exception as e:
        _log(pid, f"ERROR: {e}")
        return False
    finally:
        if zip_path.exists():
            zip_path.unlink()
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)


def main():
    done   = _load_set(COMPLETED)
    failed = _load_set(FAILED)
    todo   = [p for p in PARTICIPANT_IDS if p not in done]

    print(f"Already done: {len(done)} | Previously failed: {len(failed)} | To process: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    session = _session()
    start = time.time()

    for i, pid in enumerate(todo, 1):
        t0 = time.time()
        ok = process_one(pid, session)
        dt = time.time() - t0
        if ok:
            _append(COMPLETED, pid)
            if pid in failed:
                remaining = _load_set(FAILED) - {pid}
                FAILED.write_text("\n".join(sorted(remaining)) + ("\n" if remaining else ""))
        else:
            if pid not in failed:
                _append(FAILED, pid)
        eta_min = (time.time() - start) / i * (len(todo) - i) / 60
        print(f"  [{i}/{len(todo)}] pid={pid} {'OK' if ok else 'FAIL'} "
              f"({dt:.1f}s) | ETA {eta_min:.0f} min", flush=True)


if __name__ == "__main__":
    sys.exit(0 if main() is None else 1)