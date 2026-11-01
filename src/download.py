# src/download.py
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import USC_BASE_URL


def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=4, backoff_factor=2.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def download_zip(pid: str, dest_path: Path, session: requests.Session) -> None:
    """
    Downloads {USC_BASE_URL}/{pid}_P.zip to dest_path. Resumes if a partial file exists.
    Raises RuntimeError on failure.
    """
    if not USC_BASE_URL:
        raise RuntimeError("USC_BASE_URL not set in .env")

    url = f"{USC_BASE_URL}/{pid}_P.zip"

    headers = {}
    existing = 0
    if dest_path.exists():
        existing = dest_path.stat().st_size
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"

    with session.get(url, headers=headers, stream=True, timeout=120,
                     allow_redirects=True) as r:
        if r.status_code == 416:            # already complete
            return
        if r.status_code == 404:
            raise RuntimeError(f"404 Not Found for {url}")
        if r.status_code in (401, 403):
            raise RuntimeError(f"{r.status_code} on {url} — unexpected; verify USC_BASE_URL")
        if r.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {r.status_code} for {url}")

        # Sanity check: ensure we're getting a zip, not an HTML directory listing
        ctype = r.headers.get("Content-Type", "").lower()
        if "text/html" in ctype:
            raise RuntimeError(f"Got HTML instead of zip for {url} — check USC_BASE_URL")

        mode = "ab" if r.status_code == 206 else "wb"
        total = int(r.headers.get("Content-Length", 0)) + (existing if mode == "ab" else 0)
        written = existing if mode == "ab" else 0

        with open(dest_path, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)

        if total > 0 and written < total:
            raise RuntimeError(f"incomplete download ({written}/{total} bytes)")