"""
UBFC-rPPG Dataset Downloader
============================
Downloads all UBFC-rPPG dataset files/folders from Google Drive into the
correct subfolders. Supports resume and authenticated Google Drive access.

Uses browser cookies (cookies.txt) for authenticated downloads to bypass
Google Drive's anonymous quota limits.  Patching gdown's internal download
function to route through our authenticated requests session.

Files downloaded:
  - scripts/ubfcrppg_data_processor.py
  - scripts/ubfcrppg_data_processor.m
  - scripts/readme.txt
  - scripts/Agreement.xlsx
  - UBFC1/  (DATASET_1 folder — many subfolders with vid.avi + gtdump.xmp)
  - UBFC2/  (DATASET_2 folder — same structure)

Requires: pip install gdown requests
"""

from __future__ import annotations

import re
import sys
import csv
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(
    r"C:\Users\Asus\Desktop\intel multimodal (vision layer)\UBFC_rPPG\datasets"
)

COOKIE_FILE = BASE_DIR / "cookies.txt"

# Individual files: (drive_file_id, filename, subfolder)
SINGLE_FILES = [
    ("12jym67QYPnXHp3qSRjf1XsQGWRrC284M", "ubfcrppg_data_processor.py", "scripts"),
    ("1oHkc9-U64XHU7qvBlmvpsDhEqSwhA1EQ", "ubfcrppg_data_processor.m", "scripts"),
    ("1oL9oy_FU7jbB3qliHQIW629GtEMPJ64q", "readme.txt", "scripts"),
    ("1oT_t9EHB_SzqwoQt-7VWvyLpFgCoLZT1", "Agreement.xlsx", "scripts"),
]

# Folder downloads: (drive_folder_id, subfolder_name)
FOLDERS = [
    ("1oD7Szb9wE6XbUN9SWq6EmyH9WDnKI1Pk", "UBFC1"),  # DATASET_1
    ("1q4vWuF2GJvKP5xyeX8dxaJ2fmq97-4ai", "UBFC2"),  # DATASET_2
]

SIZE_CSV = BASE_DIR / "size.csv"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Safari/537.36"
    ),
}


# ---------------------------------------------------------------------------
# Cookie loading & session creation
# ---------------------------------------------------------------------------

def load_cookies() -> dict:
    """Load cookies from COOKIE_FILE. Supports Netscape and raw header formats."""
    cookies = {}
    if not COOKIE_FILE.is_file():
        return cookies
    raw = COOKIE_FILE.read_text(encoding="utf-8").strip()
    if not raw:
        return cookies
    for line in raw.split("\n"):
        line = line.strip()
        if not line or line.startswith("# "):
            continue
        if "\t" in line:                    # Netscape format
            parts = line.split("\t")
            if len(parts) >= 7:
                cookies[parts[5]] = parts[6]
                continue
        line = re.sub(r"^Cookie:\s*", "", line)  # raw Cookie: header
        for pair in line.split(";"):
            pair = pair.strip()
            if "=" in pair:
                name, _, value = pair.partition("=")
                cookies[name.strip()] = value.strip()
    return cookies


def make_session() -> "requests.Session":
    """Create a requests Session preloaded with browser cookies and headers."""
    import requests
    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)
    for name, value in load_cookies().items():
        session.cookies.set(name, value, domain=".google.com")
    return session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_drive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}"


def verify_size(filepath: Path, expected_sizes: dict) -> None:
    filename = filepath.name
    actual = filepath.stat().st_size
    print(f"    Size: {actual:,} bytes")
    if filename in expected_sizes:
        expected = int(expected_sizes[filename])
        if actual == expected:
            print(f"    Integrity OK — matches expected {expected:,} bytes")
        else:
            print(f"    WARNING: size mismatch! Exp {expected:,}, got {actual:,}")
    else:
        print("    (No expected size in size.csv — cannot verify integrity)")


def read_size_csv(csv_path: Path) -> dict:
    if not csv_path.is_file():
        return {}
    sizes = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                n = row.get("filename", "").strip()
                s = row.get("size_bytes", "").strip()
                if n and s:
                    sizes[n] = s
    except Exception as exc:
        print(f"WARNING: Could not parse {csv_path}: {exc}")
    return sizes


# ---------------------------------------------------------------------------
# Single-file download (authenticated requests session)
# ---------------------------------------------------------------------------

def _extract_form_fields(html_bytes: bytes) -> dict:
    """Extract hidden <input> fields from a Google Drive HTML page."""
    fields = {}
    for m in re.finditer(rb'<input\s+[^>]*type=["\']hidden["\'][^>]*>', html_bytes):
        tag = m.group(0)
        nm = re.search(rb'name=["\']([^"\']+)["\']', tag)
        vm = re.search(rb'value=["\']([^"\']*)["\']', tag)
        if nm:
            fields[nm.group(1).decode()] = vm.group(1).decode() if vm else ""
    if "confirm" not in fields:
        for pat in [rb'name="confirm"\s+value="([^"]+)"', rb"confirm=([0-9A-Za-z_-]+)"]:
            m = re.search(pat, html_bytes)
            if m:
                fields["confirm"] = m.group(1).decode()
                break
    return fields


def _extract_form_action(html_bytes: bytes) -> str | None:
    m = re.search(rb'<form\s+[^>]*action=["\']([^"\']+)["\']', html_bytes)
    if not m:
        m = re.search(rb'action=["\']([^"\']+)["\']', html_bytes)
    return m.group(1).decode() if m else None


def download_single_file(session, file_id: str, dest_path: Path) -> bool:
    """
    Download a single Google Drive file using an authenticated session.
    Handles virus-scan warning pages (confirm token flow).
    """
    import requests

    base = "https://docs.google.com/uc?export=download"

    try:
        resp = session.get(base, params={"id": file_id}, stream=True, timeout=60)
    except requests.RequestException:
        return False

    if resp.status_code != 200:
        resp.close()
        return False

    ct = resp.headers.get("Content-Type", "")
    first = resp.raw.read(512)
    is_html = "text/html" in ct or (first and first.strip().startswith(b"<!DOCTYPE"))

    if is_html:
        full = first + resp.raw.read()
        resp.close()
        fields = _extract_form_fields(full)
        if not fields or "confirm" not in fields:
            return False
        action_url = _extract_form_action(full) or base
        try:
            resp2 = session.get(action_url, params=fields, stream=True, timeout=60)
        except requests.RequestException:
            return False
        if resp2.status_code != 200 or "text/html" in resp2.headers.get("Content-Type", ""):
            resp2.close()
            return False
        return _write_stream(resp2, dest_path)

    return _write_stream_with_head(resp, first, dest_path)


def _write_stream(resp, dest_path: Path) -> bool:
    try:
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=131_072):
                if chunk:
                    fh.write(chunk)
    except OSError:
        resp.close()
        return False
    resp.close()
    return dest_path.exists() and dest_path.stat().st_size > 0


def _write_stream_with_head(resp, head: bytes, dest_path: Path) -> bool:
    try:
        with open(dest_path, "wb") as fh:
            fh.write(head)
            for chunk in resp.iter_content(chunk_size=131_072):
                if chunk:
                    fh.write(chunk)
    except OSError:
        resp.close()
        return False
    resp.close()
    return dest_path.exists() and dest_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Folder download (gdown listing + authenticated download)
# ---------------------------------------------------------------------------

def download_folder(folder_id: str, output_dir: Path) -> tuple[int, int]:
    """
    Download a Google Drive folder using gdown for folder-traversal
    but our authenticated session for the actual file downloads.
    Returns (downloaded, failed).
    """
    import sys as _sys
    import gdown

    # Silence gdown's BeautifulSoup warnings
    import warnings
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    session = make_session()

    # ── Patch gdown's download function ──
    dlf_mod = _sys.modules["gdown.download_folder"]
    dl_mod = _sys.modules["gdown.download"]
    _orig_dlf = dlf_mod.download
    _orig_dl = dl_mod.download
    _orig_gdown = gdown.download

    def _patched_download(url=None, output=None, quiet=False, fuzzy=False,
                          id=None, **kwargs):
        fid = id
        if not fid and url:
            m = re.search(r'[?&]id=([\w-]{25,})', url)
            if m:
                fid = m.group(1)
        if fid:
            dest = Path(output) if output else Path(f"{fid}.tmp")
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Skip if already downloaded
            if dest.exists() and dest.stat().st_size > 0:
                return str(dest)
            # Try authenticated download up to 3 times with delay
            for attempt in range(3):
                if download_single_file(session, fid, dest):
                    time.sleep(0.5)  # small delay to avoid rate limits
                    return str(dest)
                time.sleep(2)
            # Failed — return None so gdown continues to next file
            return None
        # Fall through to original gdown
        return _orig_dlf(url=url, output=output, quiet=True, fuzzy=fuzzy,
                         id=id, **kwargs)

    dlf_mod.download = _patched_download
    dl_mod.download = _patched_download
    gdown.download = _patched_download

    try:
        result = gdown.download_folder(
            id=folder_id, output=str(output_dir), quiet=True, skip_download=False,
        )
        n = len(result) if result else 0
        return n, 0
    except Exception as exc:
        # gdown stopped — count what we got
        downloaded = sum(1 for f in output_dir.rglob("*") if f.is_file())
        print(f"    gdown batch done: {downloaded} files so far")
        # Try remaining via fallback
        remaining = _try_download_remaining(session, folder_id, output_dir)
        return downloaded + remaining[0], remaining[1]
    finally:
        dlf_mod.download = _orig_dlf
        dl_mod.download = _orig_dl
        gdown.download = _orig_gdown


def _try_download_remaining(session, folder_id: str, output_dir: Path) -> tuple[int, int]:
    """
    Fallback: re-invoke gdown.download_folder in a loop.
    Each invocation processes another batch of files before hitting rate limits.
    Stops when two consecutive invocations produce no new files.
    """
    import gdown
    import sys as _sys

    dlf_mod = _sys.modules["gdown.download_folder"]
    dl_mod = _sys.modules["gdown.download"]
    _orig_dlf = dlf_mod.download
    _orig_dl = dl_mod.download
    _orig_gdown = gdown.download

    # Same patched download as above
    def _retry_download(url=None, output=None, quiet=False, fuzzy=False,
                        id=None, **kwargs):
        fid = id
        if not fid and url:
            m = re.search(r'[?&]id=([\w-]{25,})', url)
            if m:
                fid = m.group(1)
        if fid:
            dest = Path(output) if output else Path(f"{fid}.tmp")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size > 0:
                return str(dest)
            for attempt in range(3):
                if download_single_file(session, fid, dest):
                    time.sleep(0.5)
                    return str(dest)
                time.sleep(2)
            return None
        return _orig_dlf(url=url, output=output, quiet=True, fuzzy=fuzzy,
                         id=id, **kwargs)

    dlf_mod.download = _retry_download
    dl_mod.download = _retry_download
    gdown.download = _retry_download

    total_downloaded = 0
    dry_runs = 0

    try:
        for batch in range(10):  # max 10 batches
            before = sum(1 for f in output_dir.rglob("*") if f.is_file())
            try:
                gdown.download_folder(
                    id=folder_id, output=str(output_dir), quiet=True,
                )
            except Exception:
                pass  # expected when rate-limited
            after = sum(1 for f in output_dir.rglob("*") if f.is_file())
            new = after - before
            total_downloaded += new
            if new == 0:
                dry_runs += 1
                if dry_runs >= 3:
                    break
                time.sleep(5)
            else:
                dry_runs = 0
                time.sleep(2)
    finally:
        dlf_mod.download = _orig_dlf
        dl_mod.download = _orig_dl
        gdown.download = _orig_gdown

    return total_downloaded, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Check dependencies
    missing = []
    for lib in ("gdown", "requests"):
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        print(f"ERROR: Missing: {', '.join(missing)}")
        print(f"Install:  pip install {' '.join(missing)}")
        return 1

    cookies = load_cookies()
    if not cookies:
        print("NOTE: No cookies.txt found — downloads will be anonymous.")
        print(f"     Place browser cookies at: {COOKIE_FILE}")
    else:
        print(f"Loaded {len(cookies)} cookies for authenticated access.", flush=True)

    print("=" * 60)
    print(" UBFC-rPPG Dataset Downloader")
    print("=" * 60)
    print(f"Base: {BASE_DIR}\n")

    expected_sizes = read_size_csv(SIZE_CSV)
    session = make_session()
    results = {"downloaded": [], "skipped": [], "failed": []}

    # ── Phase 1: Single files ──
    print("─" * 40)
    print(" Single files (scripts & docs)")
    print("─" * 40)
    for file_id, filename, subfolder in SINGLE_FILES:
        folder = BASE_DIR / subfolder
        dest = folder / filename
        folder.mkdir(parents=True, exist_ok=True)

        print(f"\n[{subfolder}] {filename}")

        if dest.exists():
            print(f"  Already exists — skipping")
            results["skipped"].append(filename)
            verify_size(dest, expected_sizes)
            continue

        part = folder / (filename + ".part")
        part.unlink(missing_ok=True)

        print(f"  Downloading...", end=" ", flush=True)
        if download_single_file(session, file_id, part):
            part.rename(dest)
            print(f"OK ({dest.stat().st_size:,} bytes)")
            results["downloaded"].append(filename)
            verify_size(dest, expected_sizes)
        else:
            print(f"FAILED")
            print(f"  Open: {build_drive_url(file_id)}")
            results["failed"].append(filename)
            part.unlink(missing_ok=True)

    # ── Phase 2: Folder downloads ──
    print(f"\n{'─' * 40}")
    print(" Folder downloads (datasets)")
    print("─" * 40)
    for folder_id, subfolder in FOLDERS:
        output_dir = BASE_DIR / subfolder
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = list(output_dir.rglob("*"))
        existing_files = [f for f in existing if f.is_file()]

        print(f"\n[{subfolder}] DATASET folder")
        if existing_files:
            print(f"  {len(existing_files)} files already present — will skip those")

        print(f"  Scanning & downloading...")
        dl, fail = download_folder(folder_id, output_dir)

        all_files = list(output_dir.rglob("*"))
        new_files = [f for f in all_files if f.is_file()]
        print(f"  Result: {len(new_files)} files total ({dl} new, {fail} failed)")

        if fail == 0:
            results["downloaded"].append(f"{subfolder}/ ({dl} files)")
        elif dl > 0:
            results["downloaded"].append(f"{subfolder}/ ({dl} files)")
            results["failed"].append(f"{subfolder}/ ({fail} failed)")
        else:
            results["failed"].append(f"{subfolder}/")
            print(f"  Browser: https://drive.google.com/drive/folders/{folder_id}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(" SUMMARY")
    print("=" * 60)
    print(f"  Downloaded : {len(results['downloaded'])}")
    print(f"  Skipped    : {len(results['skipped'])}")
    print(f"  Failed     : {len(results['failed'])}")

    if results["failed"]:
        print("\n  Manual download links:")
        for name in results["failed"]:
            for fid, fname, _ in SINGLE_FILES:
                if fname == name:
                    print(f"    {build_drive_url(fid)}  →  {name}")
                    break
            else:
                for fid, sf in FOLDERS:
                    if name.startswith(sf):
                        print(f"    https://drive.google.com/drive/folders/{fid}  →  {name}")
                        break
        return 1

    print("\nAll tasks completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
