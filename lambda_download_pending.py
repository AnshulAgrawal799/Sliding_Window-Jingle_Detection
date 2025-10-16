# lambda_download_pending.py
import os
import re
import requests
from urllib.parse import unquote, urlparse
from db_mysql import get_connection

def fetch_pending(limit=5, conn=None):
    """Fetch rows with process_status = 'Pending'."""
    if conn is None:
        conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, s3_uri FROM audio_file WHERE process_status = %s LIMIT %s",
            ("Pending", limit),
        )
        return cur.fetchall()
    finally:
        if conn:
            conn.close()

def _sanitize_filename(name: str) -> str:
    # remove query string already done upstream, but also replace problematic chars
    # allow letters, numbers, space, dot, dash, underscore
    return re.sub(r"[^A-Za-z0-9 ._\-()]+", "_", name).strip()

def _unique_path(dirpath: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = filename
    i = 1
    while os.path.exists(os.path.join(dirpath, candidate)):
        candidate = f"{base}_{i}{ext}"
        i += 1
    return os.path.join(dirpath, candidate)

def update_process_status(record_id, status, conn=None):
    """Update process_status for a specific record."""
    if conn is None:
        conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE audio_file SET process_status = %s WHERE id = %s",
            (status, record_id),
        )
        conn.commit()
    finally:
        if conn:
            conn.close()

def download_file(url, record_id):
    url_decoded = unquote(url)
    parsed = urlparse(url_decoded)
    # get the basename (drop any query parameters)
    filename = os.path.basename(parsed.path) or "download.audio"
    filename = _sanitize_filename(filename)
    # Prefix with DB record id for stable downstream mapping (e.g., "15_filename.mp3")
    if not str(filename).startswith(f"{record_id}_"):
        filename = f"{record_id}_" + filename

    # Use Lambda's /tmp directory for temporary storage
    script_folder = "/tmp/audio"
    os.makedirs(script_folder, exist_ok=True)

    dest_path = _unique_path(script_folder, filename)
    temp_path = dest_path + ".part"

    try:
        with requests.get(url_decoded, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = 0
            with open(temp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)

        # rename temp file to final destination (atomic on most OSes)
        os.replace(temp_path, dest_path)
        print(
            f"[{record_id}] Downloaded {url_decoded} -> {dest_path} ({total} bytes)")
        return dest_path
    except Exception:
        # cleanup partial file on error
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise

def lambda_handler(event, context):
    # Extract limit from event, default to 5
    limit = event.get('limit', 5) if event else 5

    rows = fetch_pending(limit=limit)
    print(f"Found {len(rows)} Pending rows")
    for r in rows:
        rid = r["id"]
        url = r["s3_uri"]
        if not url:
            print(f"[{rid}] Missing s3_uri, skipping")
            continue
        try:
            download_file(url, rid)
            update_process_status(rid, "InProgress")
        except Exception as e:
            print(f"[{rid}] Failed to download: {e}")

    return {
        'statusCode': 200,
        'body': f'Processed {len(rows)} files'
    }
