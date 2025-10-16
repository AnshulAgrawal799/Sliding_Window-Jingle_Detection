# lambda_db_mysql.py
import os
import re
import json
from typing import Optional, Mapping, Any

# Try to load .env using python-dotenv if it's installed. This is opportunistic:
# failure to import dotenv is non-fatal and we simply fall back to existing os.environ.
try:
    from dotenv import find_dotenv, load_dotenv

    _dotenv_path = find_dotenv()  # searches upwards for a .env file
    if _dotenv_path:
        # load into os.environ without overriding existing env vars
        load_dotenv(_dotenv_path, override=False)
except Exception:
    # ignore any errors and continue using os.environ
    pass

MIN_VERSION = "8.0.23"
PKG_NAME = "mysql-connector-python"

def _env(name: str) -> Optional[str]:
    """
    Resolve a configuration value by checking real environment variables (os.environ).
    """
    return os.getenv(name)

def _get_config():
    host = _env("DB_HOST")
    port = int(_env("DB_PORT") or 3306)
    user = _env("DB_USER")
    password = _env("DB_PASS")
    database = _env("DB_NAME")
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
        # helpful defaults
        "charset": "utf8mb4",
        "use_unicode": True,
    }

def _parse_version_to_tuple(v: str):
    """
    Convert version string like '8.0.23' or '8.0.23a' to tuple of ints (8,0,23).
    Non-numeric suffixes are ignored. This is a simple comparator sufficient for
    comparing major.minor.patch for typical mysql-connector-python versions.
    """
    # Keep only the numeric parts at start like 8.0.23 from 8.0.23a
    m = re.match(r"^([0-9]+(?:\.[0-9]+)*)", v)
    if not m:
        return ()
    parts = m.group(1).split(".")
    try:
        return tuple(int(x) for x in parts)
    except ValueError:
        # fallback to naive parsing
        return tuple(int(re.sub(r"\D", "", x) or 0) for x in parts)

def _get_installed_package_version(pkg_name: str) -> Optional[str]:
    """
    Try importlib.metadata (py3.8+) then pkg_resources; return version string or None.
    """
    try:
        # Python 3.8+
        from importlib import metadata

        return metadata.version(pkg_name)
    except Exception:
        try:
            # setuptools pkg_resources fallback
            import pkg_resources

            return pkg_resources.get_distribution(pkg_name).version
        except Exception:
            return None

def _ensure_connector_present_and_ok():
    ver = _get_installed_package_version(PKG_NAME)
    if ver is None:
        raise ImportError(
            f"'{PKG_NAME}' is not installed. Install it with:\n\n"
            f"    pip install \"{PKG_NAME}>={MIN_VERSION}\"\n"
        )
    inst = _parse_version_to_tuple(ver)
    required = _parse_version_to_tuple(MIN_VERSION)
    if not inst or inst < required:
        raise ImportError(
            f"Installed '{PKG_NAME}' version is {ver!r} but this project requires >= {MIN_VERSION}.\n"
            f"Please upgrade with:\n\n"
            f"    pip install --upgrade \"{PKG_NAME}>={MIN_VERSION}\"\n"
        )
    # If we reach here, version is acceptable

def get_connection():
    """
    Return a new mysql.connector connection object.

    Raises ImportError if mysql-connector-python is missing or too old.
    """
    # verify package is installed and at least MIN_VERSION
    _ensure_connector_present_and_ok()

    try:
        import mysql.connector
    except Exception as exc:
        raise ImportError(
            "Unable to import mysql.connector even though the package appeared to be installed.\n"
            "Ensure your Python environment is correct and that mysql-connector-python is installed.\n\n"
            f"Original error: {exc}"
        ) from exc

    cfg = _get_config()
    try:
        conn = mysql.connector.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
            charset=cfg["charset"],
            use_unicode=cfg["use_unicode"],
            connection_timeout=30,
        )
        return conn
    except mysql.connector.Error as e:
        raise RuntimeError(
            "Failed to connect to MySQL. Please check your DB_* env vars and that the "
            "MySQL server is reachable. Original error: " + str(e)
        ) from e

def insert_segment(segment: Mapping[str, Any], conn: Optional["mysql.connector.MySQLConnection"] = None) -> None:
    """
    Insert a single segment into the `segment` table.

    Expects `segment` in the shape produced by Segment.to_dict() from src/analyze.py.
    Field mapping performed here to match DB columns.

    If `conn` is not provided, this function will open a new connection and close it after commit.
    """
    # Lazy import for type and to avoid mandatory dependency at module import time
    _ensure_connector_present_and_ok()
    import mysql.connector  # type: ignore

    own_conn = False
    if conn is None:
        conn = get_connection()
        own_conn = True

    try:
        cur = conn.cursor()

        # Map fields from pipeline dict to DB columns
        seg_id = segment.get('segment_id')
        audio_file_id = segment.get('audio_file_id')
        speaker_role = segment.get('speaker_role')
        role_confidence = segment.get('role_confidence')
        start_ms = segment.get('start_ms')
        end_ms = segment.get('end_ms')
        duration_ms = segment.get('duration_ms')
        asr_confidence = segment.get('asr_confidence')
        lang = segment.get('language') or segment.get('lang')
        text_original = segment.get('text_original') or segment.get('textTamil') or ''
        translations = segment.get('translations')
        products = segment.get('products')
        intent = segment.get('intent')
        sentiment = segment.get('sentiment')
        emotion = segment.get('emotion')
        emotions = segment.get('emotions')
        needs_review = segment.get('needs_human_review') or segment.get('needs_review') or False
        review_reasons = segment.get('review_reasons')
        model_versions = segment.get('model_versions')
        is_translated = segment.get('is_translated') or False

        # Ensure proper primitive types
        def _json_or_none(x):
            if x is None:
                return None
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                return None

        sql = (
            "INSERT INTO segment (segment_id, audio_file_id, speaker_role, role_confidence, start_ms, end_ms, "
            "duration_ms, asr_confidence, lang, text_original, translations, products, intent, sentiment, "
            "emotion, emotions, needs_review, review_reasons, model_versions, is_translated) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )

        params = (
            str(seg_id) if seg_id is not None else None,
            int(audio_file_id) if audio_file_id is not None else None,
            speaker_role,
            float(role_confidence) if role_confidence is not None else None,
            int(start_ms) if start_ms is not None else 0,
            int(end_ms) if end_ms is not None else 0,
            int(duration_ms) if duration_ms is not None else None,
            float(asr_confidence) if asr_confidence is not None else None,
            (str(lang) if lang is not None else None),
            text_original if text_original is not None else '',
            _json_or_none(translations),
            _json_or_none(products),
            _json_or_none(intent),
            _json_or_none(sentiment),
            (str(emotion) if emotion is not None else None),
            _json_or_none(emotions),
            1 if bool(needs_review) else 0,
            _json_or_none(review_reasons),
            _json_or_none(model_versions),
            1 if bool(is_translated) else 0,
        )

        cur.execute(sql, params)
        conn.commit()
    finally:
        try:
            cur.close()
        except Exception:
            pass
        if own_conn:
            try:
                conn.close()
            except Exception:
                pass

def get_audio_file_id_by_filename(filename: str, conn: Optional["mysql.connector.MySQLConnection"] = None) -> Optional[int]:
    """
    Resolve audio_file.id by matching the basename at the end of s3_uri.

    Tries exact basename match and common extension variants (.opus/.mp3/.wav) against the tail of s3_uri.

    Returns the id or None if not found.
    """
    _ensure_connector_present_and_ok()
    import mysql.connector  # type: ignore

    if not filename:
        return None
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    # If filename was locally prefixed with a numeric DB id like "16_foo.mp3",
    # also generate candidates without that prefix to match original s3_uri basenames.
    stem_no_prefix = re.sub(r"^\d+_", "", stem)
    base_no_prefix = base
    if base.lower().startswith((stem + ".").lower()):
        base_no_prefix = f"{stem_no_prefix}{os.path.splitext(base)[1]}"

    # Include URL-encoded variants for spaces and other safe chars
    def _enc(s: str):
        try:
            from urllib.parse import quote
            return quote(s)
        except Exception:
            return s

    # Build a robust candidate set: with and without numeric prefix, common extensions, and encoded.
    candidates = {
        # With prefix variants
        base,
        f"{stem}.mp3",
        f"{stem}.wav",
        f"{stem}.opus",
        _enc(base),
        _enc(f"{stem}.mp3"),
        _enc(f"{stem}.wav"),
        _enc(f"{stem}.opus"),
        # Without prefix variants
        base_no_prefix,
        f"{stem_no_prefix}.mp3",
        f"{stem_no_prefix}.wav",
        f"{stem_no_prefix}.opus",
        _enc(base_no_prefix),
        _enc(f"{stem_no_prefix}.mp3"),
        _enc(f"{stem_no_prefix}.wav"),
        _enc(f"{stem_no_prefix}.opus"),
    }

    own_conn = False
    if conn is None:
        conn = get_connection()
        own_conn = True
    try:
        cur = conn.cursor()
        # Try each candidate; stop at first hit
        for cand in candidates:
            try:
                cur.execute(
                    "SELECT id FROM audio_file WHERE s3_uri LIKE %s ORDER BY id DESC LIMIT 1",
                    (f"%" + cand + "%",)
                )
                row = cur.fetchone()
                if row:
                    return int(row[0])
            except Exception:
                continue
        return None
    finally:
        try:
            cur.close()
        except Exception:
            pass
        if own_conn:
            try:
                conn.close()
            except Exception:
                pass

def lambda_handler(event, context):
    """
    Lambda handler to perform database operations based on the event.

    Expected event format:
    {
        "action": "get_connection" | "insert_segment" | "get_audio_file_id",
        "parameters": { ... }  # Parameters specific to the action
    }
    """
    action = event.get('action')
    params = event.get('parameters', {})

    if action == 'get_connection':
        try:
            conn = get_connection()
            return {
                'statusCode': 200,
                'body': 'Connection established successfully',
                'connection': str(conn)  # Note: In production, don't expose connection details
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'body': f'Failed to get connection: {str(e)}'
            }

    elif action == 'insert_segment':
        segment = params.get('segment')
        if not segment:
            return {
                'statusCode': 400,
                'body': 'Segment data is required for insert_segment'
            }
        try:
            insert_segment(segment)
            return {
                'statusCode': 200,
                'body': 'Segment inserted successfully'
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'body': f'Failed to insert segment: {str(e)}'
            }

    elif action == 'get_audio_file_id':
        filename = params.get('filename')
        if not filename:
            return {
                'statusCode': 400,
                'body': 'Filename is required for get_audio_file_id'
            }
        try:
            audio_id = get_audio_file_id_by_filename(filename)
            return {
                'statusCode': 200,
                'body': f'Audio file ID: {audio_id}',
                'audio_id': audio_id
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'body': f'Failed to get audio file ID: {str(e)}'
            }

    else:
        return {
            'statusCode': 400,
            'body': 'Invalid action. Supported actions: get_connection, insert_segment, get_audio_file_id'
        }
