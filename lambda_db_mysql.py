# lambda_db_mysql.py
import os
import re
import json
from typing import Optional, Mapping, Any

# Load .env using python-dotenv
from dotenv import find_dotenv, load_dotenv

_dotenv_path = find_dotenv()  # searches upwards for a .env file
if _dotenv_path:
    # load into os.environ without overriding existing env vars
    load_dotenv(_dotenv_path, override=False)

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
        "action": "get_connection" | "get_audio_file_id",
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
            'body': 'Invalid action. Supported actions: get_connection, get_audio_file_id'
        }
