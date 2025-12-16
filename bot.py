#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import os
import io
import time
import json
import base64
import datetime as dt
import logging
import atexit
import tempfile
import sqlite3
import re
import unicodedata
import csv
from collections import deque, OrderedDict, defaultdict
import shutil
from pathlib import Path
from pydub import AudioSegment
from docx import Document as DocxDocument
from pypdf import PdfReader

import requests
from dotenv import load_dotenv


load_dotenv()

# ========= Переменные окружения =========

TG_TOKEN   = os.getenv("BOT_TOKEN", "").strip()
YC_API_KEY = os.getenv("YC_API_KEY", "").strip()
TTS_URL    = os.getenv("TTS_URL", "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis").strip()
TG_API     = f"https://api.telegram.org/bot{TG_TOKEN}"

DONATE_URL = os.getenv("DONATE_URL", "").strip()
PAYMENT_PROVIDER_TOKEN = os.getenv("PAYMENT_PROVIDER_TOKEN", "").strip()  # для Telegram Stars (XTR)
DONATE_TITLE = os.getenv("DONATE_TITLE", "Поддержать разработку")
DONATE_DESCRIPTION = os.getenv("DONATE_DESCRIPTION", "Спасибо за поддержку проекта!")

DEFAULT_VOICE = os.getenv("VOICE", "filipp")
DEFAULT_SPEED = os.getenv("SPEED", "1.0")
MAX_LEN       = int(os.getenv("MAX_LEN", "2000"))

# Администраторы (ID через запятую/точку с запятой)
ADMIN_IDS = {int(x) for x in os.getenv("ADMIN_IDS", "").replace(";", ",").split(",") if x.strip().isdigit()}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tts-bot")
log.setLevel(logging.INFO)


if not TG_TOKEN:
    raise RuntimeError("Не указан BOT_TOKEN. Укажите его в .env (BOT_TOKEN=...).")
if not YC_API_KEY:
    raise RuntimeError("Не указан YC_API_KEY. Укажите его в .env (YC_API_KEY=...).")


# ============== Single-instance lock ==============
def _pid_is_running(pid: int) -> bool:
    try:
        if pid <= 0:
            return False
        if os.name == 'nt':
            import ctypes
            SYNCHRONIZE = 0x00100000
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False


def _acquire_single_instance_lock():
    if os.getenv("DISABLE_SINGLE_INSTANCE") == "1":
        return None
    token_part = (TG_TOKEN or "no_token")[:8]
    lock_path = os.path.join(tempfile.gettempdir(), f"tts_bot_{token_part}.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8", errors="ignore"))
        os.close(fd)
    except FileExistsError:
        try:
            with open(lock_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = (f.read() or '').strip()
            old_pid = int(content) if content.isdigit() else -1
        except Exception:
            old_pid = -1
        if old_pid != -1 and _pid_is_running(old_pid):
            print(f"Бот уже запущен (PID {old_pid}). Lock: {lock_path}")
            raise SystemExit(0)
        try:
            os.remove(lock_path)
        except Exception:
            pass
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8", errors="ignore"))
        os.close(fd)

    def _cleanup():
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
        except Exception:
            pass

    atexit.register(_cleanup)
    return lock_path


# ============== Голоса и роли ==============
VOICE_LABELS: "OrderedDict[str,str]" = OrderedDict([
    ("marina",    "Марина"),
    ("masha",     "Маша"),
    ("lera",      "Лера"),
    ("alena",     "Алёна"),
    ("jane",      "Джейн"),
    ("dasha",     "Даша"),
    ("julia",     "Юлия"),
    ("anton",     "Антон"),
    ("alexander", "Александр"),
    ("kirill",    "Кирилл"),
    ("filipp",    "Филипп"),
    ("ermil",     "Ермил"),
    ("zahar",     "Захар"),
    ("madi_ru",   "Мади"),
    ("madirus",   "Мади (alias)"),
    ("saule_ru",  "Сауле"),
    ("omazh",     "Омаж"),
    ("yulduz_ru", "Юлдуз"),
])

ROLE_LABELS = OrderedDict([
    ("neutral",  "Нейтрально"),
])

VOICE_ROLES = {
    "ermil":     ["neutral"],
    "zahar":     ["neutral"],
    "filipp":    ["neutral"],
    "alexander": ["neutral"],
    "kirill":    ["neutral"],
    "anton":     ["neutral"],
    "marina":    ["neutral"],
    "masha":     ["neutral"],
    "lera":      ["neutral"],
    "alena":     ["neutral"],
    "dasha":     ["neutral"],
    "julia":     ["neutral"],
    "jane":      ["neutral"],
    "madi_ru":   ["neutral"],
    "madirus":   ["neutral"],
    "saule_ru":  ["neutral"],
    "omazh":     ["neutral"],
    "yulduz_ru": ["neutral"],
}

ALLOWED_FORMATS = ["OGG_OPUS", "MP3", "WAV"]
CAPTION_FORMAT_LABELS = {
    "OGG_OPUS": "OGG/Opus",
    "MP3": "MP3",
    "WAV": "WAV",
}
CAPTION_SNIPPET_MAX = 80
FILENAME_SNIPPET_MAX = 40
TTS_CHUNK_LIMIT = int(os.getenv("TTS_CHUNK_LIMIT", "200"))
ALLOWED_DOC_MIME = {
    "text/plain": "txt",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "docx",
}
ALLOWED_DOC_EXT = {
    ".txt": "txt",
    ".pdf": "pdf",
    ".docx": "docx",
}
MAX_DOCUMENT_SIZE = int(os.getenv("MAX_DOCUMENT_SIZE", "5242880"))
INPUT_FIELD_PLACEHOLDER = os.getenv("INPUT_PLACEHOLDER", "Озвучу текст до 3000 символов")
SPEED_PRESETS = OrderedDict([
    ("0.9x", "0.9"),
    ("1.0x", "1.0"),
    ("1.3x", "1.3"),
])
VARIABLE_SPEED_VOICES = {
    "marina", "masha", "alena", "dasha", "julia", "anton", "alexander", "lera",
}
DEFAULT_SPEED = "1.0"
DEFAULT_PITCH = "normal"
EMOJI_RANGES = [
    (0x1F300, 0x1F6FF),
    (0x1F900, 0x1F9FF),
    (0x1FA70, 0x1FAFF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6C5),
    (0x1F700, 0x1F77F),
    (0x1F780, 0x1F7FF),
    (0x1F800, 0x1F8FF),
]


# ============== Профили и лимиты ==============
# Stats
usage_total = 0
usage_per_user = defaultdict(int)
tts_ok = 0
tts_err = 0
usage_by_day = defaultdict(int)  # YYYY-MM-DD -> count
usage_by_user_day = defaultdict(lambda: defaultdict(int))
tts_ok_by_day = defaultdict(int)
tts_err_by_day = defaultdict(int)

# Filesystem for stats
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_last_stats_day: str | None = None

DB_PATH = os.path.join(BASE_DIR, "bot.db")
_db_conn: sqlite3.Connection | None = None
user_prefs_cache: "LruCache" | None = None
_last_rate_cleanup = 0
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
STATS_CSV_PATH = os.path.join(BASE_DIR, "stats.csv")
TELEGRAM_ERROR_LOG = os.path.join(BASE_DIR, "telegram_errors.log")
REPLY_BUTTONS = OrderedDict([
    ("🎙️ Голос", "/voice"),
    ("⚡ Скорость", "/speed"),
    ("🎧 Формат", "/format"),
    ("ℹ️ Справка", "/help"),
    ("💫 Донат", "/donate"),
])
REPLY_BUTTON_COMMANDS = {label.lower(): command for label, command in REPLY_BUTTONS.items()}
metrics_state = {
    "speechkit_errors": 0,
    "text_len_sum": 0,
    "text_len_count": 0,
    "audio_bytes_sum": 0,
    "audio_count": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "known_users": set(),
    "prefs_changed_users": set(),
    "telegram_errors": defaultdict(int),
    "text_len_per_user": defaultdict(lambda: {"sum": 0, "count": 0}),
    "audio_bytes_per_user": defaultdict(lambda: {"sum": 0, "count": 0}),
}


def _load_metrics_state():
    if not os.path.exists(METRICS_PATH):
        return
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics_state["speechkit_errors"] = data.get("speechkit_errors", 0)
        metrics_state["text_len_sum"] = data.get("text_len_sum", 0)
        metrics_state["text_len_count"] = data.get("text_len_count", 0)
        metrics_state["audio_bytes_sum"] = data.get("audio_bytes_sum", 0)
        metrics_state["audio_count"] = data.get("audio_count", 0)
        metrics_state["cache_hits"] = data.get("cache_hits", 0)
        metrics_state["cache_misses"] = data.get("cache_misses", 0)
        metrics_state["known_users"] = set(data.get("known_users", []))
        metrics_state["prefs_changed_users"] = set(data.get("prefs_changed_users", []))
        metrics_state["telegram_errors"] = defaultdict(int, data.get("telegram_errors", {}))
        text_users = {
            int(k): {"sum": v.get("sum", 0), "count": v.get("count", 0)}
            for k, v in data.get("text_len_per_user", {}).items()
        }
        audio_users = {
            int(k): {"sum": v.get("sum", 0), "count": v.get("count", 0)}
            for k, v in data.get("audio_bytes_per_user", {}).items()
        }
        metrics_state["text_len_per_user"] = defaultdict(lambda: {"sum": 0, "count": 0}, text_users)
        metrics_state["audio_bytes_per_user"] = defaultdict(lambda: {"sum": 0, "count": 0}, audio_users)
    except Exception:
        log.exception("metrics load error")


def _save_metrics_state():
    try:
        known_users = sorted(metrics_state["known_users"])
        prefs_changed = sorted(metrics_state["prefs_changed_users"])
        total_texts = max(1, metrics_state["text_len_count"])
        total_audios = max(1, metrics_state["audio_count"])
        cache_total = max(1, metrics_state["cache_hits"] + metrics_state["cache_misses"])
        total_users = max(1, len(metrics_state["known_users"]))
        data = {
            "speechkit_errors": metrics_state["speechkit_errors"],
            "text_len_sum": metrics_state["text_len_sum"],
            "text_len_count": metrics_state["text_len_count"],
            "audio_bytes_sum": metrics_state["audio_bytes_sum"],
            "audio_count": metrics_state["audio_count"],
            "cache_hits": metrics_state["cache_hits"],
            "cache_misses": metrics_state["cache_misses"],
            "known_users": known_users,
            "prefs_changed_users": prefs_changed,
            "telegram_errors": dict(metrics_state["telegram_errors"]),
            "text_len_per_user": {str(k): v for k, v in metrics_state["text_len_per_user"].items()},
            "audio_bytes_per_user": {str(k): v for k, v in metrics_state["audio_bytes_per_user"].items()},
            "avg_text_length": metrics_state["text_len_sum"] / total_texts if metrics_state["text_len_count"] else 0,
            "avg_audio_size": metrics_state["audio_bytes_sum"] / total_audios if metrics_state["audio_count"] else 0,
            "cache_hit_ratio": metrics_state["cache_hits"] / cache_total if cache_total else 0,
            "prefs_change_percent": (len(metrics_state["prefs_changed_users"]) / total_users * 100.0) if total_users else 0,
        }
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        log.exception("metrics save error")


def _metrics_summary_lines() -> list[str]:
    total_texts = metrics_state["text_len_count"]
    total_audios = metrics_state["audio_count"]
    cache_hits = metrics_state["cache_hits"]
    cache_misses = metrics_state["cache_misses"]
    cache_total = cache_hits + cache_misses
    total_users = len(metrics_state["known_users"])
    changed = len(metrics_state["prefs_changed_users"])
    avg_text = metrics_state["text_len_sum"] / total_texts if total_texts else 0
    avg_audio = metrics_state["audio_bytes_sum"] / total_audios if total_audios else 0
    cache_ratio = cache_hits / cache_total if cache_total else 0
    prefs_percent = (changed / total_users * 100.0) if total_users else 0
    lines = [
        f"SpeechKit ошибок: {metrics_state['speechkit_errors']}",
        f"Средняя длина текста: {avg_text:.1f} символов",
        f"Средний размер аудио: {avg_audio/1024:.1f} КБ",
        f"Cache hit ratio: {cache_ratio*100:.1f}%",
        f"Пользователи с кастомными настройками: {prefs_percent:.1f}%"
    ]
    text_user_lines = _metrics_top_users(metrics_state["text_len_per_user"], "симв")
    audio_user_lines = _metrics_top_users(metrics_state["audio_bytes_per_user"], "КБ", divisor=1024)
    if text_user_lines:
        lines.append(f"Текст по пользователям: {text_user_lines}")
    if audio_user_lines:
        lines.append(f"Аудио по пользователям: {audio_user_lines}")
    if metrics_state["telegram_errors"]:
        top_errors = sorted(metrics_state["telegram_errors"].items(), key=lambda x: x[1], reverse=True)
        err_lines = ", ".join([f"{name}: {cnt}" for name, cnt in top_errors[:5]])
        lines.append(f"Ошибки Telegram: {err_lines}")
    return lines


def _metrics_top_users(data_map: dict, unit: str, divisor: int = 1, limit: int = 3) -> str:
    if not data_map:
        return ""
    top = sorted(data_map.items(), key=lambda x: x[1]["sum"], reverse=True)[:limit]
    parts = []
    for uid, vals in top:
        total = vals["sum"]
        count = max(1, vals["count"])
        avg = total / count
        if divisor != 1:
            total_display = total / divisor
            avg_display = avg / divisor
        else:
            total_display = total
            avg_display = avg
        parts.append(f"{uid}: ср.{avg_display:.1f} {unit}, сум.{total_display:.1f} {unit}")
    return "; ".join(parts)


def _export_user_stats_csv(path: str = STATS_CSV_PATH) -> str:
    try:
        user_ids = sorted(metrics_state["known_users"] | set(usage_per_user.keys()))
        headers = [
            "chat_id",
            "total_messages",
            "text_sum",
            "text_avg",
            "audio_sum_bytes",
            "audio_avg_bytes",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for uid in user_ids:
                total_msgs = usage_per_user.get(uid, 0)
                text_entry = metrics_state["text_len_per_user"].get(uid, {"sum": 0, "count": 0})
                audio_entry = metrics_state["audio_bytes_per_user"].get(uid, {"sum": 0, "count": 0})
                text_sum = text_entry["sum"]
                text_count = max(1, text_entry["count"])
                audio_sum = audio_entry["sum"]
                audio_count = max(1, audio_entry["count"])
                writer.writerow([
                    uid,
                    total_msgs,
                    text_sum,
                    text_sum / text_count if text_entry["count"] else 0,
                    audio_sum,
                    audio_sum / audio_count if audio_entry["count"] else 0,
                ])
        return path
    except Exception:
        log.exception("export stats csv error")
        raise


def _prefs_is_non_default(prefs: dict) -> bool:
    return (
        prefs.get("voice") != DEFAULT_VOICE
        or prefs.get("speed") != DEFAULT_SPEED
        or prefs.get("format") != "OGG_OPUS"
        or prefs.get("role") != "neutral"
        or prefs.get("pitch") != DEFAULT_PITCH
    )


def _metrics_register_user(chat_id: int, prefs: dict):
    metrics_state["known_users"].add(chat_id)
    if _prefs_is_non_default(prefs):
        metrics_state["prefs_changed_users"].add(chat_id)
    else:
        metrics_state["prefs_changed_users"].discard(chat_id)


def _normalize_voice_settings(prefs: dict) -> None:
    voice = prefs.get("voice", DEFAULT_VOICE)
    speed_options = set(_voice_speed_options(voice).values())
    if prefs.get("speed") not in speed_options:
        prefs["speed"] = DEFAULT_SPEED if DEFAULT_SPEED in speed_options else next(iter(speed_options))
    pitch_options = set(_voice_pitch_options(voice).values())
    if prefs.get("pitch") not in pitch_options:
        prefs["pitch"] = DEFAULT_PITCH if DEFAULT_PITCH in pitch_options else next(iter(pitch_options))


def _metrics_track_text(chat_id: int, length: int):
    metrics_state["text_len_sum"] += max(0, length)
    metrics_state["text_len_count"] += 1
    entry = metrics_state["text_len_per_user"][chat_id]
    entry["sum"] += max(0, length)
    entry["count"] += 1


def _metrics_track_cache_hit():
    metrics_state["cache_hits"] += 1


def _metrics_track_cache_miss():
    metrics_state["cache_misses"] += 1


def _metrics_track_speechkit_error():
    metrics_state["speechkit_errors"] += 1


def _metrics_track_audio_size(chat_id: int, size: int):
    metrics_state["audio_bytes_sum"] += max(0, size)
    metrics_state["audio_count"] += 1
    entry = metrics_state["audio_bytes_per_user"][chat_id]
    entry["sum"] += max(0, size)
    entry["count"] += 1


def _write_telegram_error_log(message: str):
    try:
        ts = dt.datetime.now().isoformat()
        with open(TELEGRAM_ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"{ts} {message}\n")
    except Exception:
        log.exception("telegram error log write failed")


def _metrics_track_telegram_error(exc: Exception | None = None, context: str | None = None, details: str | None = None):
    label = exc.__class__.__name__
    if isinstance(exc, requests.HTTPError) and getattr(exc, "response", None) is not None:
        resp = exc.response
        try:
            payload = resp.json()
            label = payload.get("description") or payload.get("error_code") or f"HTTP {resp.status_code}"
        except Exception:
            label = f"HTTP {resp.status_code}: {resp.text[:100]}"
    elif isinstance(exc, requests.RequestException):
        label = exc.__class__.__name__
    metrics_state["telegram_errors"][label] += 1
    info = label
    if context:
        info = f"{context}: {label}"
    if details:
        info = f"{info} | {details}"
    _write_telegram_error_log(info)


def _get_db_connection() -> sqlite3.Connection:
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _db_conn.row_factory = sqlite3.Row
    return _db_conn


def _close_db_connection():
    global _db_conn
    if _db_conn is not None:
        try:
            _db_conn.close()
        except Exception:
            pass
        _db_conn = None


atexit.register(_close_db_connection)


def _init_db():
    conn = _get_db_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                chat_id INTEGER PRIMARY KEY,
                voice TEXT NOT NULL,
                speed TEXT NOT NULL,
                format TEXT NOT NULL,
                role TEXT NOT NULL,
                pitch TEXT NOT NULL DEFAULT 'normal'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_daily (
                day TEXT PRIMARY KEY,
                usage_total INTEGER NOT NULL DEFAULT 0,
                tts_ok INTEGER NOT NULL DEFAULT 0,
                tts_err INTEGER NOT NULL DEFAULT 0,
                generated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_per_user_daily (
                day TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                cnt INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(day, chat_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rate_limits (
                chat_id INTEGER PRIMARY KEY,
                timestamps TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )
        """)


def _ensure_user_pref_schema():
    conn = _get_db_connection()
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(user_preferences)")}
    if "pitch" not in columns:
        with conn:
            conn.execute("ALTER TABLE user_preferences ADD COLUMN pitch TEXT NOT NULL DEFAULT 'normal'")


def _persist_user_pref(chat_id: int, prefs: dict) -> None:
    payload = (
        chat_id,
        prefs.get("voice", DEFAULT_VOICE),
        prefs.get("speed", DEFAULT_SPEED),
        prefs.get("format", "OGG_OPUS"),
        prefs.get("role", "neutral"),
        prefs.get("pitch", DEFAULT_PITCH),
    )
    conn = _get_db_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO user_preferences (chat_id, voice, speed, format, role, pitch)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                voice=excluded.voice,
                speed=excluded.speed,
                format=excluded.format,
                role=excluded.role,
                pitch=excluded.pitch
            """,
            payload,
        )
    if user_prefs_cache is not None:
        user_prefs_cache.set(chat_id, {
            "voice": payload[1],
            "speed": payload[2],
            "format": payload[3],
            "role": payload[4],
            "pitch": payload[5],
        })


def _load_user_pref_from_db(chat_id: int) -> dict | None:
    conn = _get_db_connection()
    row = conn.execute(
        "SELECT voice, speed, format, role, pitch FROM user_preferences WHERE chat_id = ?",
        (chat_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "voice": row["voice"] or DEFAULT_VOICE,
        "speed": row["speed"] or DEFAULT_SPEED,
        "format": row["format"] or "OGG_OPUS",
        "role": row["role"] or "neutral",
        "pitch": row["pitch"] or DEFAULT_PITCH,
    }


def _persist_stats_day(day: str, snapshot: dict) -> None:
    conn = _get_db_connection()
    generated_at = snapshot.get("generated_at")
    if not generated_at:
        generated_at = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    per_user = snapshot.get("per_user") or []
    with conn:
        conn.execute(
            """
            INSERT INTO usage_daily (day, usage_total, tts_ok, tts_err, generated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(day) DO UPDATE SET
                usage_total=excluded.usage_total,
                tts_ok=excluded.tts_ok,
                tts_err=excluded.tts_err,
                generated_at=excluded.generated_at
            """,
            (
                day,
                int(snapshot.get("usage_total", 0) or 0),
                int(snapshot.get("tts_ok", 0) or 0),
                int(snapshot.get("tts_err", 0) or 0),
                generated_at,
            ),
        )
        conn.execute("DELETE FROM usage_per_user_daily WHERE day = ?", (day,))
        if per_user:
            conn.executemany(
                "INSERT INTO usage_per_user_daily (day, chat_id, cnt) VALUES (?, ?, ?)",
                [(day, int(uid), int(cnt)) for uid, cnt in per_user],
            )


def _import_existing_stats_files():
    try:
        conn = _get_db_connection()
        existing_days = {row["day"] for row in conn.execute("SELECT day FROM usage_daily")}
        for name in os.listdir(BASE_DIR):
            if not (name.startswith("stats-") and name.endswith(".json")):
                continue
            day = name[6:-5]
            if day in existing_days:
                continue
            path = os.path.join(BASE_DIR, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                log.warning("Не удалось импортировать статистику из %s", path)
                continue
            per_user_rows = []
            for item in data.get("per_user") or []:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    uid, cnt = item
                    try:
                        uid = int(uid)
                        cnt = int(cnt)
                    except Exception:
                        continue
                    per_user_rows.append((uid, cnt))
            snapshot = {
                "day": day,
                "usage_total": data.get("usage_total", 0),
                "tts_ok": data.get("tts_ok", 0),
                "tts_err": data.get("tts_err", 0),
                "per_user": per_user_rows,
                "generated_at": data.get("generated_at"),
            }
            _persist_stats_day(day, snapshot)
    except Exception:
        log.exception("stats import error")


def _load_rate_timestamps(chat_id: int) -> deque:
    conn = _get_db_connection()
    row = conn.execute("SELECT timestamps FROM rate_limits WHERE chat_id = ?", (chat_id,)).fetchone()
    dq = deque()
    if not row:
        return dq
    try:
        data = json.loads(row["timestamps"])
    except Exception:
        return dq
    for value in data:
        try:
            dq.append(float(value))
        except Exception:
            continue
    return dq


def _persist_rate_timestamps(chat_id: int, dq: deque) -> None:
    payload = json.dumps(list(dq))
    now_int = int(time.time())
    conn = _get_db_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO rate_limits (chat_id, timestamps, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                timestamps=excluded.timestamps,
                updated_at=excluded.updated_at
            """,
            (chat_id, payload, now_int),
        )


RATE_LIMIT_RETENTION = 3600


def _cleanup_rate_limits(retention_seconds: int = RATE_LIMIT_RETENTION) -> None:
    global _last_rate_cleanup
    now_int = int(time.time())
    if now_int - _last_rate_cleanup < retention_seconds:
        return
    cutoff = now_int - retention_seconds
    conn = _get_db_connection()
    with conn:
        conn.execute("DELETE FROM rate_limits WHERE updated_at < ?", (cutoff,))
    _last_rate_cleanup = now_int


def _load_state_from_db():
    global usage_total, tts_ok, tts_err, _last_stats_day
    try:
        conn = _get_db_connection()
        for row in conn.execute("SELECT day, usage_total, tts_ok, tts_err FROM usage_daily"):
            day = row["day"]
            usage_by_day[day] = row["usage_total"]
            tts_ok_by_day[day] = row["tts_ok"]
            tts_err_by_day[day] = row["tts_err"]
        for row in conn.execute("SELECT day, chat_id, cnt FROM usage_per_user_daily"):
            usage_by_user_day[row["day"]][row["chat_id"]] = row["cnt"]
            usage_per_user[row["chat_id"]] += row["cnt"]
        if usage_by_day:
            _last_stats_day = max(usage_by_day.keys())
        usage_total = sum(usage_by_day.values())
        tts_ok = sum(tts_ok_by_day.values())
        tts_err = sum(tts_err_by_day.values())
    except Exception:
        log.exception("load state from db error")


_init_db()
_ensure_user_pref_schema()
_import_existing_stats_files()
_load_state_from_db()
_load_metrics_state()


def _configure_audio_backend():
    ffmpeg_env = os.getenv("FFMPEG_BIN", "").strip()
    candidates = []
    if ffmpeg_env:
        candidates.append(Path(ffmpeg_env))
    search_root = Path(BASE_DIR) / "ffmpeg_extracted"
    if search_root.exists():
        for child in search_root.iterdir():
            bin_dir = child / "bin"
            ffmpeg_file = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            if ffmpeg_file.exists():
                candidates.append(ffmpeg_file)
    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        candidates.append(Path(which_ffmpeg))
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            bin_dir = resolved.parent
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            AudioSegment.converter = str(resolved)
            AudioSegment.ffmpeg = str(resolved)
            probe_name = resolved.name.replace("ffmpeg", "ffprobe")
            probe = resolved.with_name(probe_name)
            if probe.exists():
                AudioSegment.ffprobe = str(probe.resolve())
            else:
                which_probe = shutil.which("ffprobe")
                if which_probe:
                    AudioSegment.ffprobe = which_probe
            log.info("FFmpeg configured: %s", resolved)
            return
    log.warning("FFmpeg binary not found. Пожалуйста, установите ffmpeg и задайте путь через FFMPEG_BIN.")


_configure_audio_backend()


def _stats_day_snapshot(day: str) -> dict:
    per_user = usage_by_user_day.get(day, {})
    top = sorted(per_user.items(), key=lambda x: x[1], reverse=True)
    return {
        "day": day,
        "usage_total": usage_by_day.get(day, 0),
        "tts_ok": tts_ok_by_day.get(day, 0),
        "tts_err": tts_err_by_day.get(day, 0),
        "per_user": top,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }

def _save_stats_for_day(day: str) -> None:
    try:
        data = _stats_day_snapshot(day)
        _persist_stats_day(day, data)
        path = os.path.join(BASE_DIR, f"stats-{day}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        _save_metrics_state()
    except Exception:
        # Не мешаем работе бота из-за ошибок записи статистики
        log.exception("save stats error")


class LruCache:
    def __init__(self, max_items=200):
        self.max = max_items
        self.d = OrderedDict()
    def get(self, key):
        if key not in self.d:
            return None
        val = self.d.pop(key)
        self.d[key] = val
        return val
    def set(self, key, val):
        if key in self.d:
            self.d.pop(key)
        elif len(self.d) >= self.max:
            self.d.popitem(last=False)
        self.d[key] = val


cache = LruCache(200)
if user_prefs_cache is None:
    user_prefs_cache = LruCache(1000)

RATE_LIMIT_MAX = 30
RATE_LIMIT_WINDOW = 60

def allow_request(chat_id: int) -> bool:
    try:
        dq = _load_rate_timestamps(chat_id)
    except Exception:
        dq = deque()
    now = time.time()
    while dq and now - dq[0] > RATE_LIMIT_WINDOW:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_MAX:
        return False
    dq.append(now)
    try:
        _persist_rate_timestamps(chat_id, dq)
        _cleanup_rate_limits()
    except Exception:
        log.exception("rate limit persist error")
    return True


# ============== SpeechKit v3 (NDJSON) ==============
def _collect_audio_chunks_from_obj(obj):
    chunks = []
    if isinstance(obj, dict):
        ac = obj.get("audioChunk")
        if isinstance(ac, dict) and "data" in ac:
            chunks.append(ac["data"])
        for v in obj.values():
            chunks.extend(_collect_audio_chunks_from_obj(v))
    elif isinstance(obj, list):
        for v in obj:
            chunks.extend(_collect_audio_chunks_from_obj(v))
    return chunks


def _build_tts_hints(voice: str, speed: str, role: str | None, pitch: str | None) -> list[dict]:
    hints = [{"voice": voice}]
    if voice in VARIABLE_SPEED_VOICES and speed:
        try:
            speed_value = float(speed)
        except (TypeError, ValueError):
            speed_value = float(DEFAULT_SPEED)
        hints.append({"speed": speed_value})
    if role and role != "neutral":
        hints.append({"role": role})
    return hints


def synth_tts(text: str, voice: str, speed: str, out_format: str, role: str | None = None, pitch: str | None = None) -> bytes:
    if out_format not in ALLOWED_FORMATS:
        raise ValueError(f"Неподдерживаемый формат {out_format}. Допустимо: {', '.join(ALLOWED_FORMATS)}")
    if voice == "madirus":
        voice = "madi_ru"

    base_hints = _build_tts_hints(voice, speed, role, pitch)
    hint_variants = []
    seen = set()

    def add_variant(hints_list):
        key = json.dumps(hints_list, sort_keys=True)
        if key not in seen:
            seen.add(key)
            hint_variants.append(hints_list)

    add_variant(base_hints)
    add_variant([h for h in base_hints if "pitch" not in h])
    add_variant([h for h in base_hints if "pitch" not in h and "speed" not in h])
    add_variant([{"voice": voice}])

    last_error = None
    for hints in hint_variants:
        payload = {
            "text": text,
            "hints": hints,
            "outputAudioSpec": {"containerAudio": {"containerAudioType": out_format}},
        }
        try:
            with requests.post(
                TTS_URL,
                headers={
                    "Authorization": f"Api-Key {YC_API_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "application/x-ndjson",
                },
                data=json.dumps(payload),
                timeout=120,
                stream=True,
            ) as r:
                if r.status_code != 200:
                    try:
                        log.error("TTS ERROR %s: %s", r.status_code, r.json())
                    except Exception:
                        log.error("TTS ERROR %s: %s", r.status_code, r.text[:500])
                    r.raise_for_status()

                b64_parts = []
                for raw in r.iter_lines(decode_unicode=False):
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw.decode("utf-8", errors="ignore"))
                    except Exception:
                        continue
                    b64_parts.extend(_collect_audio_chunks_from_obj(obj))

                if not b64_parts:
                    raise RuntimeError("SpeechKit v3: пустой ответ, нет audioChunk.")

                return base64.b64decode("".join(b64_parts))
        except requests.HTTPError as e:
            err_text = ""
            try:
                err_text = e.response.text
            except Exception:
                pass
            if e.response is not None and e.response.status_code == 400 and "Hint" in err_text:
                log.warning("SpeechKit hint error, retrying with reduced hints: %s", err_text[:200])
                last_error = e
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("SpeechKit: all hint variants failed")


# ============== Caption builder ==============
def _normalize_snippet_sample(sample: str) -> str:
    return " ".join(sample.split()).strip()


def _extract_snippet(sample: str, limit: int) -> tuple[str, bool]:
    cleaned = _normalize_snippet_sample(sample)
    if not cleaned:
        return "", False
    truncated = len(cleaned) > limit
    snippet = cleaned[:limit].rstrip()
    return snippet, truncated


def _build_caption(voice: str, out_format: str, sample: str) -> str:
    voice_ru = VOICE_LABELS.get(voice, voice)
    fmt_label = CAPTION_FORMAT_LABELS.get(out_format, out_format)
    snippet, truncated = _extract_snippet(sample, CAPTION_SNIPPET_MAX)
    if snippet:
        if truncated:
            snippet = snippet.rstrip(".,:;!?- ") + "…"
        snippet_part = f" — «{snippet}»"
    else:
        snippet_part = ""
    return f"TTS v3 — {voice_ru} — {fmt_label}{snippet_part}"


def _build_audio_filename(sample: str, out_format: str) -> str:
    ext_map = {
        "OGG_OPUS": "ogg",
        "MP3": "mp3",
        "WAV": "wav",
    }
    snippet, _ = _extract_snippet(sample, FILENAME_SNIPPET_MAX)
    sanitized = re.sub(r"[^0-9A-Za-zА-Яа-яёЁ _-]", "", snippet)
    sanitized = re.sub(r"\s+", "_", sanitized.strip())
    if not sanitized:
        sanitized = "speech"
    ext = ext_map.get(out_format, "bin")
    return f"{sanitized}.{ext}"


def _normalize_text_for_tts(text: str) -> str:
    def _is_emoji(ch: str) -> bool:
        code = ord(ch)
        for start, end in EMOJI_RANGES:
            if start <= code <= end:
                return True
        return False

    cleaned_chars = []
    for ch in text:
        if _is_emoji(ch):
            continue
        if ch in ("\u200b", "\u200c", "\u200d"):
            continue
        if ch == "\r":
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in ("\n", "\t"):
            continue
        if ord(ch) >= 0x110000:
            continue
        cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars)
    cleaned = re.sub(r"https?://\S+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _split_text_by_tokens(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    tokens = re.findall(r"\S+\s*", text)
    parts = []
    buf = []
    buf_len = 0
    for token in tokens:
        token_len = len(token)
        if buf_len + token_len <= limit:
            buf.append(token)
            buf_len += token_len
            continue
        if buf:
            parts.append("".join(buf).strip())
            buf.clear()
            buf_len = 0
        if token_len > limit:
            start = 0
            while start < token_len:
                chunk = token[start:start + limit]
                if len(chunk) == limit:
                    parts.append(chunk.strip())
                else:
                    buf = [chunk]
                    buf_len = len(chunk)
                start += limit
        else:
            buf = [token]
            buf_len = token_len
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _split_text_by_sentences(text: str) -> list[str]:
    pattern = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.S)
    sentences = []
    for match in pattern.finditer(text):
        seg = match.group().strip()
        if seg:
            sentences.append(seg)
    return sentences


def _split_text_for_tts(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    sentences = _split_text_by_sentences(text)
    if not sentences:
        return _split_text_by_tokens(text, limit)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > limit:
            if current:
                parts.append(current.strip())
                current = ""
            parts.extend(_split_text_by_tokens(sentence, limit))
            continue
        if not current:
            current = sentence
            continue
        candidate = f"{current} {sentence}"
        if len(candidate) <= limit:
            current = candidate
        else:
            parts.append(current.strip())
            current = sentence
    if current:
        parts.append(current.strip())
    normalized: list[str] = []
    for part in parts:
        if len(part) <= limit:
            normalized.append(part)
        else:
            normalized.extend(_split_text_by_tokens(part, limit))
    return [p for p in normalized if p]


def _trim_chunk_to_limit(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    cut = truncated.rfind(" ")
    if cut >= int(limit * 0.5):
        truncated = truncated[:cut]
    truncated = truncated.strip()
    if not truncated:
        truncated = text[:limit].strip()
    return truncated[:limit]


def _detect_document_type(document: dict) -> str | None:
    mime = (document.get("mime_type") or "").lower()
    if mime in ALLOWED_DOC_MIME:
        return ALLOWED_DOC_MIME[mime]
    name = (document.get("file_name") or "").lower()
    _, ext = os.path.splitext(name)
    return ALLOWED_DOC_EXT.get(ext)


def _download_file_bytes(file_id: str) -> bytes:
    meta = requests.get(f"{TG_API}/getFile", params={"file_id": file_id}, timeout=30)
    meta.raise_for_status()
    file_path = meta.json().get("result", {}).get("file_path")
    if not file_path:
        raise RuntimeError("file_path not found")
    url = f"https://api.telegram.org/file/bot{TG_TOKEN}/{file_path}"
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    return resp.content


def _extract_text_from_document(data: bytes, doc_type: str) -> str:
    if doc_type == "txt":
        for enc in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="ignore")
    if doc_type == "docx":
        doc = DocxDocument(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs)
    if doc_type == "pdf":
        reader = PdfReader(io.BytesIO(data))
        texts = []
        for page in reader.pages:
            try:
                extracted = page.extract_text() or ""
            except Exception:
                extracted = ""
            texts.append(extracted)
        return "\n".join(texts)
    raise ValueError(f"Unsupported document type: {doc_type}")


def _handle_document_message(chat_id: int, document: dict) -> str | None:
    doc_type = _detect_document_type(document)
    if not doc_type:
        tg_send_text(chat_id, "Формат файла не поддерживается. Пришлите TXT, DOCX или PDF.")
        return None
    size = int(document.get("file_size") or 0)
    if size > MAX_DOCUMENT_SIZE:
        tg_send_text(chat_id, f"Файл слишком большой (> {MAX_DOCUMENT_SIZE // 1024} КБ).")
        return None
    file_id = document.get("file_id")
    if not file_id:
        tg_send_text(chat_id, "Не удалось получить файл. Попробуйте снова.")
        return None
    try:
        raw = _download_file_bytes(file_id)
        text = _extract_text_from_document(raw, doc_type)
        if not text or not text.strip():
            tg_send_text(chat_id, "Не удалось извлечь текст из файла.")
            return None
        name = document.get("file_name") or "документ"
        tg_send_text(chat_id, f"Из файла {name} извлечено {len(text)} символов.")
        return text
    except Exception as exc:
        log.exception("document parse error: %r", exc)
        tg_send_text(chat_id, "Ошибка обработки документа. Убедитесь, что он не защищён.")
        return None


def _send_audio_blob(chat_id: int, audio: bytes, out_format: str, caption: str, sample: str):
    filename = _build_audio_filename(sample, out_format)
    tg_send_chat_action(chat_id, "upload_voice")
    time.sleep(1.0)
    try:
        if out_format == "OGG_OPUS":
            tg_send_voice(chat_id, audio, caption=caption, filename=filename)
        elif out_format == "MP3":
            tg_send_audio(chat_id, audio, filename, "audio/mpeg", caption=caption)
        elif out_format == "WAV":
            tg_send_audio(chat_id, audio, filename, "audio/wav", caption=caption)
        else:
            tg_send_audio(chat_id, audio, filename, "application/octet-stream", caption=caption)
    except Exception as exc:
        _metrics_track_telegram_error(exc)
        raise
    _metrics_track_audio_size(chat_id, len(audio))


def _combine_audio_chunks(audios: list[bytes], out_format: str) -> bytes:
    if not audios:
        raise ValueError("Нет аудиоданных для склейки")
    fmt_map = {
        "OGG_OPUS": "ogg",
        "MP3": "mp3",
        "WAV": "wav",
    }
    src_format = fmt_map.get(out_format, "ogg")
    combined: AudioSegment | None = None
    for data in audios:
        segment = AudioSegment.from_file(io.BytesIO(data), format=src_format)
        combined = segment if combined is None else combined + segment
    buffer = io.BytesIO()
    export_format = src_format
    export_kwargs = {}
    if out_format == "OGG_OPUS":
        export_kwargs = {"codec": "libopus", "parameters": ["-acodec", "libopus"]}
    combined.export(buffer, format=export_format, **export_kwargs)
    return buffer.getvalue()


# ============== Telegram helpers ==============
def tg_send_text(chat_id: int, text: str, reply_markup: dict | None = None):
    url = f"{TG_API}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    markup = reply_markup or kb_reply_main()
    if markup:
        data["reply_markup"] = json.dumps(markup, ensure_ascii=False)
    if INPUT_FIELD_PLACEHOLDER:
        data["input_field_placeholder"] = INPUT_FIELD_PLACEHOLDER
    r = requests.post(url, data=data, timeout=30)
    if r.status_code != 200:
        _metrics_track_telegram_error(context="sendMessage", details=r.text[:200])
        r.raise_for_status()
    try:
        return r.json().get("result", {}).get("message_id")
    except Exception:
        return None
    if r.status_code != 200:
        log.warning("TG sendMessage ERROR %s: %s", r.status_code, r.text[:300])


def tg_send_voice(chat_id: int, ogg_bytes: bytes, caption: str | None = None, filename: str | None = None):
    url = f"{TG_API}/sendVoice"
    name = filename or "speech.ogg"
    files = {"voice": (name, io.BytesIO(ogg_bytes), "audio/ogg")}
    data  = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption
    r = requests.post(url, data=data, files=files, timeout=90)
    if r.status_code != 200:
        log.error("TG sendVoice ERROR %s: %s", r.status_code, r.text[:300])
        _metrics_track_telegram_error(context="sendVoice", details=r.text[:200])
        r.raise_for_status()


def tg_send_audio(chat_id: int, bytes_data: bytes, filename: str, mime: str, caption: str | None = None):
    url = f"{TG_API}/sendAudio"
    files = {"audio": (filename, io.BytesIO(bytes_data), mime)}
    data  = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption
    r = requests.post(url, data=data, files=files, timeout=90)
    if r.status_code != 200:
        log.error("TG sendAudio ERROR %s: %s", r.status_code, r.text[:300])
        _metrics_track_telegram_error(context="sendAudio", details=r.text[:200])
        r.raise_for_status()


def tg_send_document(chat_id: int, file_path: str, caption: str | None = None):
    url = f"{TG_API}/sendDocument"
    with open(file_path, "rb") as f:
        files = {"document": (os.path.basename(file_path), f, "text/csv")}
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
        r = requests.post(url, data=data, files=files, timeout=90)
    if r.status_code != 200:
        log.error("TG sendDocument ERROR %s: %s", r.status_code, r.text[:300])
        _metrics_track_telegram_error(context="sendDocument", details=r.text[:200])
        r.raise_for_status()


def tg_delete_message(chat_id: int, message_id: int):
    url = f"{TG_API}/deleteMessage"
    data = {"chat_id": chat_id, "message_id": message_id}
    r = requests.post(url, data=data, timeout=15)
    if r.status_code != 200:
        _metrics_track_telegram_error(context="deleteMessage", details=r.text[:200])


def tg_answer_pre_checkout(query_id: str, ok: bool = True, error_message: str | None = None):
    url = f"{TG_API}/answerPreCheckoutQuery"
    data = {"pre_checkout_query_id": query_id, "ok": "true" if ok else "false"}
    if not ok and error_message:
        data["error_message"] = error_message
    r = requests.post(url, data=data, timeout=15)
    if r.status_code != 200:
        _metrics_track_telegram_error(context="answerPreCheckoutQuery", details=r.text[:200])
        r.raise_for_status()


def tg_answer_callback(cb_id: str, text: str | None = None, show_alert: bool = False):
    url = f"{TG_API}/answerCallbackQuery"
    data = {"callback_query_id": cb_id}
    if text:
        data["text"] = text
        data["show_alert"] = "true" if show_alert else "false"
    requests.post(url, data=data, timeout=15)


def tg_edit_message_text(chat_id: int, message_id: int, text: str, reply_markup: dict | None = None):
    url = f"{TG_API}/editMessageText"
    data = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if reply_markup:
        data["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)
    requests.post(url, data=data, timeout=15)


def tg_send_chat_action(chat_id: int, action: str):
    url = f"{TG_API}/sendChatAction"
    data = {"chat_id": chat_id, "action": action}
    requests.post(url, data=data, timeout=10)


def get_updates(offset=None, timeout=25):
    url = f"{TG_API}/getUpdates"
    params = {"timeout": timeout, "allowed_updates": json.dumps(["message", "callback_query"])}
    if offset is not None:
        params["offset"] = offset
    r = requests.get(url, params=params, timeout=timeout + 10)
    r.raise_for_status()
    return r.json().get("result", [])


# ============== Клавиатуры ==============
def kb_voice():
    rows, row = [], []
    for code, ru in VOICE_LABELS.items():
        row.append({"text": ru, "callback_data": f"voice:{code}"})
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([{ "text": "⬅ Назад", "callback_data": "menu:main" }])
    return {"inline_keyboard": rows}


def kb_main():
    rows = [
        [{"text": "Выбрать голос", "callback_data": "menu:voice"}],
        [{"text": "Скорость", "callback_data": "menu:speed"}],
        [{"text": "Выбрать формат", "callback_data": "menu:format"}],
        [{"text": "Справка", "callback_data": "menu:help"}],
    ]
    if DONATE_URL:
        rows.append([{ "text": "Поддержать", "url": DONATE_URL }])
    else:
        rows.append([{ "text": "Поддержать", "callback_data": "menu:donate" }])
    return {"inline_keyboard": rows}


def kb_format():
    label = {
        "OGG_OPUS": "OGG/Opus (рекомендуется для голосовых)",
        "MP3": "MP3 (подходит для публикаций)",
        "WAV": "WAV/LPCM (без сжатия)",
    }
    rows = [[{"text": label[f], "callback_data": f"fmt:{f}"}] for f in ALLOWED_FORMATS]
    rows.append([{ "text": "⬅ Назад", "callback_data": "menu:main" }])
    return {"inline_keyboard": rows}


def _voice_speed_options(voice: str) -> OrderedDict[str, str]:
    if voice in VARIABLE_SPEED_VOICES:
        return SPEED_PRESETS
    return OrderedDict([("1.0x", DEFAULT_SPEED)])


def _voice_pitch_options(voice: str) -> OrderedDict[str, str]:
    return OrderedDict([("Нормальный", DEFAULT_PITCH)])


def kb_speed(voice: str):
    options = _voice_speed_options(voice)
    rows = [[{"text": label, "callback_data": f"speed:{val}"}] for label, val in options.items()]
    rows.append([{ "text": "⬅ Назад", "callback_data": "menu:main" }])
    return {"inline_keyboard": rows}




def kb_reply_main():
    buttons = [[{"text": label}] for label in REPLY_BUTTONS.keys()]
    return {
        "keyboard": buttons,
        "resize_keyboard": True,
        "one_time_keyboard": False,
        "input_field_placeholder": INPUT_FIELD_PLACEHOLDER
    }


def kb_donate():
    if not DONATE_URL:
        return None
    return {
        "inline_keyboard": [[
            {"text": "Поддержать проект", "url": DONATE_URL}
        ]]
    }


def kb_donate_stars():
    rows = [[
        {"text": "5⭐",  "callback_data": "donate:5"},
        {"text": "10⭐", "callback_data": "donate:10"},
        {"text": "20⭐", "callback_data": "donate:20"},
    ], [
        {"text": "50⭐", "callback_data": "donate:50"},
        {"text": "100⭐","callback_data": "donate:100"},
    ], [
        {"text": "⬅ Назад", "callback_data": "menu:main"}
    ]]
    return {"inline_keyboard": rows}


def kb_stats():
    rows = [[
        {"text": "За день", "callback_data": "stats:day"},
        {"text": "За неделю", "callback_data": "stats:week"},
        {"text": "За месяц", "callback_data": "stats:month"},
    ], [
        {"text": "⬅ Назад", "callback_data": "menu:main"}
    ]]
    return {"inline_keyboard": rows}


# ============== Тексты ==============
HELP_TEXT = (
    "Этот бот озвучивает ваш текст в речь.\n\n"
    "Команды:\n"
    "/start — запуск\n"
    "/help — помощь\n"
    "/voice — выбрать голос\n"
    "/format — выбрать формат файла\n"
    "/speed — выбрать скорость\n"
    "/donate — поддержать проект\n\n"
    "Ограничение: до {maxlen} символов, длинные сообщения отклоняются.\n"
    "Текущие настройки: голос {voice_ru}, скорость {speed}, формат {fmt}."
)

MARKUP_HELP = (
    "Разметка TTS помогает уточнить произношение:\n"
    "**+** перед ударной гласной — ставит ударение (зам+ок).\n"
    "**sil<[t]>** — пауза t мс (например, sil<[300]>).\n"
    "**<[small]>/<[medium]>/<[large]>** — контекстная пауза.\n"
    "**<[accented]>слово** или **…** — акцент.\n"
    "**[[фонемы]]** — фонетическое произношение.\n\n"
    "Разметка — подсказка, а не строгая команда: не генерирует тишину и работает только в тексте. Используйте с русским языком и API v3."
)


def stats_build_text(period: str) -> str:
    import datetime as _dt
    today = _dt.date.today()
    if period == 'day':
        days = 1
        title = 'За день'
    elif period == 'month':
        days = 30
        title = 'За месяц'
    else:
        days = 7
        title = 'За неделю'

    date_keys = [(today - _dt.timedelta(days=i)).isoformat() for i in range(days)]
    total_msgs = sum(usage_by_day.get(d, 0) for d in date_keys)
    ok_cnt = sum(tts_ok_by_day.get(d, 0) for d in date_keys)
    err_cnt = sum(tts_err_by_day.get(d, 0) for d in date_keys)

    agg_users = defaultdict(int)
    for d in date_keys:
        per_day = usage_by_user_day.get(d, {})
        for uid, cnt in per_day.items():
            agg_users[uid] += cnt
    top = sorted(agg_users.items(), key=lambda x: x[1], reverse=True)[:5]
    top_lines = "\n".join([f"{uid}: {cnt}" for uid, cnt in top]) or "—"

    metrics_lines = _metrics_summary_lines()
    metrics_block = "\n".join(metrics_lines)

    return (
        f"Статистика {title}:\n"
        f"Сообщений: {total_msgs}\n"
        f"Успешно: {ok_cnt}\n"
        f"Ошибок TTS: {err_cnt}\n\n"
        f"Топ пользователей:\n{top_lines}\n\n"
        f"Метрики:\n{metrics_block}"
    )


# ============== Команды и коллбэки ==============
def ensure_prefs(chat_id: int):
    prefs = user_prefs_cache.get(chat_id) if user_prefs_cache else None
    if prefs is None:
        prefs = _load_user_pref_from_db(chat_id)
        if prefs is None:
            prefs = {
                "voice": DEFAULT_VOICE,
                "speed": DEFAULT_SPEED,
                "format": "OGG_OPUS",
                "role": "neutral",
                "pitch": DEFAULT_PITCH,
            }
            try:
                _persist_user_pref(chat_id, prefs)
            except Exception:
                log.exception("user prefs persist error")
    if user_prefs_cache:
        user_prefs_cache.set(chat_id, prefs)
    _normalize_voice_settings(prefs)
    _metrics_register_user(chat_id, prefs)
    return prefs


def handle_command(chat_id: int, text: str):
    low = text.lower().strip()
    prefs = ensure_prefs(chat_id)
    voice_ru = VOICE_LABELS.get(prefs["voice"], prefs["voice"])

    if low in ("/start", "start"):
        tg_send_text(chat_id, "Привет! Озвучу твой текст до 3000 символов. Используй кнопки ниже или команды.", reply_markup=kb_reply_main())
        return True

    if low in ("/help", "help"):
        help_text = HELP_TEXT.format(maxlen=MAX_LEN, voice_ru=voice_ru, speed=prefs["speed"], fmt=prefs["format"])
        help_text = f"{help_text}\n\n{MARKUP_HELP}"
        tg_send_text(chat_id, help_text, reply_markup=kb_reply_main())
        return True

    if low.startswith("/voice"):
        tg_send_text(chat_id, f"Выберите голос (текущий: {voice_ru}):", reply_markup=kb_voice())
        return True

    if low.startswith("/format"):
        tg_send_text(chat_id, f"Выберите формат файла (текущий: {prefs['format']}):", reply_markup=kb_format())
        return True

    if low.startswith("/speed"):
        options = _voice_speed_options(prefs["voice"])
        if len(options) <= 1:
            tg_send_text(chat_id, f"Голос {voice_ru} поддерживает только стандартную скорость 1.0x.", reply_markup=kb_reply_main())
        else:
            tg_send_text(chat_id, f"Выберите скорость (голос: {voice_ru}):", reply_markup=kb_speed(prefs["voice"]))
        return True

    if low.startswith("/donate"):
        if DONATE_URL:
            tg_send_text(chat_id, "Спасибо за поддержку!", reply_markup=kb_donate())
        else:
            tg_send_text(chat_id, "Поддержать проект:", reply_markup=kb_donate_stars())
        return True

    if low.startswith("/export_stats"):
        if chat_id not in ADMIN_IDS:
            tg_send_text(chat_id, "Доступ запрещён")
            return True
        try:
            path = _export_user_stats_csv()
            tg_send_document(chat_id, path, caption="Статистика пользователей")
        except Exception:
            tg_send_text(chat_id, "Ошибка экспорта статистики.")
        return True

    if low.startswith("/stats"):
        if chat_id not in ADMIN_IDS:
            tg_send_text(chat_id, "Доступ запрещён")
            return True
        text = stats_build_text('week')
        tg_send_text(chat_id, text, reply_markup=kb_stats())
        return True

    return False


def handle_callback(cb: dict):
    cb_id = cb.get("id")
    msg = cb.get("message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    message_id = msg.get("message_id")
    data = cb.get("data", "")

    if not chat_id or not message_id or not data:
        tg_answer_callback(cb_id, "Ошибка данных callback")
        return

    prefs = ensure_prefs(chat_id)

    if data.startswith("menu:"):
        action = data.split(":", 1)[1]
        if action in ("main", "back", "home"):
            tg_edit_message_text(chat_id, message_id, "Главное меню:", reply_markup=kb_main())
            tg_answer_callback(cb_id)
            return
        if action == "voice":
            tg_edit_message_text(chat_id, message_id, "Выберите голос:", reply_markup=kb_voice())
            tg_answer_callback(cb_id)
            return
        if action == "format":
            tg_edit_message_text(chat_id, message_id, "Выберите формат файла:", reply_markup=kb_format())
            tg_answer_callback(cb_id)
            return
        if action == "speed":
            options = _voice_speed_options(prefs["voice"])
            if len(options) <= 1:
                tg_answer_callback(cb_id, "Этот голос поддерживает только 1.0x", True)
                return
            tg_edit_message_text(chat_id, message_id, "Выберите скорость:", reply_markup=kb_speed(prefs["voice"]))
            tg_answer_callback(cb_id)
            return
        if action == "help":
            voice_ru = VOICE_LABELS.get(prefs["voice"], prefs["voice"])
            tg_edit_message_text(chat_id, message_id, HELP_TEXT.format(maxlen=MAX_LEN, voice_ru=voice_ru, speed=prefs["speed"], fmt=prefs["format"]), reply_markup=kb_main())
            tg_answer_callback(cb_id)
            return
        if action == "stats":
            if chat_id not in ADMIN_IDS:
                tg_answer_callback(cb_id, "Доступ только администратору")
                return
            tg_edit_message_text(chat_id, message_id, stats_build_text('week'), reply_markup=kb_stats())
            tg_answer_callback(cb_id)
            return
        if action == "donate":
            if DONATE_URL:
                tg_edit_message_text(chat_id, message_id, "Поддержать проект:", reply_markup=kb_donate())
            else:
                tg_edit_message_text(chat_id, message_id, "Поддержать проект:", reply_markup=kb_donate_stars())
            tg_answer_callback(cb_id)
            return

    if data.startswith("voice:"):
        code = data.split(":", 1)[1]
        if code not in VOICE_LABELS:
            tg_answer_callback(cb_id, "Неизвестный голос")
            return
        prefs["voice"] = code if code != "madirus" else "madi_ru"
        old_speed = prefs.get("speed")
        old_pitch = prefs.get("pitch")
        _normalize_voice_settings(prefs)
        try:
            _persist_user_pref(chat_id, prefs)
        except Exception:
            log.exception("user prefs persist error")
        _metrics_register_user(chat_id, prefs)
        tg_answer_callback(cb_id, f"Голос: {VOICE_LABELS.get(code, code)}")
        tg_edit_message_text(chat_id, message_id, f"Голос выбран: {VOICE_LABELS.get(code, code)}", reply_markup=kb_main())
        if prefs.get("speed") != old_speed or prefs.get("pitch") != old_pitch:
            tg_send_text(chat_id, "Настройки скорости/тембра сброшены из-за ограничений голоса.", reply_markup=kb_reply_main())
        return

    if data.startswith("fmt:"):
        f = data.split(":", 1)[1]
        if f not in ALLOWED_FORMATS:
            tg_answer_callback(cb_id, "Неподдерживаемый формат")
            return
        prefs["format"] = f
        try:
            _persist_user_pref(chat_id, prefs)
        except Exception:
            log.exception("user prefs persist error")
        _metrics_register_user(chat_id, prefs)
        tg_answer_callback(cb_id, f"Формат: {f}")
        tg_edit_message_text(chat_id, message_id, f"Формат выбран: {f}", reply_markup=kb_main())
        return

    if data.startswith("speed:"):
        val = data.split(":", 1)[1]
        options = _voice_speed_options(prefs["voice"])
        allowed = set(options.values())
        if val not in allowed:
            tg_answer_callback(cb_id, "Скорость недоступна для этого голоса", True)
            return
        prefs["speed"] = val
        try:
            _persist_user_pref(chat_id, prefs)
        except Exception:
            log.exception("user prefs persist error")
        tg_answer_callback(cb_id, f"Скорость: {val}x")
        tg_edit_message_text(chat_id, message_id, f"Скорость выбрана: {val}x", reply_markup=kb_main())
        return


    if data.startswith("stats:"):
        period = data.split(":", 1)[1]
        if chat_id not in ADMIN_IDS:
            tg_answer_callback(cb_id, "Доступ только администратору")
            return
        if period not in ("day", "week", "month"):
            period = "week"
        tg_edit_message_text(chat_id, message_id, stats_build_text(period), reply_markup=kb_stats())
        tg_answer_callback(cb_id)
        return

    if data.startswith("role:"):
        tg_answer_callback(cb_id, "Роли не поддерживаются")
        return


    if data.startswith("donate:"):
        amount_str = data.split(":", 1)[1]
        try:
            amount = int(amount_str)
            send_stars_invoice(chat_id, amount)
            tg_answer_callback(cb_id, f"Спасибо за {amount}⭐")
        except Exception as e:
            log.exception("donate stars error: %r", e)
            tg_answer_callback(cb_id, "Ошибка оформления покупки")
        return

    tg_answer_callback(cb_id, "Ок")


# ============== Донат (Stars) ==============
def kb_donate_stars_only():
    return kb_donate_stars()


def send_stars_invoice(chat_id: int, stars: int):
    url = f"{TG_API}/sendInvoice"
    title = DONATE_TITLE
    description = DONATE_DESCRIPTION
    payload = json.dumps({"type": "donate", "stars": stars, "ts": int(time.time())}, ensure_ascii=False)
    currency = "XTR"  # Telegram Stars
    prices = json.dumps([{ "label": f"Поддержка {stars}⭐", "amount": stars }], ensure_ascii=False)
    data = {
        "chat_id": chat_id,
        "title": title,
        "description": description,
        "payload": payload,
        "currency": currency,
        "prices": prices,
    }
    if currency == "XTR":
        data["provider_token"] = ""
    elif PAYMENT_PROVIDER_TOKEN:
        data["provider_token"] = PAYMENT_PROVIDER_TOKEN
    r = requests.post(url, data=data, timeout=30)
    if r.status_code != 200:
        log.error("TG sendInvoice ERROR %s: %s", r.status_code, r.text[:500])
        _metrics_track_telegram_error(context="sendInvoice", details=r.text[:200])
        r.raise_for_status()


def handle_pre_checkout_query(query: dict):
    query_id = query.get("id")
    from_user = query.get("from") or {}
    chat_id = from_user.get("id")
    try:
        tg_answer_pre_checkout(query_id, ok=True)
        log.info("PreCheckout ok for chat %s", chat_id)
    except Exception as exc:
        log.exception("pre checkout error: %r", exc)
        _metrics_track_telegram_error(exc, context="pre_checkout", details=str(exc))


def handle_successful_payment(chat_id: int, payment: dict):
    log.info("Successful payment chat=%s payload=%s amount=%s", chat_id, payment.get("invoice_payload"), payment.get("total_amount"))
    tg_send_text(chat_id, "Спасибо за донат! Звёзды получены 🌟")


# ============== Главный цикл ==============
def main():
    _acquire_single_instance_lock()

    # sanity + сброс webhook
    try:
        me = requests.get(f"{TG_API}/getMe", timeout=15).json()
        log.info("getMe: %s", me)
    except Exception as e:
        log.error("getMe error: %r", e)
    try:
        clear = requests.get(f"{TG_API}/setWebhook", params={"url": ""}, timeout=15).json()
        log.info("Webhook clear: %s", clear)
    except Exception as e:
        log.warning("setWebhook clear error: %r", e)

    print("Режим: long polling активен. Для webhooks установите URL: setWebhook?url=")

    offset = None
    while True:
        try:
            updates = get_updates(offset=offset, timeout=25)
            for upd in updates:
                offset = upd["update_id"] + 1

                if "callback_query" in upd:
                    try:
                        handle_callback(upd["callback_query"])
                    except Exception as e:
                        log.exception("handle_callback error: %r", e)
                    continue
                if "pre_checkout_query" in upd:
                    try:
                        handle_pre_checkout_query(upd["pre_checkout_query"])
                    except Exception as e:
                        log.exception("pre_checkout handler error: %r", e)
                    continue

                msg = upd.get("message") or upd.get("edited_message") or {}
                chat = msg.get("chat") or {}
                chat_id = chat.get("id")
                document = msg.get("document")
                text = msg.get("text") or msg.get("caption")
                success_payment = msg.get("successful_payment")

                if document and not text:
                    text = _handle_document_message(chat_id, document)
                if success_payment:
                    try:
                        handle_successful_payment(chat_id, success_payment)
                    except Exception as e:
                        log.exception("successful_payment handler error: %r", e)

                if not (chat_id and isinstance(text, str)):
                    continue

                text = text.strip()
                mapped = REPLY_BUTTON_COMMANDS.get(text.lower())
                if mapped:
                    text = mapped
                if not text:
                    continue

                # команды
                if text.startswith("/"):
                    try:
                        if handle_command(chat_id, text):
                            continue
                    except Exception as e:
                        log.exception("handle_command error: %r", e)
                        tg_send_text(chat_id, "Ошибка обработки команды. Попробуйте ещё раз.")
                        continue

                # подготовка
                clean_text = _normalize_text_for_tts(text)
                if not clean_text:
                    tg_send_text(chat_id, "Сообщение содержит только неподдерживаемые символы. Удалите emoji/служебные символы и попробуйте снова.")
                    continue
                if len(clean_text) > MAX_LEN:
                    tg_send_text(chat_id, f"Сообщение слишком длинное (> {MAX_LEN} символов после очистки). Сократите текст и отправьте снова.")
                    continue
                sample = clean_text
                _metrics_track_text(chat_id, len(sample))
                chunks = _split_text_for_tts(sample, TTS_CHUNK_LIMIT)
                chunk_count = len(chunks)
                prefs = ensure_prefs(chat_id)
                voice  = prefs["voice"]
                speed  = prefs["speed"]
                outfmt = prefs["format"]
                role   = "neutral"
                pitch  = prefs.get("pitch", DEFAULT_PITCH)

                # учёт
                import datetime as _dt
                today = _dt.date.today().isoformat()
                global _last_stats_day
                if _last_stats_day is None:
                    _last_stats_day = today
                elif today != _last_stats_day:
                    _save_stats_for_day(_last_stats_day)
                    _last_stats_day = today
                global usage_total, tts_ok, tts_err
                usage_total += 1
                usage_per_user[chat_id] += 1
                usage_by_day[today] += 1
                usage_by_user_day[today][chat_id] += 1

                status_message_id = tg_send_text(chat_id, "🎙️ Диктор записывает аудио...")
                def _clear_status():
                    nonlocal status_message_id
                    if status_message_id:
                        try:
                            tg_delete_message(chat_id, status_message_id)
                        except Exception:
                            pass
                        status_message_id = None

                chunk_trim_notified = False
                message_success = True
                audio_chunks_data: list[bytes] = []

                for idx, chunk in enumerate(chunks, start=1):
                    tg_send_chat_action(chat_id, "record_voice")
                    trimmed_chunk = _trim_chunk_to_limit(chunk, TTS_CHUNK_LIMIT)
                    if trimmed_chunk != chunk and not chunk_trim_notified:
                        tg_send_text(chat_id, f"Часть {idx}/{chunk_count} укорочена до {TTS_CHUNK_LIMIT} символов из-за ограничений TTS.")
                        chunk_trim_notified = True
                    chunk = trimmed_chunk
                    chunk_cache_key = (chunk, voice, speed, pitch, outfmt, role)
                    audio = cache.get(chunk_cache_key)

                    if audio is None:
                        _metrics_track_cache_miss()
                        if not allow_request(chat_id):
                            tg_send_text(chat_id, f"Превышен лимит запросов. Часть {idx}/{chunk_count} не будет озвучена.")
                            message_success = False
                            break
                        try:
                            audio = synth_tts(chunk, voice=voice, speed=speed, out_format=outfmt, role=role, pitch=pitch)
                            cache.set(chunk_cache_key, audio)
                        except requests.exceptions.ReadTimeout:
                            log.warning("SpeechKit timeout")
                            _metrics_track_speechkit_error()
                            tg_send_text(chat_id, f"SpeechKit не ответил на части {idx}/{chunk_count}. Попробуйте ещё раз.")
                            message_success = False
                            break
                        except Exception as e:
                            log.exception("TTS error: %r", e)
                            _metrics_track_speechkit_error()
                            err_msg = "Ошибка синтеза TTS."
                            if isinstance(e, requests.HTTPError):
                                try:
                                    err_data = e.response.json()
                                    message = err_data.get("error", {}).get("message")
                                    if message:
                                        err_msg = f"Ошибка TTS: {message}"
                                except Exception:
                                    pass
                            tg_send_text(chat_id, f"{err_msg} (часть {idx}/{chunk_count})")
                            message_success = False
                            break
                    else:
                        _metrics_track_cache_hit()

                    if len(audio) > 18_000_000:
                        tg_send_text(chat_id, "Файл слишком большой (>18 МБ). Уменьшите длину текста или смените формат.")
                        message_success = False
                        break

                    audio_chunks_data.append(audio)

                if not message_success or len(audio_chunks_data) != chunk_count:
                    tts_err += 1
                    tts_err_by_day[today] += 1
                    _save_stats_for_day(today)
                    _clear_status()
                    continue

                send_success = False
                try:
                    if chunk_count > 1:
                        combined_audio = _combine_audio_chunks(audio_chunks_data, outfmt)
                        caption = _build_caption(voice, outfmt, sample) + f" (соединено из {chunk_count} частей)"
                        _send_audio_blob(chat_id, combined_audio, outfmt, caption, sample)
                    else:
                        audio = audio_chunks_data[0]
                        caption = _build_caption(voice, outfmt, sample)
                        _send_audio_blob(chat_id, audio, outfmt, caption, sample)
                    send_success = True
                except Exception as e:
                    log.exception("send combined audio error: %r", e)
                    send_success = False

                if not send_success and chunk_count > 1:
                    try:
                        tg_send_text(chat_id, "Склейка не удалась, отправляю части по отдельности.")
                        send_success = True
                        for idx, audio in enumerate(audio_chunks_data, start=1):
                            caption = _build_caption(voice, outfmt, chunks[idx - 1]) + f" [{idx}/{chunk_count}]"
                            try:
                                _send_audio_blob(chat_id, audio, outfmt, caption, chunks[idx - 1])
                            except Exception as chunk_err:
                                log.exception("send fallback chunk error: %r", chunk_err)
                                tg_send_text(chat_id, f"Ошибка при отправке части {idx}/{chunk_count}.")
                                send_success = False
                                continue
                    except Exception as e:
                        log.exception("send fallback audio error: %r", e)
                        tg_send_text(chat_id, "Ошибка отправки аудио в Telegram.")
                        send_success = False

                if send_success:
                    tts_ok += 1
                    tts_ok_by_day[today] += 1
                else:
                    tts_err += 1
                    tts_err_by_day[today] += 1
                _save_stats_for_day(today)
                _clear_status()

        except KeyboardInterrupt:
            print("\nВыход: Ctrl+C")
            break
        except requests.exceptions.ReadTimeout:
            continue
        except Exception as e:
            log.exception("Main loop error: %r", e)
            time.sleep(2)


if __name__ == "__main__":
    # При выходе — сохранить статистику за сегодня
    atexit.register(lambda: _save_stats_for_day(dt.date.today().isoformat()))
    atexit.register(_save_metrics_state)
    main()
