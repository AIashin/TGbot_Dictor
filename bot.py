#!/usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-
# In[12]:


import os
import io
import time
import json
import base64
import logging
import requests
from collections import deque, OrderedDict
from dotenv import load_dotenv
load_dotenv()


# ==== НАСТРОЙКИ =========

TG_TOKEN   = os.getenv("BOT_TOKEN", "").strip()
YC_API_KEY = os.getenv("YC_API_KEY", "").strip()
TTS_URL    = os.getenv("TTS_URL", "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis").strip()
TG_API     = f"https://api.telegram.org/bot{TG_TOKEN}"

DEFAULT_VOICE = os.getenv("VOICE", "marina")
DEFAULT_SPEED = os.getenv("SPEED", "1.0")
MAX_LEN       = int(os.getenv("MAX_LEN", "1000"))


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tts-bot")
log.setLevel(logging.INFO)


if not TG_TOKEN:
    raise RuntimeError("BOT_TOKEN пуст — положи его в .env")
if not YC_API_KEY:
    raise RuntimeError("YC_API_KEY пуст — положи его в .env")


# ============== Голоса (код -> имя на русском) ==============
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
    ("madirus",   "Мади (legacy)"),
    ("saule_ru",  "Сауле"),
    ("omazh",     "Омаж"),
    ("yulduz_ru", "Юлдуз"),
])

# ============== Роли/эмоции ==============
ROLE_LABELS = OrderedDict([
    ("neutral",   "Нейтрально"),
    ("good",      "Доброжелательно"),
    ("evil",      "Зло/жёстко"),
    ("cheerful",  "Весело"),
    ("sad",       "Грустно"),
    ("angry",     "Сердито"),
    ("strict",    "Строго"),
])

# Поддержка ролей по голосам (если голоса нет — считаем, что поддерживает только neutral)
VOICE_ROLES = {
    # мужские русские голоса
    "ermil":     ["neutral", "good", "evil"],
    "zahar":     ["neutral", "good", "evil"],
    "filipp":    ["neutral", "good", "evil"],
    "alexander": ["neutral", "good", "evil"],
    "kirill":    ["neutral", "good", "evil"],
    "anton":     ["neutral", "good", "evil"],
    # женские
    "marina":    ["neutral", "cheerful", "sad", "angry", "strict"],
    "masha":     ["neutral", "cheerful", "sad"],
    "lera":      ["neutral", "cheerful", "sad"],
    "alena":     ["neutral", "cheerful", "sad"],
    "dasha":     ["neutral", "cheerful", "sad"],
    "julia":     ["neutral", "cheerful", "sad"],
    "jane":      ["neutral", "cheerful", "sad"],
    # тюркские/казахские — оставим только neutral (расширишь после тестов)
    "madi_ru":   ["neutral"],
    "madirus":   ["neutral"],
    "saule_ru":  ["neutral"],
    "omazh":     ["neutral"],
    "yulduz_ru": ["neutral"],
}

# ============== Форматы (контейнеры) ==============
# SpeechKit v3 поддерживает WAV (LPCM), OGG_OPUS, MP3
ALLOWED_FORMATS = ["OGG_OPUS", "MP3", "WAV"]

# ============== Память процесса ==============
user_prefs = {}   # chat_id -> {"voice":..., "speed":..., "format":..., "role":...}
rate_state = {}   # chat_id -> deque[timestamps]

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

# ============== Rate limit ==============
RATE_LIMIT_MAX = 5
RATE_LIMIT_WINDOW = 30.0

def allow_request(chat_id: int) -> bool:
    dq = rate_state.setdefault(chat_id, deque())
    now = time.time()
    while dq and now - dq[0] > RATE_LIMIT_WINDOW:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_MAX:
        return False
    dq.append(now)
    return True

# ============== SpeechKit v3 (NDJSON) ==============
def _collect_audio_chunks_from_obj(obj):
    """Собрать все audioChunk.data из произвольной вложенности."""
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



def synth_tts(text: str, voice: str, speed: str, out_format: str, role: str | None = None) -> bytes:
    """Синтез речи в формате out_format (OGG_OPUS/MP3/WAV) через SpeechKit v3 NDJSON."""
    if out_format not in ALLOWED_FORMATS:
        raise ValueError(f"Формат {out_format} не поддержан. Доступно: {', '.join(ALLOWED_FORMATS)}")

    # алиас совместимости
    if voice == "madirus":
        voice = "madi_ru"

    hints = [{"voice": voice}, {"speed": speed}]
    if role and role != "neutral":
        hints.append({"role": role})

    payload = {
        "text": text,
        "hints": hints,
        "outputAudioSpec": {"containerAudio": {"containerAudioType": out_format}},
        # "unsafeMode": True,  # можно включить для очень длинных текстов
    }

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
            chunks = _collect_audio_chunks_from_obj(obj)
            if chunks:
                b64_parts.extend(chunks)

        if not b64_parts:
            raise RuntimeError("SpeechKit v3: не пришли audioChunk — проверь ключ/голос/скорость/format/роль.")

        return base64.b64decode("".join(b64_parts))



# ============== Telegram helpers ==============
def tg_send_text(chat_id: int, text: str, reply_markup: dict | None = None):
    url = f"{TG_API}/sendMessage"
    data = {"chat_id": chat_id, "text": text}
    if reply_markup:
        data["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)
    r = requests.post(url, data=data, timeout=30)
    if r.status_code != 200:
        log.warning("TG sendMessage ERROR %s: %s", r.status_code, r.text[:300])

def tg_send_voice(chat_id: int, ogg_bytes: bytes, caption: str | None = None):
    url = f"{TG_API}/sendVoice"
    files = {"voice": ("speech.ogg", io.BytesIO(ogg_bytes), "audio/ogg")}
    data  = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption
    r = requests.post(url, data=data, files=files, timeout=90)
    if r.status_code != 200:
        log.error("TG sendVoice ERROR %s: %s", r.status_code, r.text[:300])
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
            rows.append(row); row = []
    if row:
        rows.append(row)
    return {"inline_keyboard": rows}

def kb_format():
    label = {
        "OGG_OPUS": "OGG/Opus (voice)",
        "MP3": "MP3 (универсально)",
        "WAV": "WAV/LPCM (большой)",
    }
    rows = [[{"text": label[f], "callback_data": f"fmt:{f}"}] for f in ALLOWED_FORMATS]
    return {"inline_keyboard": rows}

def kb_role(voice_code: str):
    supp = VOICE_ROLES.get(voice_code, ["neutral"])
    rows, row = [], []
    for role in supp:
        row.append({"text": ROLE_LABELS.get(role, role), "callback_data": f"role:{role}"})
        if len(row) == 3:
            rows.append(row); row = []
    if row:
        rows.append(row)
    return {"inline_keyboard": rows}


# ============== Команды ==============
HELP_TEXT = (
    "Я озвучиваю текст 🎙️\n\n"
    "Команды:\n"
    "/start — приветствие\n"
    "/help — помощь\n"
    "/voice — выбрать голос\n"
    "/format — выбрать формат файла\n"
    "/role — выбрать роль/эмоцию (если поддерживается)\n\n"
    "Пришлите текст (до {maxlen} симв.) — верну аудио.\n"
    "Сейчас: голос «{voice_ru}», скорость {speed}, формат {fmt}, роль {role_ru}."
)

def ensure_prefs(chat_id: int):
    return user_prefs.setdefault(chat_id, {
        "voice": DEFAULT_VOICE,
        "speed": DEFAULT_SPEED,
        "format": "OGG_OPUS",
        "role": "neutral",
    })

def handle_command(chat_id: int, text: str):
    low = text.lower().strip()
    prefs = ensure_prefs(chat_id)
    voice_ru = VOICE_LABELS.get(prefs["voice"], prefs["voice"])
    role_ru  = ROLE_LABELS.get(prefs.get("role", "neutral"), "Нейтрально")

    if low in ("/start", "start"):
        tg_send_text(chat_id, "Привет! Пришли текст — озвучу 🎧\nКоманды: /voice /format /role /help")
        return True

    if low in ("/help", "help"):
        tg_send_text(chat_id, HELP_TEXT.format(
            maxlen=MAX_LEN, voice_ru=voice_ru, speed=prefs["speed"], fmt=prefs["format"], role_ru=role_ru))
        return True

    if low.startswith("/voice"):
        tg_send_text(chat_id, f"Выбери голос (текущий: {voice_ru}):", reply_markup=kb_voice())
        return True

    if low.startswith("/format"):
        tg_send_text(chat_id, f"Выбери формат файла (текущий: {prefs['format']}):", reply_markup=kb_format())
        return True

    if low.startswith("/role"):
        voice_code = prefs["voice"]
        tg_send_text(
            chat_id,
            f"Выбери роль/эмоцию для «{VOICE_LABELS.get(voice_code, voice_code)}» (текущая: {role_ru}):",
            reply_markup=kb_role(voice_code)
        )
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
        tg_answer_callback(cb_id, "Некорректный callback"); return

    prefs = ensure_prefs(chat_id)

    if data.startswith("voice:"):
        code = data.split(":", 1)[1]
        if code not in VOICE_LABELS:
            tg_answer_callback(cb_id, "Неизвестный голос"); return
        prefs["voice"] = code if code != "madirus" else "madi_ru"
        tg_answer_callback(cb_id, f"Голос: {VOICE_LABELS.get(code, code)}")
        tg_edit_message_text(chat_id, message_id, f"Голос выбран: {VOICE_LABELS.get(code, code)}")
        return

    if data.startswith("fmt:"):
        f = data.split(":", 1)[1]
        if f not in ALLOWED_FORMATS:
            tg_answer_callback(cb_id, "Формат недоступен"); return
        prefs["format"] = f
        tg_answer_callback(cb_id, f"Формат: {f}")
        tg_edit_message_text(chat_id, message_id, f"Формат выбран: {f}")
        return

    if data.startswith("role:"):
        role = data.split(":", 1)[1]
        voice_code = prefs.get("voice", DEFAULT_VOICE)
        allowed = VOICE_ROLES.get(voice_code, ["neutral"])
        if role not in allowed:
            tg_answer_callback(cb_id, "Эта роль не поддерживается выбранным голосом"); return
        prefs["role"] = role
        tg_answer_callback(cb_id, f"Роль: {ROLE_LABELS.get(role, role)}")
        tg_edit_message_text(chat_id, message_id, f"Роль выбрана: {ROLE_LABELS.get(role, role)}")
        return

    tg_answer_callback(cb_id, "Ок")


# ============== Основной цикл ==============
def main():
    # sanity + очистка webhook
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

    print("✅ Бот запущен локально. Открой чат с ботом и пришли текст.")
    print("Если тишина — убедись, что webhook отключён: setWebhook?url=")

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

                msg = upd.get("message") or upd.get("edited_message") or {}
                chat = msg.get("chat") or {}
                chat_id = chat.get("id")
                text = msg.get("text")

                if not (chat_id and isinstance(text, str)):
                    continue

                text = text.strip()
                if not text:
                    continue

                # команды
                if text.startswith("/"):
                    try:
                        if handle_command(chat_id, text):
                            continue
                    except Exception as e:
                        log.exception("handle_command error: %r", e)
                        tg_send_text(chat_id, "Что-то пошло не так с командой. Попробуйте ещё раз.")
                        continue

                # rate limit
                if not allow_request(chat_id):
                    tg_send_text(chat_id, "Слишком часто. Пожалуйста, подождите немного ⏳")
                    continue

                # ограничение длины
                sample = text[:MAX_LEN]
                prefs = ensure_prefs(chat_id)
                voice  = prefs["voice"]
                speed  = prefs["speed"]
                outfmt = prefs["format"]
                role   = prefs.get("role", "neutral")

                # кэш
                cache_key = (sample, voice, speed, outfmt, role)
                audio = cache.get(cache_key)

                if audio is None:
                    try:
                        audio = synth_tts(sample, voice=voice, speed=speed, out_format=outfmt, role=role)
                        cache.set(cache_key, audio)
                    except requests.exceptions.ReadTimeout:
                        log.warning("SpeechKit timeout")
                        tg_send_text(chat_id, "⚠️ SpeechKit не ответил вовремя. Попробуйте ещё раз.")
                        continue
                    except Exception as e:
                        log.exception("TTS error: %r", e)
                        tg_send_text(chat_id, "⚠️ Не удалось озвучить. Попробуйте короче или позже.")
                        continue

                # лимит размера
                if len(audio) > 18_000_000:
                    tg_send_text(chat_id, "Аудио получилось слишком большим (>18 МБ). Укоротите текст, пожалуйста.")
                    continue

                # отправка в зависимости от формата
                try:
                    voice_ru = VOICE_LABELS.get(voice, voice)
                    role_ru  = ROLE_LABELS.get(role, role)
                    if outfmt == "OGG_OPUS":
                        tg_send_voice(chat_id, audio, caption=f"TTS v3 · {voice_ru} · {role_ru} · OGG/Opus")
                    elif outfmt == "MP3":
                        tg_send_audio(chat_id, audio, "speech.mp3", "audio/mpeg",
                                      caption=f"TTS v3 · {voice_ru} · {role_ru} · MP3")
                    elif outfmt == "WAV":
                        tg_send_audio(chat_id, audio, "speech.wav", "audio/wav",
                                      caption=f"TTS v3 · {voice_ru} · {role_ru} · WAV")
                    else:
                        tg_send_audio(chat_id, audio, "speech.bin", "application/octet-stream",
                                      caption=f"TTS v3 · {voice_ru} · {role_ru} · {outfmt}")
                except Exception as e:
                    log.exception("send audio error: %r", e)
                    tg_send_text(chat_id, "Не удалось отправить аудио в Telegram.")

        except KeyboardInterrupt:
            print("\n⏹️ Остановка по Ctrl+C")
            break
        except requests.exceptions.ReadTimeout:
            continue
        except Exception as e:
            log.exception("Main loop error: %r", e)
            time.sleep(2)

if __name__ == "__main__":
    main()




