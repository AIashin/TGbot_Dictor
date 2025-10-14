#!/usr/bin/env python
# coding: utf-8

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

# ==== Конфигурация =========

TG_TOKEN   = os.getenv("BOT_TOKEN", "").strip()
YC_API_KEY = os.getenv("YC_API_KEY", "").strip()
TTS_URL    = os.getenv("TTS_URL", "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis").strip()
TG_API     = f"https://api.telegram.org/bot{TG_TOKEN}"
DONATE_URL = os.getenv("DONATE_URL", "").strip()
PAYMENT_PROVIDER_TOKEN = os.getenv("PAYMENT_PROVIDER_TOKEN", "").strip()  # не нужен для Stars (XTR)
DONATE_TITLE = os.getenv("DONATE_TITLE", "Поддержать проект")
DONATE_DESCRIPTION = os.getenv("DONATE_DESCRIPTION", "Спасибо за поддержку!")

DEFAULT_VOICE = os.getenv("VOICE", "marina")
DEFAULT_SPEED = os.getenv("SPEED", "1.0")
MAX_LEN       = int(os.getenv("MAX_LEN", "1000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tts-bot")
log.setLevel(logging.INFO)

if not TG_TOKEN:
    raise RuntimeError("Переменная BOT_TOKEN не задана. Укажите её в файле .env (BOT_TOKEN=...).")
if not YC_API_KEY:
    raise RuntimeError("Переменная YC_API_KEY не задана. Укажите её в файле .env (YC_API_KEY=...).")

# ============== Подписи голосов (код -> русское имя) ==============
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
    ("madirus",   "Мади (устар.)"),
    ("saule_ru",  "Сауле"),
    ("omazh",     "Омаж"),
    ("yulduz_ru", "Юлдуз"),
])

# ============== Роли/эмоции ==============
ROLE_LABELS = OrderedDict([
    ("neutral",  "нейтральный"),
    ("good",     "добрый"),
    ("evil",     "злой"),
    ("cheerful", "весёлый"),
    ("sad",      "грустный"),
    ("angry",    "сердитый"),
    ("strict",   "строгий"),
])

# Поддержка ролей для голосов (если не указано, используется neutral)
VOICE_ROLES = {
    # мужские, поддерживают good/evil
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
    # национальные/узкоспециальные
    "madi_ru":   ["neutral"],
    "madirus":   ["neutral"],
    "saule_ru":  ["neutral"],
    "omazh":     ["neutral"],
    "yulduz_ru": ["neutral"],
}

# ============== Аудиоформаты (поддержка) ==============
ALLOWED_FORMATS = ["OGG_OPUS", "MP3", "WAV"]

# ============== Состояние пользователя ==============
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
    """Ищет и собирает все поля audioChunk.data в произвольном JSON."""
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
    """Синтезирует речь в формате out_format (OGG_OPUS/MP3/WAV) через SpeechKit v3 NDJSON."""
    if out_format not in ALLOWED_FORMATS:
        raise ValueError(f"Формат {out_format} не поддерживается. Доступно: {', '.join(ALLOWED_FORMATS)}")

    if voice == "madirus":
        voice = "madi_ru"

    hints = [{"voice": voice}, {"speed": speed}]
    if role and role != "neutral":
        hints.append({"role": role})

    payload = {
        "text": text,
        "hints": hints,
        "outputAudioSpec": {"containerAudio": {"containerAudioType": out_format}},
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
            raise RuntimeError("SpeechKit v3: нет данных audioChunk. Проверьте текст/параметры/формат.")

        return base64.b64decode("".join(b64_parts))


# ============== Telegram helpers ==============
def tg_send_text(chat_id: int, text: str, reply_markup: dict | None = None):
    url = f"{TG_API}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
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
        "OGG_OPUS": "OGG/Opus (голосовое сообщение)",
        "MP3": "MP3 (аудиофайл)",
        "WAV": "WAV/LPCM (без сжатия)",
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


# ============== Подсказка ==============
HELP_TEXT = (
    "Я голосовой бот на Yandex SpeechKit.\n\n"
    "Команды:\n"
    "/start — приветствие\n"
    "/help — эта подсказка\n"
    "/voice — выбрать голос\n"
    "/format — выбрать формат аудио\n"
    "/donate — поддержать проект\n\n"
    "Отправьте текст (до {maxlen} символов), и я его озвучу.\n"
    "Сейчас: голос «{voice_ru}», скорость {speed}, формат {fmt}."
)

def kb_donate():
    if not DONATE_URL:
        return None
    return {
        "inline_keyboard": [[
            {"text": "Поддержать проект", "url": DONATE_URL}
        ]]
    }

def kb_donate_stars():
    # Меню с фиксированными пакетами звёзд
    rows = [[
        {"text": "5⭐",  "callback_data": "donate:5"},
        {"text": "10⭐", "callback_data": "donate:10"},
        {"text": "20⭐", "callback_data": "donate:20"},
    ]]
    return {"inline_keyboard": rows}

def send_stars_invoice(chat_id: int, stars: int):
    """Отправляет счёт в звёздах (Telegram Stars). Требует включённых Stars для бота.
    Валюта 'XTR' используется для звёзд. Провайдер-токен обычно не требуется.
    """
    url = f"{TG_API}/sendInvoice"
    title = DONATE_TITLE
    description = DONATE_DESCRIPTION
    payload = json.dumps({"type": "donate", "stars": stars, "ts": int(time.time())}, ensure_ascii=False)
    currency = "XTR"  # Telegram Stars
    # Для Stars сумма указывается в звёздах; Telegram принимает XTR без provider_token.
    prices = json.dumps([{ "label": f"Донат {stars}⭐", "amount": stars }], ensure_ascii=False)
    data = {
        "chat_id": chat_id,
        "title": title,
        "description": description,
        "payload": payload,
        "currency": currency,
        "prices": prices,
    }
    if PAYMENT_PROVIDER_TOKEN:
        data["provider_token"] = PAYMENT_PROVIDER_TOKEN
    r = requests.post(url, data=data, timeout=30)
    if r.status_code != 200:
        log.error("TG sendInvoice ERROR %s: %s", r.status_code, r.text[:500])
        r.raise_for_status()

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
    role_ru  = ROLE_LABELS.get("neutral", "нейтральный")

    if low in ("/start", "start"):
        tg_send_text(chat_id, "Привет! Это бот озвучки текста на Yandex SpeechKit.\nКоманды: /voice /format /donate /help", reply_markup=kb_donate())
        return True

    if low in ("/help", "help"):
        tg_send_text(chat_id, HELP_TEXT.format(
            maxlen=MAX_LEN, voice_ru=voice_ru, speed=prefs["speed"], fmt=prefs["format"], role_ru=role_ru))
        return True

    if low.startswith("/voice"):
        tg_send_text(chat_id, f"Выберите голос (сейчас: {voice_ru}):", reply_markup=kb_voice())
        return True

    if low.startswith("/format"):
        tg_send_text(chat_id, f"Выберите формат аудио (сейчас: {prefs['format']}):", reply_markup=kb_format())
        return True

    if low.startswith("/donate"):
        if DONATE_URL:
            tg_send_text(chat_id, "Спасибо за поддержку!", reply_markup=kb_donate())
        else:
            # Покажем меню доната звёздами
            tg_send_text(chat_id, "Выберите сумму доната звёздами:", reply_markup=kb_donate_stars())
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
        tg_answer_callback(cb_id, "Некорректные данные callback"); return

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
            tg_answer_callback(cb_id, "Неподдерживаемый формат"); return
        prefs["format"] = f
        tg_answer_callback(cb_id, f"Формат: {f}")
        tg_edit_message_text(chat_id, message_id, f"Формат выбран: {f}")
        return

    if data.startswith("role:"):
        tg_answer_callback(cb_id, "Роли временно недоступны")
        return

    if data.startswith("donate:"):
        amount_str = data.split(":", 1)[1]
        try:
            amount = int(amount_str)
            send_stars_invoice(chat_id, amount)
            tg_answer_callback(cb_id, f"Счёт на {amount}⭐")
        except Exception as e:
            log.exception("donate stars error: %r", e)
            tg_answer_callback(cb_id, "Не удалось сформировать счёт")
        return

    tg_answer_callback(cb_id, "Ок")


# ============== Главный цикл ==============
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

    print("Режим: long polling активен. Пишите текст — я озвучу ответом.")
    print("Для работы через webhook укажите URL: setWebhook?url=")

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
                        tg_send_text(chat_id, "Ошибка обработки команды. Попробуйте ещё раз позже.")
                        continue

                # rate limit
                if not allow_request(chat_id):
                    tg_send_text(chat_id, "Слишком много запросов. Пожалуйста, подождите немного и повторите.")
                    continue

                # подготовка запроса
                sample = text[:MAX_LEN]
                prefs = ensure_prefs(chat_id)
                voice  = prefs["voice"]
                speed  = prefs["speed"]
                outfmt = prefs["format"]
                # принудительно используем только нейтральную роль, чтобы избежать ошибок
                role   = "neutral"

                # кеш
                cache_key = (sample, voice, speed, outfmt, role)
                audio = cache.get(cache_key)

                if audio is None:
                    try:
                        audio = synth_tts(sample, voice=voice, speed=speed, out_format=outfmt, role=role)
                        cache.set(cache_key, audio)
                    except requests.exceptions.ReadTimeout:
                        log.warning("SpeechKit timeout")
                        tg_send_text(chat_id, "Таймаут SpeechKit. Попробуйте ещё раз чуть позже.")
                        continue
                    except Exception as e:
                        log.exception("TTS error: %r", e)
                        tg_send_text(chat_id, "Ошибка synth TTS. Проверьте текст/настройки и попробуйте снова.")
                        continue

                # ограничение на размер файла
                if len(audio) > 18_000_000:
                    tg_send_text(chat_id, "Файл аудио слишком большой (>18 МБ). Укоротите текст и попробуйте снова.")
                    continue

                # отправка аудио/голоса
                try:
                    voice_ru = VOICE_LABELS.get(voice, voice)
                    role_ru  = ROLE_LABELS.get(role, role)
                    if outfmt == "OGG_OPUS":
                        tg_send_voice(chat_id, audio, caption=f"TTS v3 — {voice_ru} — {role_ru} — OGG/Opus")
                    elif outfmt == "MP3":
                        tg_send_audio(chat_id, audio, "speech.mp3", "audio/mpeg",
                                      caption=f"TTS v3 — {voice_ru} — {role_ru} — MP3")
                    elif outfmt == "WAV":
                        tg_send_audio(chat_id, audio, "speech.wav", "audio/wav",
                                      caption=f"TTS v3 — {voice_ru} — {role_ru} — WAV")
                    else:
                        tg_send_audio(chat_id, audio, "speech.bin", "application/octet-stream",
                                      caption=f"TTS v3 — {voice_ru} — {role_ru} — {outfmt}")
                except Exception as e:
                    log.exception("send audio error: %r", e)
                    tg_send_text(chat_id, "Ошибка при отправке аудио в Telegram.")

        except KeyboardInterrupt:
            print("\nВыход по Ctrl+C")
            break
        except requests.exceptions.ReadTimeout:
            continue
        except Exception as e:
            log.exception("Main loop error: %r", e)
            time.sleep(2)

if __name__ == "__main__":
    main()
