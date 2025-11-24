# Telegram TTS Bot (Yandex SpeechKit v3)

Бот для конвертации текста в речь через Yandex SpeechKit v3. Работает через long polling, хранит настройки пользователей и статистику локально в SQLite.

## Требования
- Python 3.10+
- `ffmpeg` установлен в системе (или путь в `FFMPEG_BIN`)
- Доступ к интернету, ключи Telegram и Yandex SpeechKit

## Быстрый запуск локально
```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.template .env         # или заполнить вручную
python bot.py
```

## Переменные окружения (.env)
- `BOT_TOKEN` — токен Telegram бота.
- `ADMIN_IDS` — список chat_id админов (через запятую/точку с запятой).
- `YC_API_KEY` — API‑ключ SpeechKit v3.
- `TTS_URL` — эндпоинт TTS (по умолчанию v3).
- `VOICE` / `SPEED` / `TTS_CHUNK_LIMIT` / `MAX_LEN` — базовые настройки синтеза.
- `MAX_DOCUMENT_SIZE` — лимит размера входных файлов.
- `INPUT_PLACEHOLDER` — текст placeholder для клавиатуры.
- `FFMPEG_BIN` — путь к ffmpeg, если не в `PATH`.
- `DONATE_URL` / `PAYMENT_PROVIDER_TOKEN` / `DONATE_TITLE` / `DONATE_DESCRIPTION` — настройки донатов/Stars.

См. пример в `.env.template`. Никогда не коммить реальные секреты.

## Что не коммитить (уже в .gitignore)
- Секреты: `.env`.
- Данные и метрики: `bot.db`, `metrics.json`, `stats*.json`, `stats.csv`, `telegram_errors.log`, `bot.backup.txt`.
- Бинарники и кэш: `ffmpeg.zip`, `ffmpeg_extracted/`, `__pycache__/`.
- IDE/редактор: `.vscode/`, `.idea/`.

## Минимальные шаги для GitHub
1. Убедиться, что секреты и данные не попадают в коммит.
2. `git add` только код, шаблоны и документацию.
3. Создать репозиторий, `git remote add origin <url>`, `git push`.
4. Хранить токены в GitHub Secrets, а на сервере — в `.env`.

## Развёртывание (коротко)
- **ВМ/контейнер**: самый быстрый путь; установить Python + ffmpeg, развернуть код, запускать `python bot.py` (или через systemd/Docker).
- **Хранение данных**: файлы `bot.db`, `metrics.json`, `stats*.json`, `stats.csv` должны жить на постоянном диске/volume.
