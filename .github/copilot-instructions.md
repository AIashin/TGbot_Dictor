# Copilot Instructions for Telegram TTS Bot

## Project Overview
This is a Telegram bot for text-to-speech (TTS) synthesis using Yandex SpeechKit v3. Users send text messages to the bot, which replies with synthesized audio in various formats and voices. The bot supports voice, speed, format, and role/emotion selection via Telegram inline keyboards.

## Key Files
- `bot.py`: Main bot logic, Telegram API integration, TTS synthesis, command handling, caching, and rate limiting.
- `.env`: Required for secrets (`BOT_TOKEN`, `YC_API_KEY`, etc.).
- `requirements.txt`: Python dependencies (`requests`, `python-dotenv`).

## Architecture & Data Flow
- Telegram messages are polled via `getUpdates` (no webhook by default).
- User preferences (voice, speed, format, role) are stored in-memory per chat.
- TTS requests are sent to Yandex SpeechKit v3 using NDJSON streaming.
- Audio responses are cached (LRU) to minimize repeated synthesis.
- Audio is sent back to users in the requested format (OGG_OPUS, MP3, WAV).
- Rate limiting is enforced per chat (max 5 requests per 30 seconds).

## Developer Workflows
- **Run Locally:**
  - Ensure `.env` contains valid `BOT_TOKEN` and `YC_API_KEY`.
  - Install dependencies: `pip install -r requirements.txt`
  - Start bot: `python bot.py`
- **Debugging:**
  - Logging is set to INFO level; errors and warnings are printed to console.
  - Webhook is cleared on startup; bot runs in polling mode.
- **Configuration:**
  - Voices, roles, and formats are defined as ordered dicts in `bot.py`.
  - Extend supported voices/roles by editing `VOICE_LABELS` and `VOICE_ROLES`.

## Patterns & Conventions
- All Telegram API calls use direct HTTP requests (no external libraries).
- Inline keyboards for user selection (voice, format, role).
- Russian language is used for user-facing messages and some comments.
- Caching and rate limiting are implemented with in-memory structures (no persistence).
- Audio chunk extraction from NDJSON responses is recursive for robustness.

## Integration Points
- **Yandex SpeechKit v3:** TTS API, NDJSON streaming, requires API key.
- **Telegram Bot API:** Message polling, sending text/audio, inline keyboards.

## Examples
- To add a new voice, update `VOICE_LABELS` and `VOICE_ROLES`.
- To support a new audio format, add to `ALLOWED_FORMATS` and update handling logic.

## Limitations
- No persistent storage; all state is lost on restart.
- Only supports polling (not webhook) by default.
- Russian-centric UX and comments.

---
_If any section is unclear or missing important details, please provide feedback to improve these instructions._
