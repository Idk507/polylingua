<!-- Plan file for PolyLingua implementation; refine freely in VS Code. -->

# polylingua

PolyLingua: A Multilingual Intelligent Speech Assistant

To develop an intelligent audio-based assistant capable of:

Converting speech to text (STT)
Translating non-English input to English or responding in the native language
Converting text to speech (TTS)
Detecting spoken language automatically (Language ID)
Enabling fluid two-way conversation across multiple languages

## Plan Overview

PolyLingua will be a modular system built around a FastAPI backend (Python), with clear components for audio handling, STT, language detection, translation, TTS, and a dialogue/conversation manager. You’ll start with a simple  prototype, then refactor into clean modules with multi‑turn conversations and persistence, and finally add optional offline/local models and optimizations.

### 1. Bootstrap Project & Minimal API
- Create Python project structure (`polylingua/` package, `main.py`).
- Add dependencies: `fastapi`, `uvicorn`, `pydantic`, `python-dotenv`, HTTP client (e.g., `httpx` or `requests`).
- Implement `create_app()` and a basic `GET /health` endpoint.


### 2. Implement Audio I/O Module
- Create `polylingua/audio/` with:
  - `AudioFormat` type (sample rate, channels, encoding).
  - `AudioProcessor` class: `load_audio_from_bytes`, `resample`, `convert_to_mono`, `normalize_volume`, `export_to_wav_bytes`, `validate_audio_duration`.
- Use libraries like `pydub` or `soundfile`/`librosa` for implementation (later).
- Ensure `AudioProcessor` exposes a single main function you’ll call from the API: `prepare_for_stt(audio_bytes) -> (wav_bytes, audio_format)`.

### 3. Implement STT Module 
- Create `polylingua/stt/` with:
  - `SttResult` (text, language, confidence).
  - `BaseSttEngine` abstract interface: `transcribe(audio_bytes, audio_format, language_hint=None)`.
   - Uses open-source Whisper model locally. Requires ffmpeg for audio processing.  - `SttRouterService` with `transcribe_with_best_engine(...)`
- Wire basic logging and error mapping for STT errors.

### 4. Implement Language Detection Module
- Create `polylingua/language/` with:
  - `LanguageDetectionResult` (language_code, confidence, source).
  - `LanguageDetector` class: `detect_from_text(text)` and optional `detect_from_audio(audio_bytes)`.
  - `normalize_language_code(raw_code)` to map provider codes to standard ISO‑like codes (`en`, `es`, etc.).
  - `LanguagePreferenceManager` to track per‑session/user language preference.
- Initially, use STT’s language output when available, and fall back to `detect_from_text`.

### 5. Implement Translation Module
- Create `polylingua/translation/` with:
  - `TranslationResult` (source_text, translated_text, source_lang, target_lang, provider).
  - `BaseTranslator`: `translate(text, source_lang, target_lang)`.
  - `TranslationService` with:
    - `translate_to_internal_language(text, detected_lang, internal_lang='en')`
    - `translate_to_user_language(text, user_lang, internal_lang='en')`
    - Optional `detect_and_translate_auto(text, target_lang)`.

### 6. Implement TTS Module 
- Create `polylingua/tts/` with:
  - `TtsRequest` (text, language, voice_id, style, speed).
  - `TtsResult` (audio_bytes, audio_format, duration).
  - `BaseTtsEngine` : `synthesize(tts_request)`.
  - `TtsRouterService` with `synthesize_with_best_engine(tts_request)`.
- Decide default language/voice; later expose per‑user preferences.

### 7. Implement Dialogue & Conversation Core
- Create `polylingua/dialogue/` with:
  - `ConversationTurn` (turn_id, user_utterance_text, languages, translations, assistant_response, timestamp).
  - `ConversationState` (session_id, turns, user_language_preference, internal_language, helpers like `add_turn`, `get_recent_context`).
  - `DialogueEngine`:
    - `summarize_conversation(conversation_state)` for summaries or logs.
  - `ConversationManager`:
    - `handle_audio_turn(session_id, audio_bytes, source_lang_hint, target_lang)` that orchestrates the full pipeline.
    - `handle_text_turn(session_id, text, source_lang, target_lang)` for text‑only input.
    - `get_conversation_state(session_id)`.

### 8. Implement Persistence & Config
- Create `polylingua/config/` with `Config.from_env()`.
- Create `polylingua/persistence/` with `ConversationRepository` and `UserRepository` for storing conversations and user profiles.

### 9. Extend API Layer to Full Conversation Endpoints
- Create `polylingua/api/` with Pydantic models and routers for:
  - `POST /api/v1/turn/audio`
  - `POST /api/v1/turn/text`
  - `GET /api/v1/session/{session_id}`
  - `POST /detect-language`, `POST /translate`, `POST /tts`, and `GET /health`.

### 10. Add Simple Web UI
- Implement a minimal browser UI with microphone recording, session management, and display of transcripts and responses.

### 11. Add Local/Offline Model Support
- Implement `LocalSttEngine`, `LocalTtsEngine`, and `LocalTranslator`.

### 12. Testing Strategy
- Use `pytest` for unit tests of each module (audio, STT, LID, translation, TTS, dialogue, API, persistence).
- Add manual scripts for end‑to‑end tests with real audio and text.
