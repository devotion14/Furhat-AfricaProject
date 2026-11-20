"""Cloud model clients: chat (MiniMax, Groq as backup) and speech-to-text (Groq Whisper).

Both providers speak the OpenAI-compatible HTTP API, so one small client covers
them; switching provider or model is a change in config/live.json, not code.

API keys are NEVER stored in the repository. They are read from environment
variables (MINIMAX_API_KEY, GROQ_API_KEY) or from config/keys.json, which is
git-ignored (copy config/keys.example.json).

Every call has a hard timeout and returns None on any failure; the caller
decides the fallback. Nothing here raises into the live loop.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
import wave

import numpy as np
import requests

from .config import CONFIG_DIR

log = logging.getLogger("nuru.llm")

KEYS_FILE = CONFIG_DIR / "keys.json"


def load_key(name: str) -> str | None:
    """name = 'minimax' or 'groq'. Environment variable wins over keys.json."""
    env = os.environ.get(f"{name.upper()}_API_KEY")
    if env:
        return env.strip()
    try:
        return (json.loads(KEYS_FILE.read_text()).get(f"{name}_api_key") or "").strip() or None
    except (FileNotFoundError, ValueError):
        return None


class ChatClient:
    """OpenAI-compatible /chat/completions client."""

    def __init__(self, name: str, base_url: str, model: str, key: str | None,
                 timeout_s: float = 8.0, json_mode: bool = False, max_tokens: int = 300):
        self.name, self.base_url, self.model, self.key = name, base_url.rstrip("/"), model, key
        self.timeout_s, self.json_mode, self.max_tokens = timeout_s, json_mode, max_tokens
        self.last_latency_s = None
        self.last_error = None

    @property
    def available(self) -> bool:
        return bool(self.key)

    def chat(self, messages: list[dict], temperature: float = 0.7, timeout_s: float | None = None) -> str | None:
        if not self.key:
            self.last_error = "no API key"
            return None
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": self.max_tokens}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.time()
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=timeout_s or self.timeout_s,
                              headers={"Authorization": f"Bearer {self.key}"})
            self.last_latency_s = time.time() - t0
            if r.status_code != 200:
                self.last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                log.warning("%s chat failed: %s", self.name, self.last_error)
                return None
            text = r.json()["choices"][0]["message"]["content"] or ""
            return strip_reasoning(text)
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            self.last_latency_s = time.time() - t0
            self.last_error = str(e)
            log.warning("%s chat failed: %s", self.name, e)
            return None


def strip_reasoning(text: str) -> str:
    """Reasoning models (e.g. MiniMax-M2) may prefix <think>...</think>; drop it."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def extract_json(text: str | None) -> dict | None:
    """First JSON object in a model reply (models sometimes wrap it in prose or ```)."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except ValueError:
        return None


class FallbackChat:
    """Try each client in order (MiniMax first, Groq second)."""

    def __init__(self, clients: list[ChatClient]):
        self.clients = [c for c in clients if c.available]
        self.used = None

    @property
    def available(self) -> bool:
        return bool(self.clients)

    def chat(self, messages, temperature=0.7, deadline_s: float = 8.0) -> str | None:
        t_end = time.time() + deadline_s
        for c in self.clients:
            left = t_end - time.time()
            if left < 0.5:
                break
            out = c.chat(messages, temperature, timeout_s=min(c.timeout_s, left))
            if out:
                self.used = c.name
                return out
        self.used = None
        return None


def make_chat(L: dict) -> FallbackChat:
    """Build the chat chain from live.json -> 'llm'."""
    cfg = L.get("llm", {})
    clients = []
    for name in cfg.get("order", ["minimax", "groq"]):
        p = cfg.get(name, {})
        clients.append(ChatClient(name, p.get("base_url", ""), p.get("model", ""), load_key(name),
                                  timeout_s=float(p.get("timeout_s", 8.0)),
                                  json_mode=bool(p.get("json_mode", False)),
                                  max_tokens=int(p.get("max_tokens", 300))))
    return FallbackChat(clients)


# ----------------------------------------------------------------- speech
WHISPER_LANG = {"english": "en", "swahili": "sw", "en": "en", "sw": "sw"}


def to_wav(audio: np.ndarray, sr: int = 16000) -> bytes:
    pcm = (np.clip(audio.reshape(-1), -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class Transcriber:
    """Groq Whisper. Language is auto-detected per utterance, so English,
    Kiswahili and mixed speech all work without telling it which to expect."""

    def __init__(self, L: dict):
        cfg = L.get("stt", {})
        self.key = load_key(cfg.get("provider", "groq"))
        self.url = cfg.get("base_url", "https://api.groq.com/openai/v1").rstrip("/") + "/audio/transcriptions"
        self.model = cfg.get("model", "whisper-large-v3")
        self.prompt = cfg.get("prompt", "")
        self.timeout_s = float(cfg.get("timeout_s", 8.0))
        self.last_error = None

    @property
    def available(self) -> bool:
        return bool(self.key)

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> dict | None:
        """-> {"text", "lang" ('en'/'sw'/other)} or None on failure."""
        if not self.key:
            self.last_error = "no API key"
            return None
        data = {"model": self.model, "response_format": "verbose_json", "temperature": "0"}
        if self.prompt:
            data["prompt"] = self.prompt
        try:
            r = requests.post(self.url, headers={"Authorization": f"Bearer {self.key}"}, data=data,
                              files={"file": ("utt.wav", to_wav(audio, sr), "audio/wav")}, timeout=self.timeout_s)
            if r.status_code != 200:
                self.last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                log.warning("transcription failed: %s", self.last_error)
                return None
            j = r.json()
            lang = str(j.get("language", "")).lower()
            return {"text": (j.get("text") or "").strip(), "lang": WHISPER_LANG.get(lang, lang or "en")}
        except (requests.RequestException, ValueError) as e:
            self.last_error = str(e)
            log.warning("transcription failed: %s", e)
            return None
