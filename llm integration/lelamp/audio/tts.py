"""Optional synthesis and cancellable half-duplex playback."""

import io
import logging
import math
import os
import re
import time
import wave
from contextlib import closing
from http.client import HTTPSConnection
from urllib.parse import urlsplit


from ..ai_config import env_flag

logger = logging.getLogger(__name__)
SAMPLE_RATE = 24000
MAX_WAV_BYTES = 5 * 1024 * 1024


class MockTTS:
    mode = "simulated"

    def speak(self, text, cancel, interaction_id):
        return not cancel.is_set()


class SpeakerTTS:
    mode = "speaker"

    def speak(self, text, cancel, interaction_id):
        if cancel.is_set():
            return False
        started = time.monotonic()
        audio = self._synthesize(text, cancel)
        logger.info("interaction=%s tts_synthesis_seconds=%.3f", interaction_id,
                    time.monotonic() - started)
        if cancel.is_set():
            return False
        if not audio or len(audio) % 2:
            raise ValueError("TTS returned invalid PCM audio")

        import sounddevice as sd
        started = time.monotonic()
        try:
            with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                    device=self.device, blocksize=480, latency="low") as stream:
                for start in range(0, len(audio), 960):
                    if cancel.is_set():
                        stream.abort()
                        return False
                    stream.write(audio[start:start + 960])
                if cancel.is_set():
                    stream.abort()
                    return False
            return True
        finally:
            logger.info("interaction=%s tts_playback_seconds=%.3f", interaction_id,
                        time.monotonic() - started)


class OpenAITTS(SpeakerTTS):
    """OpenAI-compatible speech synthesis with signed 24 kHz mono PCM output."""

    def __init__(self, client, model, voice, device=None):
        self.client, self.model, self.voice, self.device = client, model, voice, device

    def _synthesize(self, text, cancel):
        response = self.client.audio.speech.create(
            model=self.model, voice=self.voice, input=text, response_format="pcm")
        return response.content


def _download_dashscope_wav(url, timeout):
    """Fetch Alibaba's signed result over TLS without logging its query or sending the API key."""
    target = urlsplit(url)
    host = target.hostname
    if (target.scheme not in ("http", "https") or not host or not host.endswith(".aliyuncs.com")
            or target.username is not None or target.port not in (None, 443) or not target.path):
        raise ValueError("DashScope returned an invalid audio URL")
    path = target.path + ("?" + target.query if target.query else "")
    with closing(HTTPSConnection(host, timeout=timeout)) as connection:
        connection.request("GET", path)
        response = connection.getresponse()
        if response.status != 200:
            logger.warning("DashScope TTS audio download HTTP %s", response.status)
            raise RuntimeError("DashScope TTS audio download failed")
        wav = response.read(MAX_WAV_BYTES + 1)
    if len(wav) > MAX_WAV_BYTES:
        raise ValueError("DashScope audio file exceeds the playback size limit")
    return wav


class DashScopeTTS(SpeakerTTS):
    """DashScope's non-streaming Qwen-Audio-TTS HTTP interface."""

    def __init__(self, client, endpoint, key, model, voice, timeout, device=None,
                 download=_download_dashscope_wav):
        self.client, self.endpoint, self.key = client, endpoint, key
        self.model, self.voice, self.timeout, self.device = model, voice, timeout, device
        self.download = download

    def _synthesize(self, text, cancel):
        response = self.client.post(
            self.endpoint, headers={"Authorization": f"Bearer {self.key}"}, timeout=self.timeout,
            json={"model": self.model, "input": {"text": text, "voice": self.voice,
                                                  "format": "wav", "sample_rate": SAMPLE_RATE}},
        )
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not response.is_success or not isinstance(payload, dict) or payload.get("code"):
            tags = [f"HTTP {response.status_code}"]
            if isinstance(payload, dict):
                for field in ("code", "request_id"):
                    value = payload.get(field)
                    if (isinstance(value, str) and value != self.key
                            and re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", value)):
                        tags.append(f"{field}={value}")
            logger.warning("DashScope TTS synthesis failed (%s)", ", ".join(tags))
            raise RuntimeError("DashScope TTS synthesis failed")
        if cancel.is_set():
            return b""
        try:
            url = payload["output"]["audio"]["url"]
        except (KeyError, TypeError):
            raise ValueError("DashScope TTS returned no audio URL") from None
        if not isinstance(url, str) or not url:
            raise ValueError("DashScope TTS returned no audio URL")
        with wave.open(io.BytesIO(self.download(url, self.timeout)), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, SAMPLE_RATE):
                raise ValueError("DashScope returned audio outside the requested 24 kHz mono PCM16 format")
            return wav.readframes(wav.getnframes())


def tts_from_env():
    if not env_flag("TTS_ENABLED"):
        return None
    provider = os.getenv("TTS_PROVIDER", "").strip().lower()
    if provider == "mock":
        return MockTTS()
    if provider not in ("openai", "dashscope"):
        raise ValueError("Set TTS_PROVIDER to openai, dashscope or mock")

    key, url, model, voice = (os.getenv(f"TTS_{name}", "").strip()
                              for name in ("API_KEY", "BASE_URL", "MODEL", "VOICE"))
    if not all((key, url, model, voice)):
        raise ValueError("Set TTS_API_KEY, TTS_BASE_URL, TTS_MODEL and TTS_VOICE")
    device = os.getenv("TTS_DEVICE", "").strip()
    timeout = float(os.getenv("TTS_TIMEOUT", "15"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("TTS_TIMEOUT must be positive and finite")
    device = int(device) if device else None
    if provider == "dashscope":
        import httpx

        base = url.rstrip("/")
        parts = urlsplit(base)
        if parts.scheme != "https" or not parts.hostname or parts.path != "/api/v1" or parts.query or parts.fragment:
            raise ValueError("DashScope TTS_BASE_URL must be an HTTPS DashScope base ending in /api/v1")
        return DashScopeTTS(httpx, base + "/services/audio/tts/SpeechSynthesizer",
                            key, model, voice, timeout, device)
    from openai import OpenAI

    return OpenAITTS(OpenAI(api_key=key, base_url=url, timeout=timeout, max_retries=0),
                     model, voice, device)
