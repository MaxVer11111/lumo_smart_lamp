"""Short English recordings -> cloud transcription -> explicit action allowlist."""

import base64
import io
import os
import re
import wave

from .ai_config import redact_secrets
from .demo_motion import ACTIONS
from .demo_recordings import VOICE_RECORDINGS

COMMANDS = {
    "nod": "nod", "nod your head": "nod",
    "headshake": "headshake", "head shake": "headshake", "shake your head": "headshake",
    "hello": "greet", "say hello": "greet",
    "curious": "curious", "be curious": "curious",
    "excited": "excited", "be excited": "excited",
    "happy": "happy_wiggle", "happy wiggle": "happy_wiggle", "be happy": "happy_wiggle",
    "sad": "sad", "be sad": "sad",
    "scan": "scanning", "scanning": "scanning", "look around": "scanning",
    "shock": "shock", "shocked": "shock", "be shocked": "shock",
    "shy": "shy", "be shy": "shy",
    "wake up": "wake_up",
    "idle": "idle", "be idle": "idle",
}


def voice_commands(source="recording"):
    if source not in ("recording", "generated"):
        raise ValueError(f"Unknown motion source: {source}")
    allowed = VOICE_RECORDINGS if source == "recording" else ACTIONS
    return {phrase: action for phrase, action in COMMANDS.items() if action in allowed}


def match_command(text, source="recording"):
    text = re.sub(r"\s+", " ", text.strip().lower())
    return voice_commands(source).get(text.rstrip(".!?"))


def record_audio(device=None, duration=4.0):
    import sounddevice as sd
    if not 0 < duration <= 30:
        raise ValueError("Recording duration must be between 0 and 30 seconds")
    audio = sd.rec(round(duration * 16000), samplerate=16000, channels=1,
                   dtype="int16", device=device)
    try:
        sd.wait()
    except BaseException:
        sd.stop()
        raise
    from .audio.capture import pcm_wav

    return pcm_wav(audio.tobytes())


class ASRServiceError(RuntimeError):
    """A cloud ASR error safe to show in either demo."""


class ASRAuthenticationError(ASRServiceError):
    """Cloud rejected the credentials."""


def _asr_http_error(exc, audio_data):
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error", body)
    if not isinstance(error, dict):
        error = {}
    code = error.get("code") or body.get("code")
    request_id = (exc.request_id or exc.response.headers.get("x-dashscope-request-id")
                  or error.get("request_id") or body.get("request_id") or body.get("requestId"))
    if not request_id:
        # The OpenAI SDK unwraps the "error" object and can drop a sibling request ID.
        try:
            envelope = exc.response.json()
        except ValueError:
            envelope = None
        if isinstance(envelope, dict):
            request_id = envelope.get("request_id") or envelope.get("requestId")
    tags = []
    for name, value in (("code", code), ("request_id", request_id)):
        if (isinstance(value, str) and not value.lower().startswith("sk-")
                and re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", value)):
            tags.append(f"{name}={redact_secrets(value)}")

    # The provider can echo input or a URL in its message. Display just a short,
    # single-line diagnostic with configured secrets, audio and URLs removed.
    message = error.get("message")
    if isinstance(message, str):
        message = redact_secrets(message.replace(audio_data, "[audio]")
                                 .replace(audio_data.partition(",")[2], "[audio]"))
        message = re.sub(r"(?i)https?://[^\s]+|data:audio/[^\s]+", "[redacted]", message)
        message = re.sub(r"(?i)bearer\s+[^\s]+|sk-[A-Za-z0-9_-]+", "[redacted]", message)
        message = re.sub(r"[A-Za-z0-9+/=_-]{80,}", "[redacted]", message)
        message = " ".join("".join(char if char.isprintable() else " " for char in message).split())
        if message:
            tags.append(message[:280])

    hints = {
        400: "Check the ASR model and request reported by Bailian.",
        403: "Check that the key can access the ASR model in this workspace.",
        404: "Check ASR_MODEL and ASR_BASE_URL against the model's API guide in the console.",
        429: "Check ASR quota and rate limits before retrying.",
    }
    detail = f"Bailian ASR HTTP {exc.status_code}"
    if tags:
        detail += " | " + " | ".join(tags)
    hint = hints.get(exc.status_code)
    if exc.status_code == 400 and isinstance(message, str) and "workspace endpoint is invalid" in message.lower():
        hint = ("Check ASR_BASE_URL. Use the API Host from the workspace that issued the key, "
                "or the shared DashScope base URL for the key's region.")
    return ASRServiceError(f"{detail} {hint}" if hint else detail)


class BailianASR:
    def __init__(self, client=None, model=None):
        self.model = model or os.getenv("ASR_MODEL", "qwen3-asr-flash")
        if client is None:
            from openai import OpenAI
            key = os.getenv("DASHSCOPE_API_KEY", "").strip()
            url = os.getenv("ASR_BASE_URL", "").strip()
            if not key or not url:
                raise ValueError("Set DASHSCOPE_API_KEY and ASR_BASE_URL for the same region")
            client = OpenAI(api_key=key, base_url=url, timeout=20, max_retries=0)
        self.client = client

    def transcribe(self, audio):
        # Only accept the same uncompressed format produced by our recorder.
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise ValueError("Expected 16 kHz mono PCM16 WAV")
            if not 0 < wav.getnframes() <= 30 * 16000:
                raise ValueError("Expected 0–30 seconds of audio")
        data = "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")
        from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError
        try:
            result = self.client.chat.completions.create(
                model=self.model, stream=False,
                messages=[{"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {"data": data}}
                ]}],
                extra_body={"asr_options": {"language": "en", "enable_itn": False}},
            )
        except AuthenticationError:
            raise ASRAuthenticationError(
                "Bailian returned HTTP 401. Check DASHSCOPE_API_KEY is an active Model Studio API Key, "
                "and ASR_BASE_URL matches its region and workspace. Coding Plan/Token Plan keys "
                "cannot be mixed with general model endpoints. Shell environment variables override .env; "
                "after updating credentials, restart the program. No action was triggered."
            ) from None
        except APITimeoutError:
            raise ASRServiceError("Bailian ASR request timed out") from None
        except APIConnectionError:
            raise ASRServiceError("Could not connect to Bailian ASR") from None
        except APIStatusError as exc:
            raise _asr_http_error(exc, data) from None
        text = result.choices[0].message.content
        if not isinstance(text, str):
            raise ValueError("ASR returned no transcription text")
        return text.strip()
