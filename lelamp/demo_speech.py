"""Short English recordings -> cloud transcription -> explicit action allowlist."""

import base64
import io
import os
import re
import wave

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
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(audio.tobytes())
    return output.getvalue()


class ASRAuthenticationError(RuntimeError):
    """Cloud rejected the credentials; contains no server response or secret."""


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
        from openai import AuthenticationError
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
        text = result.choices[0].message.content
        if not isinstance(text, str):
            raise ValueError("ASR returned no transcription text")
        return text.strip()
