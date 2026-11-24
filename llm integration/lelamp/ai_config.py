import os


def env_flag(name, default=False):
    setting = os.getenv(name)
    if setting is None:
        return default
    value = setting.strip().lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"{name} must be true or false")


def redact_secrets(text):
    for name in ("LLM_API_KEY", "DASHSCOPE_API_KEY", "TTS_API_KEY", "OPENAI_API_KEY",
                 "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        secret = os.getenv(name, "").strip()
        if secret:
            text = text.replace(secret, "[redacted]")
    return text
