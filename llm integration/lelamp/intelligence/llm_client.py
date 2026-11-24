"""One non-streaming Chat Completions request per interaction."""

import math
import os

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .prompts import system_prompt


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, *, client=None, model=None, temperature=0.3, max_tokens=512,
                 timeout=15, response_format="text", token_parameter="max_tokens"):
        if not model or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Set LLM_MODEL and a positive finite LLM_TIMEOUT")
        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
        if max_tokens <= 0 or response_format not in ("text", "json_object"):
            raise ValueError("Invalid LLM_MAX_TOKENS or LLM_RESPONSE_FORMAT")
        if token_parameter not in ("max_tokens", "max_completion_tokens", ""):
            raise ValueError("Invalid LLM_TOKEN_PARAMETER")
        self.client = client
        self.model = model
        self.options = {}
        if temperature is not None:
            self.options["temperature"] = temperature
        if token_parameter:
            self.options[token_parameter] = max_tokens
        if response_format == "json_object":
            self.options["response_format"] = {"type": response_format}
        if self.client is None:
            url = os.getenv("LLM_BASE_URL", "").strip()
            key = os.getenv("LLM_API_KEY", "").strip()
            if not url or not key:
                raise ValueError("Set LLM_BASE_URL and LLM_API_KEY")
            self.client = OpenAI(base_url=url, api_key=key, timeout=timeout, max_retries=0)

    @classmethod
    def from_env(cls):
        from ..ai_config import env_flag
        if not env_flag("LLM_ENABLED", True):
            raise ValueError("LLM_ENABLED is false")
        temperature = os.getenv("LLM_TEMPERATURE", "0.3").strip()
        try:
            temperature = float(temperature) if temperature else None
            max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))
            timeout = float(os.getenv("LLM_TIMEOUT", "15"))
        except ValueError:
            raise ValueError("Check numeric LLM_TEMPERATURE, LLM_MAX_TOKENS and LLM_TIMEOUT settings") from None
        return cls(model=os.getenv("LLM_MODEL", "").strip(), temperature=temperature,
                   max_tokens=max_tokens, timeout=timeout,
                   response_format=os.getenv("LLM_RESPONSE_FORMAT", "text").strip(),
                   token_parameter=os.getenv("LLM_TOKEN_PARAMETER", "max_tokens").strip())

    def decide(self, text, capabilities):
        messages = [{"role": "system", "content": system_prompt(capabilities)},
                    {"role": "user", "content": text}]
        try:
            result = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=False, **self.options)
        except APITimeoutError:
            raise LLMError("LLM request timed out") from None
        except APIConnectionError:
            raise LLMError("Could not connect to the LLM service") from None
        except APIStatusError as exc:
            details = {401: "check LLM_API_KEY", 403: "check model access",
                       429: "rate limit or quota reached", 400: "check model and request options"}
            reason = details.get(exc.status_code, "service rejected the request")
            raise LLMError(f"LLM HTTP {exc.status_code}: {reason}") from None
        except Exception:
            raise LLMError("LLM request failed") from None
        try:
            choice = result.choices[0]
            content = choice.message.content
        except (AttributeError, IndexError, TypeError):
            raise LLMError("LLM returned no completion") from None
        if getattr(choice, "finish_reason", None) not in (None, "stop") or getattr(choice.message, "refusal", None):
            raise LLMError("LLM completion was incomplete or refused")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM returned empty content")
        return content
