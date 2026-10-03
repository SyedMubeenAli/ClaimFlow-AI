"""
The AI provider layer. Agents talk to `LLMProvider.complete()` and never to a
vendor SDK, so switching providers is a change to LLM_PROVIDER, not to code.

Providers
    GeminiProvider        Google Gemini over REST (text and images)
    OpenAICompatProvider  Groq and OpenRouter, which share the OpenAI chat format
    FakeProvider          offline and scripted, for tests and first run
    UnavailableProvider   stands in when a key is missing; fails only when called

Wrappers
    CachedProvider        saves every answer on disk and serves an identical request again.

Demo Mode does not use a provider at all. See demo_model.py for the scripted offline model.

Only the `requests` library is needed, and Streamlit already depends on it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from .config import KEY_VARS, Settings
from .errors import (
    RETRYABLE_LLM_ERRORS, ConfigError, LLMAuthError, LLMCacheMiss, LLMError,
    LLMRateLimitError, LLMResponseError, LLMUnavailableError,
)
from .logging_setup import get_logger

log = get_logger("llm")


# ---------------------------------------------------------------- request and response

@dataclass(frozen=True)
class ImageInput:
    data: bytes
    mime: str


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    system: str = ""
    json_mode: bool = False
    temperature: float = 0.0
    max_tokens: int = 4096
    images: tuple[ImageInput, ...] = ()

    def cache_key(self, provider: str, model: str) -> str:
        payload = {
            "provider": provider, "model": model, "system": self.system, "prompt": self.prompt,
            "json": self.json_mode, "temperature": self.temperature, "max_tokens": self.max_tokens,
            "images": [hashlib.sha256(i.data).hexdigest() + i.mime for i in self.images],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    cached: bool = False
    attempts: int = 1
    latency_ms: int = 0


# ---------------------------------------------------------------- provider base

class LLMProvider(ABC):
    name: str = "base"
    model: str = ""
    vision_model: str = ""

    @property
    def supports_vision(self) -> bool:
        return bool(self.vision_model)

    def model_for(self, request: LLMRequest) -> str:
        return self.vision_model if request.images else self.model

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        ...


def _system_with_json_hint(request: LLMRequest) -> str:
    """Some providers refuse JSON mode unless the prompt itself mentions JSON."""
    if request.json_mode and "json" not in (request.system + request.prompt).lower():
        return (request.system + "\nRespond with a single JSON object and nothing else.").strip()
    return request.system


# ---------------------------------------------------------------- HTTP providers

class _HttpProvider(LLMProvider):
    """Shared HTTP handling: error mapping and one retry on transient failures."""

    def __init__(self, name: str, model: str, vision_model: str, api_key: str, *,
                 timeout_s: int = 60, retry_delay_s: float = 1.5, retries: int = 1,
                 session: Any = None, sleep: Callable[[float], None] = time.sleep):
        self.name = name
        self.model = model
        self.vision_model = vision_model
        self._api_key = api_key
        self.timeout_s = timeout_s
        self.retry_delay_s = retry_delay_s
        self.retries = retries
        self.session = session or requests.Session()
        self._sleep = sleep

    def __repr__(self) -> str:  # never print the key
        return f"<{type(self).__name__} {self.name}:{self.model}>"

    def _post_once(self, url: str, headers: dict, payload: dict) -> dict:
        try:
            response = self.session.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise LLMUnavailableError(f"Could not reach {self.name}.",
                                      hint="Check the internet connection and try again.") from exc

        status = response.status_code
        if status == 200:
            try:
                return response.json()
            except ValueError as exc:
                raise LLMResponseError(f"{self.name} sent a reply that is not valid JSON.") from exc
        detail = self._error_detail(response)
        if status in (401, 403):
            raise LLMAuthError(f"{self.name} rejected the API key.",
                               hint=f"Check {KEY_VARS.get(self.name, 'the API key')} in .env.")
        if status == 404:
            raise LLMResponseError(f"{self.name} does not know the model '{self.model}'.",
                                   hint="Set LLM_MODEL to a model name from the provider's current model list.")
        if status == 429:
            raise LLMRateLimitError(f"{self.name} is rate limiting requests.",
                                    hint="Wait a minute, or turn on DEMO_MODE to replay recorded answers.")
        if status >= 500:
            raise LLMUnavailableError(f"{self.name} had a server error ({status}).")
        raise LLMResponseError(f"{self.name} rejected the request ({status}): {detail}")

    @staticmethod
    def _error_detail(response: Any) -> str:
        try:
            body = response.json()
            message = body["error"]["message"] if isinstance(body.get("error"), dict) else body.get("error")
            return str(message or body)[:300]
        except Exception:
            return str(getattr(response, "text", ""))[:300]

    def _post(self, url: str, headers: dict, payload: dict) -> tuple[dict, int]:
        """POST with retries. Returns the JSON body and the number of attempts used."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._post_once(url, headers, payload), attempt
            except RETRYABLE_LLM_ERRORS as exc:
                if attempt > self.retries:
                    raise
                log.warning("%s: %s Retrying in %.1fs.", self.name, exc.message, self.retry_delay_s)
                self._sleep(self.retry_delay_s)

    def _check_model(self, request: LLMRequest) -> str:
        model = self.model_for(request)
        if not model:
            raise LLMResponseError(f"{self.name} has no vision model configured.",
                                   hint="Set VISION_MODEL to a model that accepts images.")
        return model


class GeminiProvider(_HttpProvider):
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model: str, vision_model: str, api_key: str, **kwargs: Any):
        super().__init__("gemini", model, vision_model, api_key, **kwargs)

    def complete(self, request: LLMRequest) -> LLMResponse:
        model = self._check_model(request)
        parts: list[dict] = [{"text": request.prompt}]
        for image in request.images:
            parts.append({"inlineData": {"mimeType": image.mime,
                                         "data": base64.b64encode(image.data).decode("ascii")}})
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": request.temperature, "maxOutputTokens": request.max_tokens},
        }
        system = _system_with_json_hint(request)
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if request.json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        started = time.perf_counter()
        data, attempts = self._post(
            f"{self.BASE_URL}/models/{model}:generateContent",
            {"x-goog-api-key": self._api_key, "Content-Type": "application/json"},
            body,
        )
        candidates = data.get("candidates") or []
        if not candidates:
            blocked = (data.get("promptFeedback") or {}).get("blockReason")
            raise LLMResponseError("Gemini returned no answer" + (f" (blocked: {blocked})." if blocked else "."))
        candidate = candidates[0]
        text = "".join(p.get("text", "") for p in (candidate.get("content") or {}).get("parts") or []
                       if isinstance(p, dict))
        if not text.strip():
            raise LLMResponseError("Gemini returned an empty answer "
                                   f"(finish reason: {candidate.get('finishReason', 'unknown')}).")
        return LLMResponse(text, self.name, model, attempts=attempts,
                           latency_ms=int((time.perf_counter() - started) * 1000))


class OpenAICompatProvider(_HttpProvider):
    """Any server that speaks the OpenAI chat-completions format (Groq, OpenRouter)."""

    def __init__(self, name: str, base_url: str, model: str, vision_model: str, api_key: str, **kwargs: Any):
        super().__init__(name, model, vision_model, api_key, **kwargs)
        self.base_url = base_url.rstrip("/")

    def complete(self, request: LLMRequest) -> LLMResponse:
        model = self._check_model(request)
        if request.images:
            content: Any = [{"type": "text", "text": request.prompt}]
            for image in request.images:
                uri = f"data:{image.mime};base64,{base64.b64encode(image.data).decode('ascii')}"
                content.append({"type": "image_url", "image_url": {"url": uri}})
        else:
            content = request.prompt

        messages = []
        system = _system_with_json_hint(request)
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        body: dict[str, Any] = {"model": model, "messages": messages,
                                "temperature": request.temperature, "max_tokens": request.max_tokens}
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        data, attempts = self._post(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            body,
        )
        choices = data.get("choices") or []
        text = ""
        if choices:
            text = ((choices[0].get("message") or {}).get("content")) or ""
        if not text.strip():
            reason = choices[0].get("finish_reason", "unknown") if choices else "no choices"
            raise LLMResponseError(f"{self.name} returned an empty answer (finish reason: {reason}).")
        return LLMResponse(text, self.name, model, attempts=attempts,
                           latency_ms=int((time.perf_counter() - started) * 1000))


# ---------------------------------------------------------------- offline providers

class FakeProvider(LLMProvider):
    """
    Offline provider. With no script it echoes the start of the prompt. With a
    script ({text the prompt contains: reply}) it returns that reply, which lets
    tests control exactly what an agent sees. Every request is kept in `calls`.
    """
    name = "fake"

    def __init__(self, script: dict[str, Any] | None = None, *, default: str | None = None,
                 model: str = "fake-model", vision: bool = True, fail_with: Exception | None = None):
        self.script = list((script or {}).items())
        self.default = default
        self.model = model
        self.vision_model = model if vision else ""
        self.fail_with = fail_with
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self.fail_with is not None:
            raise self.fail_with
        for needle, reply in self.script:
            if needle in request.prompt:
                text = reply(request) if callable(reply) else reply
                break
        else:
            if self.default is not None:
                text = self.default
            elif request.json_mode:
                text = json.dumps({"echo": request.prompt[:80]})
            else:
                text = f"[fake reply] {request.prompt[:80]}"
        return LLMResponse(text, self.name, self.model_for(request))


class UnavailableProvider(LLMProvider):
    """Used when a key is missing. The app still starts; the first call explains the problem."""

    def __init__(self, name: str, model: str, vision_model: str, reason: str):
        self.name = name
        self.model = model
        self.vision_model = vision_model
        self.reason = reason

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise ConfigError(self.reason)


# ---------------------------------------------------------------- cache wrapper

class CachedProvider(LLMProvider):
    """
    Saves each answer to <cache_dir>/<hash>.json and serves it again for the same
    request. The file stores the answer and which model gave it, not the prompt.
    With replay_only=True (demo mode) the wrapped provider is never called.
    """

    def __init__(self, inner: LLMProvider, cache_dir: Path, *, replay_only: bool = False):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.replay_only = replay_only
        self.name = inner.name
        self.model = inner.model
        self.vision_model = inner.vision_model

    def complete(self, request: LLMRequest) -> LLMResponse:
        model = self.inner.model_for(request)
        path = self.cache_dir / f"{request.cache_key(self.name, model)}.json"

        entry = self._read(path)
        if entry is not None:
            return LLMResponse(entry["text"], self.name, entry.get("model", model), cached=True, attempts=0)

        if self.replay_only:
            raise LLMCacheMiss("Demo mode has no recorded answer for this request.",
                               hint="Run this claim once with DEMO_MODE=false and a real key to record it.")

        response = self.inner.complete(request)
        self._write(path, response)
        return response

    @staticmethod
    def _read(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            return entry if isinstance(entry.get("text"), str) else None
        except (OSError, ValueError):
            log.warning("Ignoring unreadable cache file %s", path.name)
            return None

    def _write(self, path: Path, response: LLMResponse) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"text": response.text, "provider": response.provider,
                                       "model": response.model}, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)  # atomic: a crash never leaves a half-written entry
        except OSError as exc:
            log.warning("Could not write the response cache: %s", exc)  # caching is optional


# ---------------------------------------------------------------- factory

GROQ_URL = "https://api.groq.com/openai/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"


def build_provider(settings: Settings, *, session: Any = None) -> LLMProvider:
    """Build the provider named in settings, wrapped in the cache when appropriate."""
    if settings.demo_mode:
        from .demo_model import DemoProvider   # imported here: demo_model builds on this module
        return DemoProvider()
    inner = _build_inner(settings, session)
    if settings.provider == "fake" or not settings.cache_enabled:
        return inner
    return CachedProvider(inner, settings.cache_dir)


def _build_inner(settings: Settings, session: Any) -> LLMProvider:
    name = settings.provider
    if name == "fake":
        return FakeProvider(model=settings.model, vision=bool(settings.vision_model))
    key = settings.api_key(name)
    if not key:
        return UnavailableProvider(
            name, settings.model, settings.vision_model,
            f"{KEY_VARS[name]} is not set. Add it to .env, or set DEMO_MODE=true to replay recorded answers.")
    common = dict(timeout_s=settings.llm_timeout_s, retry_delay_s=settings.llm_retry_delay_s, session=session)
    if name == "gemini":
        return GeminiProvider(settings.model, settings.vision_model, key, **common)
    base_url = GROQ_URL if name == "groq" else OPENROUTER_URL
    return OpenAICompatProvider(name, base_url, settings.model, settings.vision_model, key, **common)


# ---------------------------------------------------------------- reading JSON from a model

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any]:
    """
    Turn a model reply into a dict. Models sometimes wrap JSON in a code fence or
    add a sentence around it, so strip a fence, then fall back to the outermost braces.
    """
    if not isinstance(text, str):
        raise LLMResponseError("The model did not return any text.")
    cleaned = _FENCE.sub("", text.strip()).strip()
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise LLMResponseError("The model did not return a JSON object.",
                           hint="Run the step again. If it keeps happening, try another model.")
