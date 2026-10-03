"""
Configuration, read once from environment variables (and a local .env file).

Secrets never appear in repr(), logs or the Settings page. On Streamlit
Community Cloud, top-level secrets are exposed as environment variables, so
the same names work there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

from .errors import ConfigError

ROOT = Path(__file__).resolve().parent.parent

PROVIDERS = ("fake", "gemini", "groq", "openrouter")
KEY_VARS = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
# Model names change often. These are starting points. If one is rejected, the
# Settings page shows a clear "model not found" message: set LLM_MODEL.
DEFAULT_MODELS = {
    "fake": "fake-model",
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
}
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
EMBEDDING_BACKENDS = ("auto", "fastembed", "hashing")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _to_bool(name: str, value: str, default: bool) -> bool:
    if value == "":
        return default
    lowered = value.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ConfigError(f"{name} must be true or false, not '{value}'.")


def _to_number(name: str, value: str, default: float, low: float, high: float) -> float:
    if value == "":
        return default
    try:
        number = float(value)
    except ValueError:
        raise ConfigError(f"{name} must be a number, not '{value}'.") from None
    if not low <= number <= high:
        raise ConfigError(f"{name} must be between {low:g} and {high:g}.")
    return number


def _to_path(value: str, default: str) -> Path:
    path = Path(value or default).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    vision_model: str
    demo_mode: bool
    cache_enabled: bool
    database_path: Path
    upload_dir: Path
    cache_dir: Path
    index_dir: Path
    embedding_backend: str
    embedding_model: str
    log_dir: Path
    log_level: str
    max_upload_mb: int
    llm_timeout_s: int
    llm_retry_delay_s: float
    gemini_api_key: str = field(default="", repr=False)
    groq_api_key: str = field(default="", repr=False)
    openrouter_api_key: str = field(default="", repr=False)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def api_key(self, provider: str | None = None) -> str:
        return {
            "gemini": self.gemini_api_key,
            "groq": self.groq_api_key,
            "openrouter": self.openrouter_api_key,
        }.get(provider or self.provider, "")

    def has_api_key(self, provider: str | None = None) -> bool:
        return bool(self.api_key(provider))

    def secrets(self) -> list[str]:
        """Values that must never be written to a log."""
        return [k for k in (self.gemini_api_key, self.groq_api_key, self.openrouter_api_key) if k]

    def safe_summary(self) -> dict[str, str]:
        """Settings for display. Never includes a key."""
        return {
            "Provider": self.provider,
            "Model": self.model,
            "Vision model": self.vision_model or "not configured",
            "Demo mode": "on (scripted offline model, no internet needed)" if self.demo_mode else "off (live AI)",
            "Response cache": "on" if self.cache_enabled else "off",
            "Database": str(self.database_path),
            "Upload folder": str(self.upload_dir),
            "Cache folder": str(self.cache_dir),
            "Policy index folder": str(self.index_dir),
            "Embedding backend": f"{self.embedding_backend} ({self.embedding_model})",
            "Max upload size": f"{self.max_upload_mb} MB",
            "Log level": self.log_level,
        }

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """
        Build settings. With no argument it loads .env (without overriding real
        environment variables) and reads os.environ. Tests pass a plain dict.
        """
        if env is None:
            load_dotenv(ROOT / ".env", override=False)
            env = os.environ

        def get(name: str, default: str = "") -> str:
            return (env.get(name) or default).strip()

        provider = get("LLM_PROVIDER", "fake").lower()
        if provider not in PROVIDERS:
            raise ConfigError(
                f"LLM_PROVIDER '{provider}' is not supported.",
                hint=f"Use one of: {', '.join(PROVIDERS)}.",
            )

        model = get("LLM_MODEL") or DEFAULT_MODELS[provider]
        # Gemini reads images with the same model. The others only get a vision
        # model if one is named explicitly.
        vision_model = get("VISION_MODEL") or (model if provider in ("gemini", "fake") else "")

        embedding_backend = get("EMBEDDING_BACKEND", "auto").lower()
        if embedding_backend not in EMBEDDING_BACKENDS:
            raise ConfigError(f"EMBEDDING_BACKEND '{embedding_backend}' is not supported.",
                              hint=f"Use one of: {', '.join(EMBEDDING_BACKENDS)}.")

        log_level = get("LOG_LEVEL", "INFO").upper()
        if log_level not in LOG_LEVELS:
            raise ConfigError(f"LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}.")

        return cls(
            provider=provider,
            model=model,
            vision_model=vision_model,
            # With nothing configured (the offline provider) there is no real model to call,
            # so Demo Mode is on by default and the first run shows a working workflow.
            demo_mode=_to_bool("DEMO_MODE", get("DEMO_MODE"), provider == "fake"),
            cache_enabled=_to_bool("LLM_CACHE", get("LLM_CACHE"), True),
            database_path=_to_path(get("DATABASE_PATH"), "data/claimflow.db"),
            upload_dir=_to_path(get("UPLOAD_DIR"), "data/uploads"),
            cache_dir=_to_path(get("LLM_CACHE_DIR"), "data/llm_cache"),
            index_dir=_to_path(get("INDEX_DIR"), "data/policy_index"),
            embedding_backend=embedding_backend,
            embedding_model=get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
            log_dir=_to_path(get("LOG_DIR"), "logs"),
            log_level=log_level,
            max_upload_mb=int(_to_number("MAX_UPLOAD_MB", get("MAX_UPLOAD_MB"), 10, 1, 50)),
            llm_timeout_s=int(_to_number("LLM_TIMEOUT_S", get("LLM_TIMEOUT_S"), 60, 5, 300)),
            llm_retry_delay_s=_to_number("LLM_RETRY_DELAY_S", get("LLM_RETRY_DELAY_S"), 1.5, 0, 30),
            gemini_api_key=get("GEMINI_API_KEY"),
            groq_api_key=get("GROQ_API_KEY"),
            openrouter_api_key=get("OPENROUTER_API_KEY"),
        )
