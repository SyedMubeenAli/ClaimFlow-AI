import logging

from claimflow.config import ROOT, Settings
from claimflow.errors import ConfigError
from claimflow.logging_setup import RedactingFilter, setup_logging


def raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"{exc_type.__name__} was not raised")


def test_defaults_use_the_offline_provider():
    s = Settings.from_env({})
    assert s.provider == "fake" and s.model == "fake-model"
    assert s.demo_mode is True and s.cache_enabled is True      # nothing configured: Demo Mode, so the first run works
    assert s.max_upload_bytes == 10 * 1024 * 1024
    assert s.database_path.is_absolute() and ROOT in s.database_path.parents


def test_provider_name_is_case_insensitive_and_checked():
    assert Settings.from_env({"LLM_PROVIDER": " Gemini "}).provider == "gemini"
    error = raises(ConfigError, Settings.from_env, {"LLM_PROVIDER": "openai"})
    assert "gemini" in error.user_message and "openai" in error.user_message


def test_bad_values_are_rejected_with_the_variable_name():
    assert "DEMO_MODE" in raises(ConfigError, Settings.from_env, {"DEMO_MODE": "ture"}).message
    assert "MAX_UPLOAD_MB" in raises(ConfigError, Settings.from_env, {"MAX_UPLOAD_MB": "0"}).message
    assert "MAX_UPLOAD_MB" in raises(ConfigError, Settings.from_env, {"MAX_UPLOAD_MB": "lots"}).message
    assert "LOG_LEVEL" in raises(ConfigError, Settings.from_env, {"LOG_LEVEL": "LOUD"}).message


def test_booleans_accept_common_spellings():
    for text in ("1", "true", "YES", "on"):
        assert Settings.from_env({"DEMO_MODE": text}).demo_mode is True
    for text in ("0", "false", "No", "off"):
        assert Settings.from_env({"DEMO_MODE": text}).demo_mode is False


def test_keys_never_appear_in_repr_or_summary():
    s = Settings.from_env({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "sk-very-secret-123"})
    assert "sk-very-secret-123" not in repr(s)
    assert "sk-very-secret-123" not in str(s.safe_summary())
    assert s.has_api_key() and not s.has_api_key("groq")
    assert s.secrets() == ["sk-very-secret-123"]


def test_vision_model_defaults_by_provider():
    assert Settings.from_env({"LLM_PROVIDER": "gemini"}).vision_model == "gemini-2.5-flash"
    assert Settings.from_env({"LLM_PROVIDER": "groq"}).vision_model == ""
    assert Settings.from_env({"LLM_PROVIDER": "groq", "VISION_MODEL": "some-vision"}).vision_model == "some-vision"


def test_relative_paths_resolve_from_the_project_root():
    s = Settings.from_env({"DATABASE_PATH": "data/x.db"})
    assert s.database_path == (ROOT / "data/x.db").resolve()


def test_log_filter_redacts_secrets():
    record = logging.LogRecord("x", logging.INFO, "", 0, "calling with key %s", ("abc123456",), None)
    RedactingFilter(["abc123456"]).filter(record)
    assert "abc123456" not in record.getMessage() and "[REDACTED]" in record.getMessage()


def test_short_values_are_not_treated_as_secrets():
    record = logging.LogRecord("x", logging.INFO, "", 0, "the cat sat", (), None)
    RedactingFilter(["cat"]).filter(record)
    assert record.getMessage() == "the cat sat"


def test_setup_logging_does_not_stack_handlers():
    setup_logging("ERROR")
    first = len(logging.getLogger("claimflow").handlers)
    setup_logging("ERROR")
    assert len(logging.getLogger("claimflow").handlers) == first


def test_rag_settings_have_safe_defaults_and_are_validated():
    s = Settings.from_env({})
    assert s.embedding_backend == "auto" and s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.index_dir.is_absolute() and s.index_dir.name == "policy_index"
    assert Settings.from_env({"EMBEDDING_BACKEND": "Hashing"}).embedding_backend == "hashing"
    assert "EMBEDDING_BACKEND" in raises(ConfigError, Settings.from_env, {"EMBEDDING_BACKEND": "openai"}).message
    assert "Embedding backend" in s.safe_summary()


def test_demo_mode_defaults_on_only_when_no_real_provider_is_configured():
    assert Settings.from_env({"DEMO_MODE": "false"}).demo_mode is False                     # the offline test provider, explicitly live
    assert Settings.from_env({"LLM_PROVIDER": "gemini"}).demo_mode is False                 # a real provider means live AI by default
    assert Settings.from_env({"LLM_PROVIDER": "gemini", "DEMO_MODE": "true"}).demo_mode is True
    assert "scripted offline model" in Settings.from_env({}).safe_summary()["Demo mode"]
