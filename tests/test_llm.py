import tempfile
from pathlib import Path

import requests

from claimflow.config import Settings
from claimflow.errors import (
    ConfigError, LLMAuthError, LLMCacheMiss, LLMRateLimitError, LLMResponseError, LLMUnavailableError,
)
from claimflow.llm import (
    CachedProvider, FakeProvider, GeminiProvider, ImageInput, LLMRequest, OpenAICompatProvider,
    UnavailableProvider, build_provider, parse_json_object,
)


class FakeResponse:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    """Stands in for requests.Session: hands out queued responses and records the calls."""
    def __init__(self, *queue):
        self.queue, self.calls = list(queue), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def gemini_ok(text="hello"):
    return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": text}]}}]})


def raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"{exc_type.__name__} was not raised")


def gemini(session, sleeps=None):
    return GeminiProvider("text-model", "vision-model", "SECRET-KEY-123", session=session,
                          sleep=(sleeps.append if sleeps is not None else (lambda s: None)))


# ---------------------------------------------------------------- Gemini

def test_gemini_request_shape():
    session = FakeSession(gemini_ok("hi"))
    response = gemini(session).complete(LLMRequest(prompt="Describe", system="Be brief", json_mode=True, max_tokens=50))
    call = session.calls[0]
    assert call["url"].endswith("/models/text-model:generateContent")
    assert call["headers"]["x-goog-api-key"] == "SECRET-KEY-123" and "SECRET-KEY-123" not in call["url"]
    body = call["json"]
    system_text = body["systemInstruction"]["parts"][0]["text"]
    assert system_text.startswith("Be brief") and "JSON" in system_text   # hint added: prompt did not mention JSON
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["maxOutputTokens"] == 50
    assert response.text == "hi" and response.model == "text-model" and response.attempts == 1


def test_the_json_hint_is_not_added_when_the_prompt_already_mentions_json():
    session = FakeSession(gemini_ok("{}"))
    gemini(session).complete(LLMRequest(prompt="Return JSON with a name field", system="Be brief", json_mode=True))
    assert session.calls[0]["json"]["systemInstruction"]["parts"][0]["text"] == "Be brief"


def test_gemini_sends_images_with_the_vision_model():
    session = FakeSession(gemini_ok())
    gemini(session).complete(LLMRequest(prompt="What is damaged?", images=(ImageInput(b"\x89PNGdata", "image/png"),)))
    call = session.calls[0]
    assert "/models/vision-model:" in call["url"]
    part = call["json"]["contents"][0]["parts"][1]["inlineData"]
    assert part["mimeType"] == "image/png" and part["data"]


def test_retries_once_on_rate_limit_then_succeeds():
    sleeps = []
    session = FakeSession(FakeResponse(429, {"error": {"message": "slow down"}}), gemini_ok("done"))
    response = gemini(session, sleeps).complete(LLMRequest(prompt="x"))
    assert response.text == "done" and response.attempts == 2 and len(session.calls) == 2 and len(sleeps) == 1


def test_gives_up_after_one_retry():
    session = FakeSession(FakeResponse(429), FakeResponse(429), gemini_ok())
    raises(LLMRateLimitError, gemini(session).complete, LLMRequest(prompt="x"))
    assert len(session.calls) == 2


def test_server_errors_and_timeouts_are_retried_as_unavailable():
    session = FakeSession(FakeResponse(503), FakeResponse(500))
    raises(LLMUnavailableError, gemini(session).complete, LLMRequest(prompt="x"))
    assert len(session.calls) == 2
    session = FakeSession(requests.Timeout("slow"), requests.ConnectionError("down"))
    raises(LLMUnavailableError, gemini(session).complete, LLMRequest(prompt="x"))


def test_bad_key_is_not_retried_and_names_the_variable():
    session = FakeSession(FakeResponse(401), gemini_ok())
    error = raises(LLMAuthError, gemini(session).complete, LLMRequest(prompt="x"))
    assert len(session.calls) == 1 and "GEMINI_API_KEY" in error.user_message
    assert "SECRET-KEY-123" not in error.user_message


def test_unknown_model_error_tells_the_user_what_to_change():
    error = raises(LLMResponseError, gemini(FakeSession(FakeResponse(404))).complete, LLMRequest(prompt="x"))
    assert "text-model" in error.message and "LLM_MODEL" in error.user_message


def test_blocked_or_empty_answers_are_errors():
    blocked = FakeResponse(200, {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}})
    assert "SAFETY" in raises(LLMResponseError, gemini(FakeSession(blocked)).complete, LLMRequest(prompt="x")).message
    empty = FakeResponse(200, {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]})
    assert "MAX_TOKENS" in raises(LLMResponseError, gemini(FakeSession(empty)).complete, LLMRequest(prompt="x")).message


def test_non_json_reply_is_an_error():
    raises(LLMResponseError, gemini(FakeSession(FakeResponse(200, None))).complete, LLMRequest(prompt="x"))


def test_provider_repr_hides_the_key():
    assert "SECRET-KEY-123" not in repr(gemini(FakeSession()))


# ---------------------------------------------------------------- OpenAI-compatible

def compat(session, vision="vision-model"):
    return OpenAICompatProvider("groq", "https://example.test/v1/", "text-model", vision, "KEY-XYZ-999",
                                session=session, sleep=lambda s: None)


def test_openai_compatible_request_and_reply():
    reply = FakeResponse(200, {"choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}]})
    session = FakeSession(reply)
    response = compat(session).complete(LLMRequest(prompt="Give json", system="sys", json_mode=True))
    call = session.calls[0]
    assert call["url"] == "https://example.test/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer KEY-XYZ-999"
    assert call["json"]["response_format"] == {"type": "json_object"}
    assert call["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert response.text == '{"a": 1}'


def test_openai_compatible_images_use_data_uris():
    reply = FakeResponse(200, {"choices": [{"message": {"content": "a dent"}}]})
    session = FakeSession(reply)
    compat(session).complete(LLMRequest(prompt="Look", images=(ImageInput(b"abc", "image/jpeg"),)))
    content = session.calls[0]["json"]["messages"][-1]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert session.calls[0]["json"]["model"] == "vision-model"


def test_openai_compatible_without_a_vision_model_refuses_images_clearly():
    error = raises(LLMResponseError, compat(FakeSession(), vision="").complete,
                   LLMRequest(prompt="Look", images=(ImageInput(b"abc", "image/png"),)))
    assert "VISION_MODEL" in error.user_message


def test_json_mode_adds_a_json_hint_when_the_prompt_lacks_one():
    session = FakeSession(FakeResponse(200, {"choices": [{"message": {"content": "{}"}}]}))
    compat(session).complete(LLMRequest(prompt="List the items", json_mode=True))
    assert "JSON" in session.calls[0]["json"]["messages"][0]["content"]


# ---------------------------------------------------------------- fake provider

def test_fake_provider_scripts_and_records_calls():
    fake = FakeProvider({"policy": "POLICY REPLY", "photo": lambda r: "PHOTO REPLY"})
    assert fake.complete(LLMRequest(prompt="read the policy")).text == "POLICY REPLY"
    assert fake.complete(LLMRequest(prompt="look at the photo")).text == "PHOTO REPLY"
    assert fake.complete(LLMRequest(prompt="other", json_mode=True)).text.startswith('{"echo"')
    assert len(fake.calls) == 3
    raises(LLMUnavailableError, FakeProvider(fail_with=LLMUnavailableError("down")).complete, LLMRequest(prompt="x"))


# ---------------------------------------------------------------- cache

def test_cache_serves_repeat_requests_without_calling_the_provider():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        inner = FakeProvider(default="answer")
        cached = CachedProvider(inner, Path(tmp))
        first = cached.complete(LLMRequest(prompt="same"))
        second = cached.complete(LLMRequest(prompt="same"))
        assert first.cached is False and second.cached is True and second.text == "answer"
        assert len(inner.calls) == 1
        cached.complete(LLMRequest(prompt="different"))
        assert len(inner.calls) == 2


def test_cache_key_depends_on_model_prompt_and_images():
    base = LLMRequest(prompt="a")
    assert base.cache_key("p", "m") == LLMRequest(prompt="a").cache_key("p", "m")
    assert base.cache_key("p", "m") != base.cache_key("p", "other-model")
    assert base.cache_key("p", "m") != LLMRequest(prompt="b").cache_key("p", "m")
    assert base.cache_key("p", "m") != LLMRequest(prompt="a", images=(ImageInput(b"x", "image/png"),)).cache_key("p", "m")


def test_replay_only_mode_never_calls_the_provider():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        recorder = CachedProvider(FakeProvider(default="recorded"), Path(tmp))
        recorder.complete(LLMRequest(prompt="demo step"))
        inner = FakeProvider(default="live")
        replay = CachedProvider(inner, Path(tmp), replay_only=True)
        assert replay.complete(LLMRequest(prompt="demo step")).text == "recorded"
        error = raises(LLMCacheMiss, replay.complete, LLMRequest(prompt="never recorded"))
        assert "DEMO_MODE=false" in error.user_message and inner.calls == []


def test_unreadable_or_unwritable_cache_never_breaks_a_call():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        bad = Path(tmp) / "afile"
        bad.write_text("not a folder")
        inner = FakeProvider(default="ok")
        assert CachedProvider(inner, bad / "cache").complete(LLMRequest(prompt="x")).text == "ok"
        good = CachedProvider(inner, Path(tmp) / "c")
        good.complete(LLMRequest(prompt="y"))
        for entry in (Path(tmp) / "c").glob("*.json"):
            entry.write_text("{ broken")
        assert good.complete(LLMRequest(prompt="y")).text == "ok"


# ---------------------------------------------------------------- factory

def test_factory_picks_the_provider_from_settings():
    fake = build_provider(Settings.from_env({"DEMO_MODE": "false"}))
    assert isinstance(fake, FakeProvider)
    with_key = build_provider(Settings.from_env({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k" * 10}))
    assert isinstance(with_key, CachedProvider) and isinstance(with_key.inner, GeminiProvider)
    groq = build_provider(Settings.from_env({"LLM_PROVIDER": "groq", "GROQ_API_KEY": "k" * 10, "LLM_CACHE": "false"}))
    assert isinstance(groq, OpenAICompatProvider) and groq.base_url.endswith("groq.com/openai/v1")
    assert groq.model == "llama-3.3-70b-versatile" and groq.supports_vision is False


def test_missing_key_gives_a_provider_that_explains_itself():
    provider = build_provider(Settings.from_env({"LLM_PROVIDER": "openrouter"}))
    assert isinstance(provider.inner, UnavailableProvider)
    assert "OPENROUTER_API_KEY" in raises(ConfigError, provider.complete, LLMRequest(prompt="x")).message


def test_demo_mode_uses_the_scripted_model_and_needs_no_key_or_network():
    from claimflow.demo_model import DemoProvider
    for env in ({"LLM_PROVIDER": "gemini", "DEMO_MODE": "true"}, {}):
        provider = build_provider(Settings.from_env(env))
        assert isinstance(provider, DemoProvider) and provider.supports_vision and provider.name == "demo"


# ---------------------------------------------------------------- reading JSON

def test_parse_json_object_handles_fences_and_chatter():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go: {"a": {"b": 2}} Hope it helps') == {"a": {"b": 2}}


def test_parse_json_object_rejects_everything_else():
    for text in ("no json here", "[1, 2, 3]", '"just a string"', "{ not valid }", "", None, 42):
        raises(LLMResponseError, parse_json_object, text)
