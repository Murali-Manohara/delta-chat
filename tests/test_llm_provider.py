import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chat.llm import ExtractiveFallbackClient, GroqClient, AnthropicClient, get_default_client


def _clear_llm_env(monkeypatch):
    for key in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)


def test_defaults_to_extractive_fallback_with_no_keys(monkeypatch):
    _clear_llm_env(monkeypatch)
    client = get_default_client()
    assert isinstance(client, ExtractiveFallbackClient)


def test_picks_groq_when_groq_key_present(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-wiring-test")
    client = get_default_client()
    assert isinstance(client, GroqClient)
    assert client.model_name == "llama-3.3-70b-versatile"


def test_groq_model_overridable_via_env(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-wiring-test")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    client = get_default_client()
    assert client.model_name == "llama-3.1-8b-instant"


def test_explicit_llm_provider_env_var_wins(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    client = get_default_client()
    assert isinstance(client, AnthropicClient)


def test_groq_client_raises_without_key(monkeypatch):
    _clear_llm_env(monkeypatch)
    try:
        GroqClient()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "GROQ_API_KEY" in str(e)
