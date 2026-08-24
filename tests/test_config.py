"""Sanity checks that Settings loads and required fields are present.

These run against whatever .env is on disk. They assert structure/typing,
not specific secret values.
"""
from pathlib import Path

from resume_rag.config import Settings, get_settings


def test_settings_singleton():
    """get_settings() is cached -> same object everytime call."""
    assert get_settings() is get_settings()

def test_required_fields_present():
    s = get_settings()
    assert s.anthropic_api_key, "ANTHROPIC_API_KEY missing"
    assert s.cohere_api_key, "COHERE_API_KEY missing"
    

def test_paths_are_path_objects():
    s = get_settings()
    assert isinstance(s.chroma_persist_dir, Path)
    assert isinstance(s.raw_dir, Path)


def test_retrieval_defaults_sane():
    s = get_settings()
    assert s.rerank_top_n <= s.retrieval_k, "can't keep more than we retrieve"


def test_missing_required_raises(monkeypatch, tmp_path):
    """With no env vars and no .env, instantiating Settings fails fast."""
    for var in ("ANTHROPIC_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    empty = tmp_path / "empty.env"
    empty.write_text("")
    try:
        Settings(_env_file=str(empty))  # type: ignore[call-arg]
    except Exception as e:
        assert "validation error" in str(e).lower() or "field required" in str(e).lower()
    else:
        raise AssertionError("expected validation error for missing required keys")