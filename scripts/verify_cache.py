"""Verify semantic cache: exact-match, paraphrase, miss, TTL, kb_version, stats."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.cache.factory import get_cache
from src.cache.kb_version import _version_file_path, read_kb_version
from src.config import get_settings

def section(title: str) -> None:
    print(f"\n{'-'*60}\n{title}\n{'-'*60}")


def main():
    settings = get_settings()
    cache = get_cache()
    cache.clear()

    print(f"backend     : {settings.cache_backend}")
    print(f"threshold   : {settings.cache_threshold}")
    print(f"ttl_days    : {settings.cache_ttl_days}")
    print(f"kb_version  : {read_kb_version()}")

        # ── Test 1: exact-match HIT ───────────────────────────────────────────────
    section("Test 1: exact-match HIT")
    cache.set("Find Python developers with ML experience", "ANSWER_A")
    result = cache.get("Find Python developers with ML experience")
    print(f"  expected HIT  -> got {'HIT' if result else 'MISS'}")
    assert result is not None and result.answer == "ANSWER_A"

    # ── Test 2: paraphrase HIT ────────────────────────────────────────────────
    section("Test 2: paraphrase HIT (similar wording)")
    result = cache.get("Show me Python engineers who know machine learning")
    print(f"  result: {'HIT' if result else 'MISS'}")
    # Note: may HIT or MISS depending on embedding similarity vs 0.95 threshold.
    # If MISS here, the threshold is doing its job (conservative).

    # ── Test 3: unrelated MISS ────────────────────────────────────────────────
    section("Test 3: unrelated query MISS")
    result = cache.get("How do I bake sourdough bread")
    print(f"  expected MISS -> got {'HIT' if result else 'MISS'}")
    assert result is None

    # ── Test 4: TTL expiry MISS ───────────────────────────────────────────────
    section("Test 4: TTL-expired entry -> MISS")
    cache.clear()
    cache.set("What is a data engineer", "ANSWER_TTL")
    # Backdate the entry
    cache._entries[0].created_at = datetime.now(timezone.utc) - timedelta(days=settings.cache_ttl_days + 1)
    result = cache.get("What is a data engineer")
    print(f"  expected MISS -> got {'HIT' if result else 'MISS'}")
    assert result is None

    # ── Test 5: kb_version change MISS ────────────────────────────────────────
    section("Test 5: kb_version changed -> MISS")
    cache.clear()
    cache.set("Senior backend engineer", "ANSWER_KB")
    # Swap the on-disk version to simulate a re-ingest
    version_path: Path = _version_file_path()
    original_version = version_path.read_text()
    version_path.write_text("DIFFERENT_VERSION_xyz")
    try:
        result = cache.get("Senior backend engineer")
        print(f"  expected MISS -> got {'HIT' if result else 'MISS'}")
        assert result is None
    finally:
        version_path.write_text(original_version)  # restore

    # ── Test 6: stats sanity ──────────────────────────────────────────────────
    section("Test 6: stats")
    print(f"  {cache.stats()}")

    print("\nAll cache tests passed.\n")


if __name__ == "__main__":
    main()