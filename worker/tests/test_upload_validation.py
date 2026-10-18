"""Upload filename boundary regression tests; no database or LLM needed."""
import pytest

from tigerchain_app.ingestion.upload_validation import validate_upload_filename


@pytest.mark.parametrize("filename", ["paper.pdf", "notes 2026.md", "a-b_01.txt"])
def test_accept_safe_upload_names(filename):
    assert validate_upload_filename(filename) == filename


@pytest.mark.parametrize("filename", [None, "", ".", "..", "../escape", "/tmp/evil.pdf", "nested/a.pdf", "nested\\\\evil.pdf", "a\\x00b", "a\\nb", "a" * 256])
def test_reject_unsafe_upload_names(filename):
    with pytest.raises(ValueError, match="Unsafe upload filename"):
        validate_upload_filename(filename)
