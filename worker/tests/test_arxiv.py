from __future__ import annotations

from pathlib import Path

from tigerchain_app.ingestion.arxiv import fetch_arxiv_paper, normalise_arxiv_id


class DummyResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:  # pragma: no cover - no-op for tests
        return None


class DummyStreamResponse(DummyResponse):
    def __enter__(self) -> "DummyStreamResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - nothing to clean
        return False

    def iter_content(self, chunk_size: int):
        yield self.content


class DummySession:
    def __init__(self, responses: list[DummyResponse]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, **kwargs):
        if not self._responses:
            raise AssertionError("No more responses configured for DummySession")
        response = self._responses.pop(0)
        self.calls.append((url, kwargs))
        return response


def test_normalise_arxiv_id_variants() -> None:
    assert normalise_arxiv_id("arXiv:2301.12345") == "2301.12345"
    assert normalise_arxiv_id("https://arxiv.org/abs/2301.12345v2") == "2301.12345v2"
    assert normalise_arxiv_id("https://arxiv.org/pdf/2301.12345.pdf") == "2301.12345"
    assert normalise_arxiv_id("hep-th/9901001v1") == "hep-th/9901001v1"


def test_fetch_arxiv_paper_parses_metadata(tmp_path: Path) -> None:
    feed = b"""<?xml version='1.0' encoding='UTF-8'?>
    <feed xmlns='http://www.w3.org/2005/Atom' xmlns:arxiv='http://arxiv.org/schemas/atom'>
      <entry>
        <id>http://arxiv.org/abs/2301.12345v1</id>
        <updated>2023-01-10T12:34:56Z</updated>
        <published>2023-01-09T12:00:00Z</published>
        <title> Test Title </title>
        <summary> Summary text </summary>
        <author><name>Alice</name></author>
        <author><name>Bob</name></author>
        <link href='http://arxiv.org/pdf/2301.12345v1.pdf' type='application/pdf' rel='related'/>
        <arxiv:doi>10.1000/test</arxiv:doi>
        <arxiv:primary_category term='cs.AI'/>
        <category term='cs.AI'/>
        <category term='cs.LG'/>
      </entry>
    </feed>
    """
    pdf_bytes = b"%PDF-1.4 test"
    session = DummySession([DummyResponse(feed), DummyStreamResponse(pdf_bytes)])

    paper = fetch_arxiv_paper("https://arxiv.org/abs/2301.12345v1", session=session)
    assert paper.arxiv_id == "2301.12345v1"
    assert paper.title == "Test Title"
    assert paper.authors == ["Alice", "Bob"]
    assert paper.pdf_path.exists()
    assert paper.pdf_path.read_bytes() == pdf_bytes
    metadata = paper.to_ingestion_metadata()
    assert metadata["abs_url"].endswith("2301.12345v1")
    assert metadata["doi"] == "10.1000/test"
    assert set(metadata["categories"]) == {"cs.AI", "cs.LG"}

    paper.cleanup()
    assert not paper.pdf_path.exists()

    # ensure both metadata and PDF requests were issued
    urls = [call[0] for call in session.calls]
    assert any(url.endswith("api/query") for url in urls)
    assert any(url.endswith("2301.12345v1.pdf") for url in urls)
