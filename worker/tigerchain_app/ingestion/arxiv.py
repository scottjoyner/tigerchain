from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from xml.etree import ElementTree

import requests

from ..utils.logging import get_logger

logger = get_logger(__name__)

_ARXIV_ID_RE = re.compile(
    r"^(?P<identifier>(?:arxiv:)?(?:(?:\d{4}\.\d{4,5})|(?:[a-z\-]+/\d{7}))(?P<version>v\d+)?)$",
    re.IGNORECASE,
)
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", re.IGNORECASE)
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"
_API_URL = "https://export.arxiv.org/api/query"
_DEFAULT_USER_AGENT = "TigerChain-ArXivFetcher/1.0"


@dataclass
class ArxivPaper:
    """Container describing a downloaded arXiv paper and its metadata."""

    arxiv_id: str
    title: str
    summary: str
    authors: list[str]
    pdf_path: Path
    pdf_url: str
    abs_url: str
    published: Optional[datetime]
    updated: Optional[datetime]
    categories: list[str] = field(default_factory=list)
    primary_category: Optional[str] = None
    doi: Optional[str] = None
    workdir: Optional[Path] = field(default=None, repr=False)

    def to_ingestion_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "id": self.arxiv_id,
            "title": self.title,
            "summary": self.summary,
            "authors": list(self.authors),
            "pdf_url": self.pdf_url,
            "abs_url": self.abs_url,
        }
        if self.categories:
            metadata["categories"] = list(self.categories)
        if self.primary_category:
            metadata["primary_category"] = self.primary_category
        if self.published:
            metadata["published"] = self.published.isoformat()
        if self.updated:
            metadata["updated"] = self.updated.isoformat()
        if self.doi:
            metadata["doi"] = self.doi
        return metadata

    def cleanup(self) -> None:
        if self.workdir and self.workdir.exists():
            shutil.rmtree(self.workdir, ignore_errors=True)
            logger.debug("Removed temporary arXiv workdir %s", self.workdir)


def normalise_arxiv_id(value: str) -> str:
    """Extract the canonical arXiv identifier from common input formats."""

    candidate = value.strip()
    if not candidate:
        raise ValueError("arXiv identifier or URL cannot be blank")

    match = _ARXIV_URL_RE.search(candidate)
    if match:
        candidate = match.group(1)
    candidate = candidate.split("?")[0].split("#")[0]
    if candidate.lower().startswith("arxiv:"):
        candidate = candidate.split(":", 1)[1]
    if candidate.lower().endswith(".pdf"):
        candidate = candidate[:-4]

    candidate = candidate.strip()
    regex_match = _ARXIV_ID_RE.match(candidate)
    if not regex_match:
        raise ValueError(f"Invalid arXiv identifier: {value}")

    identifier = regex_match.group("identifier")
    if identifier is None:
        raise ValueError(f"Invalid arXiv identifier: {value}")
    identifier = identifier.lower()
    if identifier.startswith("arxiv:"):
        identifier = identifier.split(":", 1)[1]
    return identifier


def fetch_arxiv_paper(identifier: str, *, session: Optional[requests.Session] = None) -> ArxivPaper:
    """Fetch metadata and the PDF for the supplied arXiv identifier."""

    arxiv_id = normalise_arxiv_id(identifier)
    session = session or requests.Session()
    headers = {"User-Agent": _DEFAULT_USER_AGENT}

    logger.info("Fetching arXiv metadata for %s", arxiv_id)
    response = session.get(
        _API_URL,
        params={"search_query": f"id:{arxiv_id}", "start": 0, "max_results": 1},
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    entry = _extract_entry(response.content)
    if entry is None:
        raise ValueError(f"No arXiv entry found for identifier '{arxiv_id}'")

    title = _get_text(entry, "title")
    summary = _get_text(entry, "summary")
    authors = _extract_authors(entry)
    published = _parse_date(_get_text(entry, "published"))
    updated = _parse_date(_get_text(entry, "updated"))
    pdf_url = _extract_pdf_url(entry) or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    doi = _get_text(entry, f"{_ARXIV_NS}doi")
    primary_category = _get_attribute(entry, f"{_ARXIV_NS}primary_category", "term")
    categories = sorted({term for term in _extract_categories(entry) if term})

    workdir = Path(tempfile.mkdtemp(prefix="arxiv_"))
    filename = f"{arxiv_id.replace('/', '_')}.pdf"
    pdf_path = workdir / filename

    logger.info("Downloading arXiv PDF from %s", pdf_url)
    with session.get(pdf_url, headers=headers, stream=True, timeout=60) as pdf_response:
        pdf_response.raise_for_status()
        with pdf_path.open("wb") as buffer:
            for chunk in pdf_response.iter_content(chunk_size=65536):
                if chunk:
                    buffer.write(chunk)

    paper = ArxivPaper(
        arxiv_id=arxiv_id,
        title=title,
        summary=summary,
        authors=authors,
        pdf_path=pdf_path,
        pdf_url=pdf_url,
        abs_url=abs_url,
        published=published,
        updated=updated,
        categories=categories,
        primary_category=primary_category,
        doi=doi or None,
        workdir=workdir,
    )
    return paper


def _extract_entry(feed: bytes) -> Optional[ElementTree.Element]:
    try:
        tree = ElementTree.fromstring(feed)
    except ElementTree.ParseError as exc:  # pragma: no cover - defensive logging
        raise ValueError("Failed to parse arXiv metadata feed") from exc
    return tree.find(f"{_ATOM_NS}entry")


def _get_text(element: ElementTree.Element, name: str) -> str:
    node = element.find(f"{_ATOM_NS}{name}" if not name.startswith("{") else name)
    if node is not None and node.text:
        return node.text.strip()
    return ""


def _extract_authors(entry: ElementTree.Element) -> list[str]:
    names: list[str] = []
    for author in entry.findall(f"{_ATOM_NS}author"):
        name_node = author.find(f"{_ATOM_NS}name")
        if name_node is not None and name_node.text:
            names.append(name_node.text.strip())
    return names


def _extract_categories(entry: ElementTree.Element) -> Iterable[str]:
    for category in entry.findall(f"{_ATOM_NS}category"):
        term = category.get("term")
        if term:
            yield term


def _extract_pdf_url(entry: ElementTree.Element) -> Optional[str]:
    for link in entry.findall(f"{_ATOM_NS}link"):
        if link.get("type") == "application/pdf":
            href = link.get("href")
            if href:
                return href
    return None


def _get_attribute(entry: ElementTree.Element, tag: str, attribute: str) -> Optional[str]:
    node = entry.find(tag)
    if node is not None:
        return node.get(attribute)
    return None


def _parse_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    candidate = candidate.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None
