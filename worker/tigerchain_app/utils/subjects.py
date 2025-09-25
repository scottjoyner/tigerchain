from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set


DEFAULT_SUBJECT_KEYWORDS: Dict[str, Sequence[str]] = {
    "compliance": ("policy", "regulation", "gdpr", "sox", "hipaa", "compliance"),
    "security": ("encryption", "vulnerability", "incident", "breach", "security"),
    "product": ("roadmap", "feature", "release", "product"),
    "engineering": ("architecture", "api", "deployment", "infrastructure", "design"),
    "support": ("faq", "troubleshoot", "support", "issue", "ticket"),
    "sales": ("prospect", "pipeline", "deal", "sales", "pricing"),
    "people": ("hiring", "benefit", "culture", "people", "hr"),
}


class SubjectClassifier:
    """Lightweight keyword classifier used for subject tagging and routing."""

    def __init__(self, keyword_map: Dict[str, Sequence[str]] | None = None) -> None:
        self.keyword_map = {
            key.strip().lower(): tuple({kw.strip().lower() for kw in values if kw})
            for key, values in (keyword_map or DEFAULT_SUBJECT_KEYWORDS).items()
        }

    def classify(self, text: str, existing_tags: Iterable[str] | None = None) -> List[str]:
        text_lower = text.lower()
        hits: Set[str] = set(tag.strip().lower() for tag in existing_tags or [] if tag)
        for subject, keywords in self.keyword_map.items():
            if any(keyword in text_lower for keyword in keywords):
                hits.add(subject)
        if not hits:
            hits.add("general")
        return sorted(hits)

    def build_collections(self, tags: Iterable[str]) -> List[dict]:
        normalised = [tag.strip().lower() for tag in tags if tag]
        counts = Counter(normalised)
        collections: List[dict] = []
        for tag, count in counts.most_common():
            collections.append(
                {
                    "name": tag,
                    "display_name": tag.replace("_", " ").title(),
                    "document_count": count,
                }
            )
        return collections

    def priorities_from_categories(self, categories: Iterable[str] | None) -> Dict[str, float]:
        priorities: Dict[str, float] = {}
        for category in categories or []:
            if not isinstance(category, str):
                continue
            tag = category.strip().lower()
            if not tag:
                continue
            priorities[tag] = max(priorities.get(tag, 0.0), 0.6)
            for subject, keywords in self.keyword_map.items():
                if tag in keywords:
                    priorities[subject] = max(priorities.get(subject, 0.0), 0.8)
        return {key: round(value, 3) for key, value in priorities.items()}


__all__ = ["SubjectClassifier", "DEFAULT_SUBJECT_KEYWORDS"]
